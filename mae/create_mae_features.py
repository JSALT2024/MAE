import os
import cv2
import h5py
import torch
import predict_mae
import time
import json
import numpy as np
import datetime
import argparse
import pandas as pd
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_to_h5(features_list_h5, label, index_dataset, chunk_batch, chunk_size):
    if index_dataset == chunk_batch * chunk_size:
        chunk_batch += 1
        features_list_h5.resize(chunk_batch * chunk_size, axis=0)
    features_list_h5[index_dataset:index_dataset + chunk_size] = label
    index_dataset += chunk_size
    return index_dataset, chunk_batch


def add_to_h5(clip_name, clip_features, index_dataset, chunk_batch, chunk_size):
    feature_shape = clip_features.shape
    features_list_h5 = video_h5.create_dataset(
        clip_name,
        shape=feature_shape,
        maxshape=(None, feature_shape[-1]),
        dtype=np.dtype('float16')
    )
    num_full_chunks = len(clip_features) // chunk_size
    last_chunk_size = len(clip_features) % chunk_size
    for c in range(num_full_chunks):
        feature = clip_features[index_dataset:index_dataset + chunk_size]
        index_dataset, chunk_batch = save_to_h5(features_list_h5, feature, index_dataset, chunk_batch,
                                                chunk_size)
    if last_chunk_size > 0:
        feature = clip_features[index_dataset:index_dataset + last_chunk_size]
        index_dataset, chunk_batch = save_to_h5(features_list_h5, feature, index_dataset, chunk_batch,
                                                last_chunk_size)


def get_batch_idx(num_samples, batch_size):
    steps = int(np.floor(num_samples / batch_size))
    idx = []
    for i in range(1, steps + 1):
        start = (i - 1) * batch_size
        end = i * batch_size
        idx.append((start, end))

    if steps * batch_size != num_samples:
        idx.append((steps * batch_size, num_samples))
    return idx


def batch_data(images, batch_size):
    batched_data = []
    batch_idxs = get_batch_idx(len(images), batch_size)
    for start_idx, end_idx in batch_idxs:
        batched_data.append(images[start_idx:end_idx])
    return batched_data


def load_video_cv(path: str):
    video = []

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret = True
    while ret:
        ret, img = cap.read()
        if ret:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            video.append(img)
    cap.release()
    return video, fps


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)

    parser.add_argument('--input_folder', type=str, help='Path to folder with clips.')
    parser.add_argument('--output_folder', type=str, help='Path to folder where to save features.')
    parser.add_argument('--dataset_name', type=str, help="Name of the dataset. Used only for naming of the "
                                                         "output file.")
    parser.add_argument('--split_name', default="train", type=str, help="Name of the data subset examples: dev, "
                                                                        "train, test. Used only for naming of the "
                                                                        "output file.")

    parser.add_argument('--annotation_file', default="", type=str, help="If the name is not in the format: "
                                                                        "'video_name.time_stamp.mp4' and can't be "
                                                                        "parsed, annotation file with: SENTENCE_NAME "
                                                                        "and VIDEO_ID columns should be provided.")

    parser.add_argument('--checkpoint', default="", type=str, help="Path to pretrained weights.")
    parser.add_argument('--arch', default='vit_base_patch16', type=str, help="Architecture of the model.")
    parser.add_argument('--num_splits', type=int, help="Number of splits/shards dataset will be split into.")
    parser.add_argument('--split', type=int, help="Index of the split/shard.")


    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    output_file_name = f'{args.dataset_name}.mae.{args.split_name}.{args.split}.h5'
    meta_file_name = f"{args.dataset_name}.mae.{args.split_name}.{args.split}.json"
    os.makedirs(args.output_folder, exist_ok=True)

    # prepare mapping between clip names and video names
    video_to_clips = defaultdict(list)
    clip_names = os.listdir(args.input_folder)
    clip_names = [file for file in clip_names if file.endswith(".mp4")]
    clip_names.sort()
    for idx in range(len(clip_names)):
        name_split = clip_names[idx].split(".")[:-1]
        clip_names[idx] = ".".join(name_split)

    if args.annotation_file:
        annotations = pd.read_csv(args.annotation_file, sep='\t')
        _clip_to_video = dict(zip(annotations.SENTENCE_NAME, annotations.VIDEO_ID))
        for clip_name in clip_names:
            video_to_clips[_clip_to_video[clip_name]].append(clip_name)
    else:
        for clip_name in clip_names:
            name_split = clip_name.split(".")[:-1]
            video_name = ".".join(name_split)
            video_to_clips[video_name].append(clip_name)
    video_to_clips = dict(video_to_clips)

    # split to chunks
    num_samples = len(video_to_clips)
    batch_size = int(np.ceil(num_samples / (args.num_splits)))
    idxs = get_batch_idx(num_samples, batch_size)
    start, end = idxs[args.split]
    video_names = list(video_to_clips.keys())
    video_names.sort()
    video_names = video_names[start:end]
    print(idxs)
    print(f"Number of splits: {len(idxs)}")
    print(f"Number of videos: {len(video_names)}")

    # load model
    model = predict_mae.create_mae_model(args.arch, args.checkpoint)
    model = model.to(device)

    # h5py file initialization
    f_out = h5py.File(os.path.join(args.output_folder, output_file_name), 'w')

    # predict
    prediction_times = []
    frames = []
    metadata = {}
    for video_idx, video_name in enumerate(video_names):
        clip_names = video_to_clips[video_name]
        metadata[video_name] = args.split
        video_h5 = f_out.create_group(video_name)
        start_time = time.time()
        for clip_name in clip_names:
            clip_path = os.path.join(args.input_folder, clip_name)

            # predict video features
            video, fps = load_video_cv(clip_path)
            if len(video) == 0:
                print("FAILED:", clip_path)
                continue
            batches = batch_data(video, 16)
            features = []
            for batch in batches:
                _features = predict_mae.mae_predict(batch, model, predict_mae.transform_mae, device)
                features.append(_features)
            features = np.concatenate(features, 0)

            # save features in hd5
            add_to_h5(
                clip_name,
                features,
                index_dataset=0,
                chunk_batch=1,
                chunk_size=len(features)
            )
            frames.append(len(video))

        # print stats
        end_time = time.time()
        prediction_times.append(end_time - start_time)

        print(f"[{video_idx + 1}/{len(video_names)}]")
        print(f"average time: {np.mean(prediction_times):.3f}")
        print(f"average frames: {np.mean(frames):.2f}")
        secs = ((len(video_names) - (video_idx + 1)) * np.mean(prediction_times))
        print(f"eta: {str(datetime.timedelta(seconds=secs)).split('.')[0]}")
        print()

    with open(os.path.join(args.output_folder, meta_file_name), "w") as f:
        json.dump(metadata, f)

    f_out.close()

    # merge json files
    json_files = [name for name in os.listdir(args.output_folder) if ".json" in name]
    if args.num_splits == len(json_files):
        merged_data = {}
        for name in json_files:
            path = os.path.join(args.output_folder, name)
            with open(path, "r") as f:
                data = json.load(f)
            merged_data.update(data)

        with open(os.path.join(args.output_folder, f"{args.dataset_name}.mae.{args.split_name}.json"), "w") as f:
            json.dump(merged_data, f)
