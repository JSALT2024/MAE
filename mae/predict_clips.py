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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_to_h5(features_list_h5, label, index_dataset, chunk_batch, chunk_size):
    if index_dataset == chunk_batch * chunk_size:
        chunk_batch += 1
        features_list_h5.resize(chunk_batch * chunk_size, axis=0)
    features_list_h5[index_dataset:index_dataset + chunk_size] = label
    index_dataset += chunk_size
    return index_dataset, chunk_batch


def add_to_h5(clip_name, clip_features, index_dataset, chunk_batch, chunk_size):
    features_list_h5 = video_h5.create_dataset(clip_name, shape=(len(clip_features),), maxshape=(None,), dtype=dt)
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

    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--clip_folder', type=str)
    parser.add_argument('--checkpoint_path', default="", type=str)
    parser.add_argument('--arch', default='vit_base_patch16', type=str)
    parser.add_argument('--num_splits', type=int)
    parser.add_argument('--split', type=int)

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    output_file_name = f'mae_features-{args.split}.h5'
    meta_file_name = f"mae_features-{args.split}.json"

    # load clip paths
    clip_names = os.listdir(args.clip_folder)
    clip_names = [file for file in clip_names if ".mp4" in file]

    # group clips based on the video name
    clip_names_sorted = np.sort(clip_names)
    video_to_clips = {}
    for file in clip_names_sorted:
        file = os.path.join(args.clip_folder, file)
        name = os.path.basename(file)
        name_split = name.split(".")[:-1]
        video_name = ".".join(name_split[:-1])
        if video_name in video_to_clips:
            video_to_clips[video_name].append(file)
        else:
            video_to_clips[video_name] = [file]

    # split to chunks
    num_samples = len(video_to_clips)
    batch_size = int(np.ceil(num_samples / (args.num_splits)))
    idxs = get_batch_idx(num_samples, batch_size)
    start, end = idxs[args.split]
    video_names = list(video_to_clips.keys())[start:end]
    print(idxs)
    print(f"Number of splits: {len(idxs)}")
    print(f"Number of videos: {len(video_names)}")

    # load model
    model = predict_mae.create_mae_model(args.arch, args.checkpoint_path)
    model = model.to(device)

    # h5py file initialization
    f_out = h5py.File(os.path.join(args.output_folder, output_file_name), 'w')
    # special data type for numpy array with variable length
    dt = h5py.vlen_dtype(np.dtype('float16'))

    # predict
    prediction_times = []
    frames = []
    metadata = {}
    for video_idx, video_name in enumerate(video_names):
        clip_paths = video_to_clips[video_name]
        metadata[video_name] = args.split
        video_h5 = f_out.create_group(video_name)
        start_time = time.time()
        for clip_path in clip_paths:
            # parse name
            name = os.path.basename(clip_path)
            name_split = name.split(".")[:-1]
            clip_name = ".".join(name_split)

            # predict video features
            video, fps = load_video_cv(clip_path)
            batches = batch_data(video, 16)
            video_features = []
            for batch in batches:
                features = predict_mae.mae_predict(batch, model, predict_mae.transform_mae, device)
                video_features.append(features)
            clip_features = np.concatenate(video_features, 0)

            # save features in hd5
            add_to_h5(
                clip_name,
                clip_features,
                index_dataset=0,
                chunk_batch=1,
                chunk_size=len(clip_features)
            )

        # print stats
        end_time = time.time()
        prediction_times.append(end_time - start_time)
        frames.append(len(video))

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

        with open(os.path.join(args.output_folder, "mae_features.json"), "w") as f:
            json.dump(merged_data, f)
