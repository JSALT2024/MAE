import os
import random

import decord
from decord import VideoReader
from torch.utils.data import Dataset

if VideoReader is None:
    raise ImportError("Unable to import `decord` which is required to read videos.")


class Asl_Dataset(Dataset):

    def __init__(self, data_path, transform=None, seed=42):
        self.data_path = data_path
        self.data = []
        self.seed = seed
        self.transform = transform

        random.seed(seed)

        for video_name in os.listdir(data_path):
            if not video_name.endswith("mp4"):
                continue 
            video_path = os.path.join(data_path, video_name)
            self.data.append(video_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path = self.data[idx]
        decord_vr = decord.VideoReader(video_path, num_threads=1)
        duration = len(decord_vr)
        rand_id = random.randrange(duration)

        frame = decord_vr.get_batch([rand_id]).asnumpy().squeeze()

        if self.transform is not None:
            frame = self.transform(frame)

        # image, class (ensures compatibility with pretrain engine)
        return frame, -1

