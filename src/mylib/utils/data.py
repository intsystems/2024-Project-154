import torch
import numpy as np
import itertools
import os
from torch.utils.data import Dataset


class TaskDataset(Dataset):
    """Generate data for the Match/Mismatch task."""

    def __init__(self, files, window_length, hop_length, number_of_mismatch, max_files=100):
        self.labels = dict()
        assert number_of_mismatch != 0
        self.window_length = window_length
        self.hop_length = hop_length
        self.number_of_mismatch = number_of_mismatch
        self.files = files
        self.max_files = max_files
        self.group_recordings()
        self.frame_recordings()
        self.create_imposter_segments()
        self.create_labels_randomize_positions()

    def group_recordings(self):
        new_files = []
        grouped = itertools.groupby(sorted(self.files), lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))

        for recording_name, feature_paths in grouped:
            sub_recordings = sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)
            eeg, envelope = np.load(sub_recordings[0]), np.load(sub_recordings[1])  # eeg [L, C], env [L, 1]
            new_files += [[torch.tensor(eeg.T).float(), torch.tensor(envelope.T).float()]]

            if self.max_files is not None and len(new_files) == self.max_files:
                break

        self.files = new_files

    def frame_recordings(self):
        new_files = []
        for i in range(len(self.files)):
            self.files[i][0] = self.files[i][0].unfold(
                1, self.window_length, self.hop_length).transpose(0, 1)  # [num_of_frames, C, window_length]
            self.files[i][1] = self.files[i][1].unfold(
                1, self.window_length, self.hop_length).transpose(0, 1)  # [num_of_frames, C, window_length]
            eegs = list(torch.tensor_split(self.files[i][0], self.files[i][0].shape[0], dim=0))
            envs = list(torch.tensor_split(self.files[i][1], self.files[i][1].shape[0], dim=0))
            for eeg, env in zip(eegs, envs):
                new_files.append([eeg.squeeze(), env.squeeze(dim=0)])
        self.files = new_files

    def create_imposter_segments(self):
        for i in range(len(self.files)):
            for _ in range(self.number_of_mismatch):
                t = self.files[i][-1].view(-1)
                t = t[torch.randperm(t.shape[-1])].view(self.files[i][-1].shape)
                self.files[i].append(t)

    def create_labels_randomize_positions(self):
        roll = lambda x, n: x[-n % len(x):] + x[: -n % len(x)]
        for i in range(len(self.files)):
            self.labels[i] = torch.tensor(0)
            for j in range(1, self.number_of_mismatch + 1):
                envs = self.files[i][1:]
                rolled_envs = roll(envs, j)
                self.files.append([self.files[i][0], *rolled_envs])
                self.labels[len(self.files) - 1] = torch.tensor(j)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx], self.labels[idx]
