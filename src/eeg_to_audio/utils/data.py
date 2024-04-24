import torch
import numpy as np
import itertools
import os
import glob
from torch.utils.data import Dataset

project_path = os.path.abspath("eeg_to_audio/utils")


class TaskDataset(Dataset):
    """Класс для подготовки данных"""

    def __init__(self, files, window_length, hop_length, number_of_mismatch, use_embeddings=False, embedding_type=None,
                 max_files=None):
        self.labels = dict()
        self.window_length = window_length
        self.hop_length = hop_length
        self.number_of_mismatch = number_of_mismatch
        self.files = files
        self.max_files = max_files

        # Препроцессинг данных. Требуется пройтись окном по ЭЭГ и стимулам и получить пары. Для этих пар
        # потом подберем ложные стимулы, взяв из других пар
        self.group_recordings(use_embeddings, embedding_type)
        self.frame_recordings()
        self.create_imposter_segments()
        self.create_labels_randomize_positions()

    def group_recordings(self, use_embeddings, embedding_type):
        """Выделим соответствующие пары ЭЭГ-стимулы"""
        new_files = []
        grouped = itertools.groupby(sorted(self.files), lambda x: "_-_".join(os.path.basename(x).split("_-_")[:3]))

        for recording_name, feature_paths in grouped:
            eeg_path, envelope_path = sorted(feature_paths, key=lambda x: "0" if x == "eeg" else x)

            if use_embeddings:
                # Найдем соответствующий эмбеддинг
                envelope_name = os.path.basename(envelope_path).split("_-_")[2]
                for emb_path in glob.glob(
                        os.path.join(f"{project_path}/embeddings", f"{embedding_type}_resampled/*.npy")):
                    audio_name = "_".join(os.path.basename(emb_path).split("_")[:-1])
                    if audio_name == envelope_name:
                        envelope_path = emb_path
                        break
            eeg, envelope = np.load(eeg_path), np.load(envelope_path)  # eeg [L, C], env [L, 1]
            new_files += [[torch.tensor(eeg.T).float(), torch.tensor(envelope.T).float()]]

            if self.max_files is not None and len(new_files) == self.max_files:
                break
        self.files = new_files

    def frame_recordings(self):
        """Пройдемся скользящим окном по записям"""
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
        """Сгенерируем ложные стимулы, number_of_mismatch раз выберем случайно стимулы из других записи, в дополнение
        к истинному"""
        for i in range(len(self.files)):
            # Сгенерируем индексы для получения ложных стимулов. Чтобы истинный стимул не вошел два раза, убедимся, что
            # индекс i не содержится в сгенерированном наборе индексов.
            indices = np.random.choice(np.arange(len(self.files)), size=self.number_of_mismatch, replace=False)
            while i in indices:
                indices = np.random.choice(np.arange(len(self.files)), size=self.number_of_mismatch, replace=False)

            for idx in indices:
                self.files[i].append(self.files[idx][1])

    def __roll(self, x, n):
        return x[-n % len(x):] + x[: -n % len(x)]

    def create_labels_randomize_positions(self):
        """Создадим метки и объекты, которые содержат истинный стимул в других позициях"""
        for i in range(len(self.files)):
            self.labels[i] = torch.tensor(0)
            for j in range(1, self.number_of_mismatch + 1):
                envs = self.files[i][1:]
                rolled_envs = self.__roll(envs, j)
                self.files.append([self.files[i][0], *rolled_envs])
                self.labels[len(self.files) - 1] = torch.tensor(j)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return self.files[idx], self.labels[idx]
