from email.mime import audio
import os 
import json 
import glob
import numpy as np
from mne.filter import resample
from tqdm import tqdm

file = os.path.abspath('')
experiment_folder = os.path.dirname(file)

# Load the config file
with open(os.path.join(file, "src/mylib/utils/config.json")) as file_path:
    config = json.load(file_path)

# Path to the dataset, which is already split to train, val, test
data_folder = os.path.join(config["dataset_folder"], config['derivatives_folder'], config["preprocessed_stimuli_folder"])

train_files = [x for x in glob.glob(os.path.join(data_folder, "*.npy"))]

# resampling wav2vec embeddings
print("Resampling Wav2Vec embeddings")
path_to_embedds = os.path.join(file, "code/embeddings/wav2vec_resampled")
os.makedirs(path_to_embedds, exist_ok=True)

processed = set()
pairs = []
for x in tqdm(train_files):
    audio_name = "_".join(os.path.basename(x).split("_")[:-1])
    if audio_name in processed:
       continue
    for emb_f in glob.glob(os.path.join(file, "code/embeddings/wav2vec/*")):
      f_name = os.path.basename(emb_f).split(".")[0]
      if audio_name == f_name:
         pairs.append([x, emb_f])
    processed.add(audio_name)

for p in tqdm(pairs):
    audio_name = "_".join(os.path.basename(p[0]).split("_")[:-1])
    stimul = np.load(p[0])
    emb = np.load(p[1]).astype(np.float64).squeeze(axis=0)
    resampled = resample(emb, stimul.shape[0]/emb.shape[0], axis=0)
    assert resampled.shape[0] == stimul.shape[0]
    print(stimul.shape, resampled.shape)
    np.save(os.path.join(path_to_embedds, f"{audio_name}_resampled"), resampled)

# resampling whisper embeddings
print("Resampling Whisper embeddings")

path_to_embedds = os.path.join(file, "code/embeddings/whisper_resampled")
os.makedirs(path_to_embedds, exist_ok=True)

processed = set()
pairs = []
for x in tqdm(train_files):
    audio_name = "_".join(os.path.basename(x).split("_")[:-1])
    if audio_name in processed:
       continue
    for emb_f in glob.glob(os.path.join(file, "code/embeddings/whisper/*")):
      f_name = os.path.basename(emb_f).split(".")[0]
      if audio_name == f_name:
         print(emb_f)
         pairs.append([x, emb_f])
    processed.add(audio_name)

for p in tqdm(pairs):
    audio_name = "_".join(os.path.basename(p[0]).split("_")[:-1])
    stimul = np.load(p[0])
    emb = np.load(p[1]).astype(np.float64).squeeze(axis=0)
    resampled = resample(emb, stimul.shape[0]/emb.shape[0], axis=0)
    assert resampled.shape[0] == stimul.shape[0]
    print(stimul.shape, resampled.shape)
    np.save(os.path.join(path_to_embedds, f"{audio_name}_resampled"), resampled)