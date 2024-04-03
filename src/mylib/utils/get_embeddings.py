import gzip
import os
import json
import glob
import librosa
import torch
from tqdm import tqdm
import numpy as np
from scipy.io.wavfile import write

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

experiment_folder = os.path.dirname(os.path.abspath(''))
with open(os.path.join(experiment_folder, "utils/config.json")) as file_path:
    config = json.load(file_path)

path_to_stimuli = os.path.join(config["dataset_folder"], config["stimuli"])
path_to_audio = os.path.join(experiment_folder, "../../code/audio")
os.makedirs(path_to_audio, exist_ok=True)

stimuli = [x for x in glob.glob(os.path.join(path_to_stimuli, "*.npz.gz"))]

print("Generating Audio")
for stimulus in tqdm(stimuli):
    path_to_save = os.path.join(path_to_audio, f"{os.path.basename(stimulus)[:-7]}.wav")
    if os.path.exists(path_to_save):
        continue
    with gzip.open(stimulus, "rb") as f:
        data = dict(np.load(f))
        data = {
            'data': data['audio'],
            'sr': data['fs']
        }
        rate = data['sr']
        scaled = np.int16(data['data'] / np.max(np.abs(data['data'])) * 32767)
        write(path_to_save, rate, scaled)

print("Getting embeddings")
path_to_embedds = os.path.join(experiment_folder, "../../code/embeddings")
os.makedirs(path_to_embedds, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
for asr_model_name in ["Clementapa/wav2vec2-base-960h-phoneme-reco-dutch"]:
    if "wav2vec" in asr_model_name:
        path_asr_embedds = os.path.join(path_to_embedds, "wav2vec")
        os.makedirs(path_asr_embedds, exist_ok=True)
        print("Wav2Vec Embeddings")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(asr_model_name)
    asr_model = Wav2Vec2Model.from_pretrained(asr_model_name).to(device)

    for audio in tqdm(glob.glob(os.path.join(path_to_audio, "*.wav"))):
        if os.path.exists(os.path.join(path_asr_embedds, os.path.basename(audio))):
            continue
        input_audio, sr = librosa.load(audio, sr=16000)
        sr = int(sr)
        embed = []
        for j in range(0, len(input_audio), sr):
            part = input_audio[j: j + sr]
            i = feature_extractor(part, return_tensors='pt', sampling_rate=sr).to(device)

            with torch.no_grad():
                output = asr_model(i.input_values)
            embed.append(output.last_hidden_state.cpu().numpy())
        embed = np.concatenate(embed, axis=1)
        np.save(os.path.join(path_asr_embedds, os.path.basename(audio)), embed)
