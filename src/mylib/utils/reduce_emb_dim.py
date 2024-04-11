import os 
import glob 
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

# Уменьшаем размерность до 1, как в базовом случае
pca = PCA(n_components=1) 
for emb_type in ["wav2vec", "whisper"]:
  print(f"Reducing dimensions for {emb_type} embeddings")
  for emb_path in tqdm(glob.glob(os.path.join(os.path.abspath("."), f"code/embeddings/{emb_type}_resampled/*.npy"))):
    embedding = np.load(emb_path)

    # Уменьшаем размерность    
    embedding = pca.fit_transform(embedding)

    np.save(emb_path, embedding)