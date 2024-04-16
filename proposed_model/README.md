- `models.py` --- модель для эксперимента. По умолчанию инициализируется базовая модель. При указании `use_transformers=True` инициализируется базовая модель с трансформером, а при `use_embeddings=True` --- модель который использует эмбеддинги стимулов.
- `eeg_encoder.py`, `stimulus_encoders.py` --- энкодеры для ЭЭГ и стимулов. 
- `train.py` --- базовый класс `Trainer` для обучения модели.
- `data.py` --- файл с описанием наследника класса `torch.utils.data.Dataset`ю
- `get_embeddings.py`, `reduce_emb_dim.py`, `align_embeddings.py` --- скрипты для подготовки эмбеддингов.

Пример запуска модели смотрите в `code/main.ipynb`.