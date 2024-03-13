# Запус эксперимента

Загрузить данные из репозитория https://github.com/exporl/auditory-eeg-dataset/tree/master.

В файле `config.json` меняем `--absolute path to dataset folder--` на абсолютный путь к датасету. 

Важно согласно репозиторию датасета предварительно разделить данные на тренировочные, валидационные и тестовые. 

Запуск эксперимента: `python3 baseline_experiment.py`