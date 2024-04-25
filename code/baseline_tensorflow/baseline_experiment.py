"""Example experiment for the 2 mismatched segments dilation model."""
import glob
import json
import logging
import os, sys
import tensorflow as tf
import keras

import sys

# add base path to sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from baseline_model import dilation_model

from dataset_generator import DataGenerator, batch_equalizer_fn, create_tf_dataset


def evaluate_model(model, test_dict):
    """Evaluate a model.

    Parameters
    ----------
    model: tf.keras.Model
        Model to evaluate.
    test_dict: dict
        Mapping between a subject and a tf.data.Dataset containing the test
        set for the subject.

    Returns
    -------
    dict
        Mapping between a subject and the loss/evaluation score on the test set
    """
    evaluation = {}
    for subject, ds_test in test_dict.items():
        logging.info(f"Scores for subject {subject}:")
        results = model.evaluate(ds_test, verbose=2)
        metrics = model.metrics_names
        evaluation[subject] = dict(zip(metrics, results))
    return evaluation


if __name__ == "__main__":
    # Parameters
    # Length of the decision window
    window_length_s = 5
    fs = 64

    window_length = window_length_s * fs  # 5 seconds
    # Hop length between two consecutive decision windows
    hop_length = 64

    epochs = 2
    patience = 5
    batch_size = 16
    only_evaluate = False
    number_mismatch = 4

    training_log_filename = "training_log_{}_{}.csv".format(number_mismatch, window_length_s)

    # Get the path to the config file
    experiments_folder = os.path.dirname(__file__)
    task_folder = os.path.dirname(experiments_folder)
    config_path = os.path.join(experiments_folder, 'config.json')

    # Load the config
    with open(config_path) as fp:
        config = json.load(fp)

    # Provide the path of the dataset
    # which is split already to train, val, test
    data_folder = os.path.join(config["dataset_folder"], config['derivatives_folder'], config["split_folder"])

    # stimulus feature which will be used for training the model. Can be either 'envelope' ( dimension 1) or 'mel' (dimension 28)
    stimulus_features = ["envelope"]
    stimulus_dimension = 1

    # uncomment if you want to train with the mel spectrogram stimulus representation
    # stimulus_features = ["mel"]
    # stimulus_dimension = 10

    features = ["eeg"] + stimulus_features

    # Create a directory to store (intermediate) results
    results_folder = os.path.join(experiments_folder,
                                  "results_dilated_convolutional_model_{}_MM_{}_s_{}".format(number_mismatch,
                                                                                             window_length_s,
                                                                                             stimulus_features[0]))
    os.makedirs(results_folder, exist_ok=True)

    # create dilation model
    model = dilation_model(time_window=window_length, eeg_input_dimension=64, env_input_dimension=stimulus_dimension,
                           num_mismatched_segments=number_mismatch)

    model_path = os.path.join(results_folder,
                              "model_{}_MM_{}_s_{}.keras".format(number_mismatch, window_length_s,
                                                                 stimulus_features[0]))

    if only_evaluate:
        model = tf.keras.models.load_model(model_path)

    else:

        train_files = [x for x in glob.glob(os.path.join(data_folder, "train_-_*")) if
                       os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        print(features)
        # Create list of numpy array files
        train_generator = DataGenerator(train_files, window_length)
        import pdb

        dataset_train = create_tf_dataset(train_generator, window_length, batch_equalizer_fn,
                                          hop_length, batch_size,
                                          number_mismatch=number_mismatch,
                                          data_types=(tf.float32, tf.float32),
                                          feature_dims=(64, stimulus_dimension))

        # Create the generator for the validation set
        val_files = [x for x in glob.glob(os.path.join(data_folder, "val_-_*")) if
                     os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
        val_generator = DataGenerator(val_files, window_length)
        dataset_val = create_tf_dataset(val_generator, window_length, batch_equalizer_fn,
                                        hop_length, batch_size,
                                        number_mismatch=number_mismatch,
                                        data_types=(tf.float32, tf.float32),
                                        feature_dims=(64, stimulus_dimension))

        # Train the model
        model.fit(
            dataset_train,
            epochs=epochs,
            validation_data=dataset_val,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
                tf.keras.callbacks.CSVLogger(os.path.join(results_folder, training_log_filename)),
                tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True),
            ],
        )

    test_window_lengths = [2, 4]
    number_mismatch_test = [2, 3, 4, 8]
    for number_mismatch in number_mismatch_test:
        for window_length_s in test_window_lengths:
            window_length = window_length_s * fs
            results_filename = 'eval_{}_{}_s.json'.format(number_mismatch, window_length_s)

            model = dilation_model(time_window=window_length, eeg_input_dimension=64,
                                   env_input_dimension=stimulus_dimension, num_mismatched_segments=number_mismatch)

            model.load_weights(model_path)
            # Evaluate the model on test set
            # Create a dataset generator for each test subject
            test_files = [x for x in glob.glob(os.path.join(data_folder, "test_-_*")) if
                          os.path.basename(x).split("_-_")[-1].split(".")[0] in features]
            # Get all different subjects from the test set
            subjects = list(set([os.path.basename(x).split("_-_")[1] for x in test_files]))
            datasets_test = {}
            # Create a generator for each subject
            for sub in subjects:
                files_test_sub = [f for f in test_files if sub in os.path.basename(f)]
                test_generator = DataGenerator(files_test_sub, window_length)
                datasets_test[sub] = create_tf_dataset(test_generator, window_length, batch_equalizer_fn,
                                                       hop_length, batch_size=1,
                                                       number_mismatch=number_mismatch,
                                                       data_types=(tf.float32, tf.float32),
                                                       feature_dims=(64, stimulus_dimension))

            evaluation = evaluate_model(model, datasets_test)

            # We can save our results in a json encoded file
            results_path = os.path.join(results_folder, results_filename)
            with open(results_path, "w") as fp:
                json.dump(evaluation, fp)
            logging.info(f"Results saved at {results_path}")
