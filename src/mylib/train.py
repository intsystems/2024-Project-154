import os

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from src.mylib.utils.data import TaskDataset
from sklearn.metrics import classification_report


class Trainer(object):
    r"""Base class for all trainer."""

    def __init__(self, model, train_files, val_files, test_files, args, optimizer, loss_fn):
        r"""Constructor method

        :param train_files: path to train files
        :type train_files: list

        :param val_files: path to val files
        :type val_files: list

        :param test_files: path to test files
        :type test_files: list
        """
        self.model = model
        self.args = args
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.test_files = test_files
        self.initialize_dataloaders(train_files, val_files, test_files)

    def initialize_dataloaders(self, train_files, val_files, test_files):
        r"""Initialize dataloaders"""

        conf = {"window_length": self.args["window_length"], "hop_length": self.args["hop_length"],
                "number_of_mismatch": self.args["number_of_mismatch"], "max_files": self.args["max_files"]}
        self.train_dataloader = torch.utils.data.DataLoader(TaskDataset(train_files, **conf),
                                                            batch_size=self.args["batch_size"])
        self.val_dataloader = torch.utils.data.DataLoader(TaskDataset(val_files, **conf),
                                                          batch_size=self.args["batch_size"])
        self.test_dataloader = torch.utils.data.DataLoader(TaskDataset(test_files, **conf),
                                                           batch_size=1)

    def train_one_epoch(self, epoch_index, writer):
        r"""Train one epoch"""

        running_loss = 0
        last_loss = 0

        for i, data in enumerate(self.train_dataloader):
            inputs, labels = data

            self.optimizer.zero_grad()
            outputs = self.model(inputs[0], inputs[1:])

            # TODO: CLASSIFICATION METRIC DURING TRAINING
            # probs = (torch.nn.functional.softmax(outputs.data, dim=1) >= 0.5)
            # _, predicted = torch.max(probs.data, 1)

            loss = self.loss_fn(outputs, labels)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 100
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                x = epoch_index * len(self.train_dataloader) + i + 1
                writer.add_scalar('Loss/train', last_loss, x)
                running_loss = 0

        return last_loss

    def train_model(self, epochs, run_name):
        r""" Train models"""

        writer = SummaryWriter(f"runs/{run_name}_{self.model.__class__.__name__}")

        best_vloss = 1_000_000
        if not os.path.isdir("saved_models"):
            os.makedirs("saved_models")

        for epoch in range(epochs):
            print(f"EPOCH {epoch + 1}:")
            self.model.train()
            avg_loss = self.train_one_epoch(epoch + 1, writer)

            running_vloss = 0.0
            self.model.eval()
            with torch.no_grad():
                for i, vdata in enumerate(self.val_dataloader):
                    vinputs, vlabels = vdata
                    voutputs = self.model(vinputs[0], vinputs[1:])
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss.item()

            avg_vloss = running_vloss / (i + 1)
            print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

            writer.add_scalars("Training vs. Validation Loss",
                               {"Training": avg_loss, "Validation": avg_vloss},
                               epoch + 1)
            writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = f"saved_models/{self.model.__class__.__name__}_{epoch}"
                torch.save(self.model.state_dict(), model_path)

    def eval(self):
        r"""Evaluate model for initial validation dataset."""
        pass

    def test(self):
        r"""Evaluate model for given dataset"""

        total = 0
        self.model.eval()
        y_pred = []
        y_true = []
        subjects = list(set([os.path.basename(x).split("_-_")[1] for x in self.test_files]))
        loss_fn = nn.functional.cross_entropy
        with torch.no_grad():
            for sub in subjects:
                sub_test_files = [f for f in self.test_files if sub in os.path.basename(f)]
                test_dataloader = torch.utils.data.DataLoader(TaskDataset(sub_test_files, self.args["window_length"], self.args["hop_length"]))
                loss = 0
                correct = 0
                for inputs, label in test_dataloader:
                    outputs = self.model(inputs[0], inputs[1:])

                    loss += loss_fn(outputs, label).item()
                    probs = (torch.nn.functional.softmax(outputs.data, dim=1) >= 0.5)
                    _, predicted = torch.max()

            for data in self.test_dataloader:
                inputs, labels = data

                outputs = self.model(inputs[0], inputs[1:])
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                y_pred.extend(predicted.tolist())
                y_true.extend(labels.tolist())

                correct += (predicted == labels).sum().item()

        return classification_report(y_true, y_pred)
