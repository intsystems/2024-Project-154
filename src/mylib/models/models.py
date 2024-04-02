import torch
import torch.nn as nn
from src.mylib.models.eeg_encoders import BaselineEEGEncoder, MultiheadAttentionEEGEncoder
from src.mylib.models.stimulus_encoders import BaselineStimulusEncoder


class BaselineModel(nn.Module):
    """Baseline model"""

    def __init__(self,
                 layers=3,
                 kernel_size=3,
                 spatial_filters=8,
                 dilation_filters=16):
        super(BaselineModel, self).__init__()

        # EEG spatial transformation
        self.spatial_transformation = nn.Conv1d(
            in_channels=64,
            out_channels=spatial_filters,
            kernel_size=1,
            bias=True
        )

        args = {"dilation_filters": dilation_filters, "kernel_size": kernel_size, "layers": layers}

        # EEG encoder
        self.eeg_encoder = BaselineEEGEncoder(in_channels=spatial_filters, **args)

        # Stimulus encoder
        self.stimulus_encoder = BaselineStimulusEncoder(**args)

        self.fc = nn.Linear(in_features=dilation_filters * dilation_filters,
                            out_features=1,
                            bias=True)

    def forward(self, eeg, stimuli):
        eeg = self.spatial_transformation(eeg)
        eeg = self.eeg_encoder(eeg)

        # shared weights for stimuli
        for i in range(len(stimuli)):
            stimuli[i] = self.stimulus_encoder(stimuli[i])

        cosine_sim = []
        for stimulus in stimuli:
            cosine_sim.append(eeg @ stimulus.transpose(-1, -2))
        sim_projections = [self.fc(torch.flatten(sim, start_dim=1)) for sim in cosine_sim]
        return torch.cat(sim_projections, dim=1)


class MHAModel(nn.Module):
    """Model with transformer block as spatial transformation"""

    def __init__(self,
                 layers=3,
                 kernel_size=3,
                 dilation_filters=16):
        super(MHAModel, self).__init__()

        # EEG spatial transformation
        self.spatial_transformation = MultiheadAttentionEEGEncoder(embed_dim=64, ff_dim=32)

        args = {"dilation_filters": dilation_filters, "kernel_size": kernel_size, "layers": layers}

        # EEG encoder
        self.eeg_encoder = BaselineEEGEncoder(in_channels=64, **args)

        # Stimulus encoder
        self.stimulus_encoder = BaselineStimulusEncoder(**args)

        self.fc = nn.Linear(in_features=dilation_filters * dilation_filters,
                            out_features=1,
                            bias=True)

    def forward(self, eeg, stimuli):
        eeg = self.spatial_transformation(eeg.transpose(1, 2))
        eeg = self.eeg_encoder(eeg.transpose(1, 2))

        # shared weights for stimuli
        for i in range(len(stimuli)):
            stimuli[i] = self.stimulus_encoder(stimuli[i])

        cosine_sim = []
        for stimulus in stimuli:
            cosine_sim.append(eeg @ stimulus.transpose(-1, -2))
        sim_projections = [self.fc(torch.flatten(sim, start_dim=1)) for sim in cosine_sim]
        return torch.cat(sim_projections, dim=1)
