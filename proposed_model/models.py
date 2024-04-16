import torch
import torch.nn as nn
from src.mylib.models.eeg_encoders import BaselineEEGEncoder, MultiheadAttentionEEGEncoder
from src.mylib.models.stimulus_encoders import BaselineStimulusEncoder, PhysicsInformedStimulusEncoder


class Model(nn.Module):
    """Модель основанная на базовом решении"""

    def __init__(self,
                 layers=3,
                 kernel_size=3,
                 spatial_filters=8,
                 dilation_filters=16,
                 use_transformer=False,
                 use_embeddings=False):
        super(Model, self).__init__()

        args = {"dilation_filters": dilation_filters, "kernel_size": kernel_size, "layers": layers}
        self.use_transformer = use_transformer
        self.use_embeddings = use_embeddings

        # Пространственное преобразование ЭЭГ + Энкодер ЭЭГ
        if use_transformer:
            self.spatial_transformation = MultiheadAttentionEEGEncoder(embed_dim=64, ff_dim=32)
            self.eeg_encoder = BaselineEEGEncoder(in_channels=64, **args)
        else:
            self.spatial_transformation = nn.Conv1d(
                in_channels=64,
                out_channels=spatial_filters,
                kernel_size=1,
                bias=True
            )
            self.eeg_encoder = BaselineEEGEncoder(in_channels=spatial_filters, **args)

        # Энкодер стимула
        if use_embeddings:
            self.stimulus_encoder = PhysicsInformedStimulusEncoder(**args)
        else:
            self.stimulus_encoder = BaselineStimulusEncoder(**args)

        self.fc = nn.Linear(in_features=dilation_filters * dilation_filters,
                            out_features=1,
                            bias=True)

    def forward(self, eeg_stimuli):
        eeg = eeg_stimuli[0]
        stimuli = eeg_stimuli[1:]
        if self.use_transformer:
            eeg = eeg.transpose(1, 2)
        eeg = self.spatial_transformation(eeg)
        if self.use_transformer:
            eeg = eeg.transpose(1, 2)
        eeg = self.eeg_encoder(eeg)

        # Общие веса для всех стимулов
        for i in range(len(stimuli)):
            stimuli[i] = self.stimulus_encoder(stimuli[i])
            stimuli[i] = stimuli[i][:, :, :eeg.shape[-1]]

        cosine_sim = []
        for stimulus in stimuli:
            cosine_sim.append(eeg @ stimulus.transpose(-1, -2))

        sim_projections = [self.fc(torch.flatten(sim, start_dim=1)) for sim in cosine_sim]
        return torch.cat(sim_projections, dim=1)
