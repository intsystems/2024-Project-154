import torch.nn as nn


class BaselineStimulusEncoder(nn.Module):
    """Энкодер стимула взятый из базового решения"""

    def __init__(self, dilation_filters=16, kernel_size=3, layers=3):
        super(BaselineStimulusEncoder, self).__init__()

        self.env_convos = nn.Sequential()
        for layer_index in range(layers):
            self.env_convos.add_module(f"conv1d_lay{layer_index}",
                                       nn.Conv1d(
                                           in_channels=dilation_filters * (layer_index != 0) + (layer_index == 0),
                                           out_channels=dilation_filters,
                                           kernel_size=kernel_size,
                                           dilation=kernel_size ** layer_index,
                                           bias=True))
            self.env_convos.add_module(f"relu_lay{layer_index}", nn.ReLU())

    def forward(self, stimulus):
        return self.env_convos(stimulus)


class PhysicsInformedStimulusEncoder(nn.Module):
    """Физико-информированный энкодер для стимула"""

    def __init__(self, *args, **kwargs):
        super(PhysicsInformedStimulusEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, bias=True),
            nn.ReLU()
        )

    def forward(self, embedding):
        return self.encoder(embedding)
