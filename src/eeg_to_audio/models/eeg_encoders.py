import torch.nn as nn


class BaselineEEGEncoder(nn.Module):
    """Базовый энкодер для ЭЭГ из базового решения"""

    def __init__(self, in_channels=8, dilation_filters=16, kernel_size=3, layers=3):
        super(BaselineEEGEncoder, self).__init__()

        self.eeg_convos = nn.Sequential()

        for layer_index in range(layers):
            self.eeg_convos.add_module(f"conv1d_lay{layer_index}",
                                       nn.Conv1d(
                                           in_channels=dilation_filters * (layer_index != 0) + (
                                                   layer_index == 0) * in_channels,
                                           out_channels=dilation_filters,
                                           kernel_size=kernel_size,
                                           dilation=kernel_size ** layer_index,
                                           bias=True))
            self.eeg_convos.add_module(f"relu_lay{layer_index}", nn.ReLU())

    def forward(self, eeg):
        return self.eeg_convos(eeg)


class TransformerBlockEEGEncoder(nn.Module):
    """Трансформер-кодировщик"""

    def __init__(self, embed_dim, ff_dim):
        super(TransformerBlockEEGEncoder, self).__init__()

        self.mha_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=2)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
        self.layer_norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)

    def forward(self, x):
        attn_output, _ = self.mha_attention(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layer_norm1(attn_output + x)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out = self.layer_norm2(out1 + ffn_output)
        return out
