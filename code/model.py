import tensorflow as tf 
import os

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self,  embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim), ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dropout2 = tf.keras.layers.Dropout(0.5)

    def build(self, input_shape):
        super(TransformerBlock, self).build(input_shape)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs) 
        attn_output = self.dropout1(attn_output) 
        out1 = self.layernorm1(inputs + attn_output) 
        print(out1.shape)
        ffn_output = self.ffn(out1) 
        print(ffn_output.shape)
        ffn_output = self.dropout2(ffn_output) 
        out = self.layernorm2(out1 + ffn_output) 
        return out


def Model(
    time_window=None,
    eeg_input_dimension=64,
    env_input_dimension=1,
    layers=3,
    kernel_size=3,
    spatial_filters=8,
    dilation_filters=16,
    activation="relu",
    compile=True,
    num_mismatched_segments=2
):
    """Convolutional dilation model.

    Code was taken and adapted from
    https://github.com/exporl/eeg-matching-eusipco2020

    Parameters
    ----------
    time_window : int or None
        Segment length. If None, the model will accept every time window input
        length.
    eeg_input_dimension : int
        number of channels of the EEG
    env_input_dimension : int
        dimemsion of the stimulus representation.
        if stimulus == envelope, env_input_dimension =1
        if stimulus == mel, env_input_dimension =28
    layers : int
        Depth of the network/Number of layers
    kernel_size : int
        Size of the kernel for the dilation convolutions
    spatial_filters : int
        Number of parallel filters to use in the spatial layer
    dilation_filters : int
        Number of parallel filters to use in the dilation layers
    activation : str or list or tuple
        Name of the non-linearity to apply after the dilation layers
        or list/tuple of different non-linearities
    compile : bool
        If model should be compiled
    inputs : tuple
        Alternative inputs

    Returns
    -------
    tf.Model
        The dilation model


    References
    ----------
    Accou, B., Jalilpour Monesi, M., Montoya, J., Van hamme, H. & Francart, T.
    Modeling the relationship between acoustic stimulus and EEG with a dilated
    convolutional neural network. In 2020 28th European Signal Processing
    Conference (EUSIPCO), 1175–1179, DOI: 10.23919/Eusipco47968.2020.9287417
    (2021). ISSN: 2076-1465.

    Accou, B., Monesi, M. J., hamme, H. V. & Francart, T.
    Predicting speech intelligibility from EEG in a non-linear classification
    paradigm. J. Neural Eng. 18, 066008, DOI: 10.1088/1741-2552/ac33e9 (2021).
    Publisher: IOP Publishing
    """

    eeg = tf.keras.layers.Input(shape=[time_window, eeg_input_dimension])
    stimuli_input = [tf.keras.layers.Input(shape=[time_window, env_input_dimension]) for _ in range(num_mismatched_segments+1)]

    all_inputs = [eeg]
    all_inputs.extend(stimuli_input)


    stimuli_proj = [x for x in stimuli_input]

    # Activations to apply
    if isinstance(activation, str):
        activations = [activation] * layers
    else:
        activations = activation

    transformer_block = TransformerBlock(embed_dim=eeg_input_dimension, num_heads=2, ff_dim=32)
    eeg_proj_1 = transformer_block(eeg)
    print(eeg_proj_1.shape)

    # Construct dilation layers
    for layer_index in range(layers):
        # dilation on EEG
        eeg_proj_1 = tf.keras.layers.Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size ** layer_index,
            strides=1,
            activation=activations[layer_index],
        )(eeg_proj_1)

        # Dilation on envelope data, share weights
        env_proj_layer = tf.keras.layers.Conv1D(
            dilation_filters,
            kernel_size=kernel_size,
            dilation_rate=kernel_size ** layer_index,
            strides=1,
            activation=activations[layer_index],
        )

        stimuli_proj = [env_proj_layer(stimulus_proj) for stimulus_proj in stimuli_proj]

    # Comparison
    cos = [tf.keras.layers.Dot(1, normalize=True)([eeg_proj_1, stimulus_proj]) for stimulus_proj in stimuli_proj]

    linear_proj_sim = tf.keras.layers.Dense(1, activation="linear")

    # Linear projection of similarity matrices
    cos_proj = [linear_proj_sim(tf.keras.layers.Flatten()(cos_i)) for cos_i in cos]

    # Classification
    out = tf.keras.activations.softmax((tf.keras.layers.Concatenate()(cos_proj)))



    model = tf.keras.Model(inputs=all_inputs, outputs=[out])

    if compile:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            metrics=["accuracy"],
            loss=["categorical_crossentropy"],
        )
        print(model.summary())
    return model
