import tensorflow as tf
from tensorflow.keras import activations, layers, Model


class EncoderDecoder(Model):

    def __init__(
            self,
            input_length,
            input_vocab_size,
            input_embedding_size,
            context_vector_size,
            output_length,
            output_vocab_size,
            output_embedding_size,
            enable_masking=False,
    ):
        super(EncoderDecoder, self).__init__()
        self.embedding = layers.Embedding(
            input_dim=input_vocab_size,
            output_dim=input_embedding_size,
            input_length=input_length,
            mask_zero=enable_masking,
        )
        self.encoder = Encoder(
            units=context_vector_size,
        )
        self.repeater = layers.RepeatVector(output_length)
        self.decoder = Decoder(
            units=output_embedding_size,
        )
        dense_layer = layers.Dense(
            units=output_vocab_size,
            activation='softmax',
        )
        self.dense = layers.TimeDistributed(layer=dense_layer)

    def call(self, inputs):
        encoder_inputs = self.embedding(inputs=inputs)
        mask = self.embedding.compute_mask(inputs=inputs)
        context_vector = self.encoder(inputs=encoder_inputs, mask=mask)
        decoder_inputs = self.repeater(inputs=context_vector)
        dense_inputs = self.decoder(inputs=decoder_inputs)
        model_outputs = self.dense(inputs=dense_inputs)
        return model_outputs


class Encoder(layers.Bidirectional):

    def __init__(self, units):
        cell = RNNCell(units=units)
        layer = layers.RNN(cell=cell)
        super(Encoder, self).__init__(layer=layer)

    def get_config(self):
        return dict(
            super().get_config(),
            units=self.units,
        )


class Decoder(layers.Bidirectional):

    def __init__(self, units):
        cell = RNNCell(units=units)
        layer = layers.RNN(cell=cell, return_sequences=True)
        super(Decoder, self).__init__(layer=layer)

    def get_config(self):
        return dict(
            super().get_config(),
            units=self.units,
        )


class RNNCell(layers.Layer):

    def __init__(self, units, activation='tanh', **kwargs):
        super(RNNCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.output_size = units
        self.activation = activations.get(activation)

    def build(self, input_shape):
        input_weights_init = tf.initializers.glorot_uniform()(
            shape=(input_shape[-1], self.units),
            dtype='float32',
        )
        self.input_weights = tf.Variable(
            initial_value=input_weights_init,
            trainable=True,
        )
        state_weights_init = tf.initializers.glorot_uniform()(
            shape=(self.units, self.units),
            dtype='float32',
        )
        self.state_weights = tf.Variable(
            initial_value=state_weights_init,
            trainable=True,
        )
        bias_init = tf.initializers.zeros()(
            shape=(self.units,),
            dtype='float32',
        )
        self.bias = tf.Variable(
            initial_value=bias_init,
            trainable=True,
        )
        self.built = True

    def call(self, inputs, states):
        input_term = tf.matmul(inputs, self.input_weights)
        state_term = tf.matmul(states[0], self.state_weights)
        outputs = self.activation(input_term + state_term + self.bias)
        new_states = [outputs]
        return outputs, new_states

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if batch_size is None:
            batch_size = inputs[0]
        if dtype is None:
            dtype = inputs.dtype
        return tf.zeros(
            shape=(batch_size, self.units),
            dtype=dtype,
        )

    def get_config(self):
        return dict(
            super().get_config(),
            units=self.units,
            activation=self.activation,
        )
