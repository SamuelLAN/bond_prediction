import tensorflow as tf
import numpy as np

keras = tf.keras
layers = keras.layers


def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask zero out padding tokens.
    if mask is not None:
        logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(logits, axis=-1)

    return tf.matmul(attention_weights, value)


class MultiHeadAttention(layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = layers.Dense(units=d_model)
        self.key_dense = layers.Dense(units=d_model)
        self.value_dense = layers.Dense(units=d_model)

        self.dense = layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        outputs = self.dense(concat_attention)

        return outputs


class PositionalEncoding(layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs, start_from_end=None):
        return inputs + self.pos_encoding[:, start_from_end:tf.shape(inputs)[1], :]


def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(d_model, num_heads, name="attention")({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': padding_mask
    })
    attention = layers.Dropout(rate=dropout)(attention)
    attention = layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = layers.Dense(units=units, activation='relu')(attention)
    outputs = layers.Dense(units=d_model)(outputs)
    outputs = layers.Dropout(rate=dropout)(outputs)
    outputs = layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def get_emb(x_mask, emb_layer, ranges):
    x_mask = tf.expand_dims(tf.cast(x_mask, tf.float32), axis=-1)
    embeddings = emb_layer(ranges) * x_mask
    embeddings = tf.reduce_sum(embeddings, axis=-2)
    return embeddings


def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
    inputs = keras.Input(shape=(None,), name="inputs")
    padding_mask = keras.Input(shape=(1, 1, None), name="padding_mask")

    # self designed embeddings for multi-label classification
    ranges = np.expand_dims(np.arange(vocab_size), axis=0)
    emb_layer = layers.Embedding(vocab_size, d_model)
    embeddings = get_emb(inputs, emb_layer, ranges)

    # embeddings = layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

    outputs = layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoder_layer_{}".format(i),
        )([outputs, padding_mask])

    return keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = keras.Input(shape=(1, 1, None), name='padding_mask')

    attention1 = MultiHeadAttention(d_model, num_heads, name="attention_1")(inputs={
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': look_ahead_mask
    })
    attention1 = layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(d_model, num_heads, name="attention_2")(inputs={
        'query': attention1,
        'key': enc_outputs,
        'value': enc_outputs,
        'mask': padding_mask
    })
    attention2 = layers.Dropout(rate=dropout)(attention2)
    attention2 = layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    outputs = layers.Dense(units=units, activation='relu')(attention2)
    outputs = layers.Dense(units=d_model)(outputs)
    outputs = layers.Dropout(rate=dropout)(outputs)
    outputs = layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    inputs = keras.Input(shape=(None,), name='inputs')
    enc_outputs = keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = keras.Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = keras.Input(shape=(1, 1, None), name='padding_mask')

    # self designed embeddings for multi-label classification
    ranges = np.expand_dims(np.arange(vocab_size), axis=0)
    emb_layer = layers.Embedding(vocab_size, d_model)
    embeddings = get_emb(inputs, emb_layer, ranges)

    # embeddings = layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model)(embeddings, tf.shape(inputs)[1] - 1)

    outputs = layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i),
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

    return keras.Model(
        inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
        outputs=outputs,
        name=name)


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
    inputs = keras.Input(shape=(None,), name="inputs")
    dec_inputs = keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None), name='enc_padding_mask')(inputs)

    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(dec_inputs)

    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')(inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

