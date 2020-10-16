import tensorflow as tf

# tf.enable_eager_execution()
import numpy as np

keras = tf.keras
layers = keras.layers


class FC(keras.Model):

    def __init__(self, input_dim, emb_dim, unit_list, dropout_rate=0.0, activation='tanh',
                 mode='concat', use_embeddings=True, name='fc'):
        super(FC, self).__init__(name=name)

        self.__mode = mode
        self.__use_embeddings = use_embeddings
        if self.__use_embeddings:
            # initialize embedding layers
            self.__ranges = np.expand_dims(np.arange(input_dim), axis=0)
            self.__emb = layers.Embedding(input_dim, emb_dim)

        len_layers = len(unit_list)

        # initialize dropout layers
        self.__dropout_layers = [layers.Dropout(dropout_rate, seed=i) for i in range(len_layers)]

        # initialize dense layers
        self.__dense_layers = [layers.Dense(units, activation=activation) for i, units in enumerate(unit_list)]

    def __get_emb(self, x_mask):
        x_mask = tf.expand_dims(tf.cast(x_mask, tf.float32), axis=-1)
        embeddings = self.__emb(self.__ranges) * x_mask
        embeddings = tf.reduce_sum(embeddings, axis=-2)
        return embeddings

    def call(self, inputs, training=None, mask=None):
        if not self.__use_embeddings:
            embeddings = inputs
            embeddings = tf.cast(embeddings, tf.float64)
        else:
            embeddings = self.__get_emb(inputs)

        if self.__mode == 'sum':
            # sum embeddings by their time steps
            x = tf.reduce_sum(embeddings, axis=1)
        else:
            # concat embeddings by their time steps
            x = tf.reshape(embeddings, [-1, embeddings.shape[1] * embeddings.shape[2]])

        for i, dense in enumerate(self.__dense_layers):
            x = self.__dropout_layers[i](x, training=training)
            x = dense(x)

        return x

# a = np.array([[3, 2, 2, 0, 1], [0, 0, 1, 3, 0]])
# b = np.array([[0, 0, 0, 1, 1], [0, 1, 1, 0, 0]])
# aa = np.array([a, a, a, b, b, a])
# # c = [a, b]
#
# lstm_layer = FC(5, 3, [5, 5], 0.2, mode='sum')
# _input = lstm_layer(aa)
#
# print('!!!!!!!!!!!!!!!!!!!!!')
# print(_input)
