import tensorflow as tf
from tensorflow.keras import layers, models

from util import ID_TO_CLASS


class BasicAttentiveBiGRU(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyBasicAttentiveBiGRU, self).__init__()

        self.num_classes = len(ID_TO_CLASS)

        self.decoder = layers.Dense(units=self.num_classes)
        self.omegas = tf.Variable(tf.random.normal((hidden_size*2, 1)))
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))

        ### TODO(Students) START
        #define forward and backward layers of gru
        self.layer_forward = tf.keras.layers.GRU(hidden_size, return_sequences=True, activation='tanh', recurrent_activation='sigmoid' , use_bias=True)
        self.layer_backward = tf.keras.layers.GRU(hidden_size, return_sequences=True, activation='tanh', recurrent_activation='sigmoid' , use_bias=True, go_backwards=True)

        #combine forward and backward layers using bidirectional wrapper around GRU
        self.biGRUbaseline = tf.keras.layers.Bidirectional(self.layer_forward, backward_layer=self.layer_backward, input_shape=(embed_dim, hidden_size))

        ### TODO(Students) END

    def attn(self, rnn_outputs):
        
        #compute term alpha
        M = tf.math.tanh(rnn_outputs)
        alpha = tf.nn.softmax(tf.matmul(M, self.omegas), axis =1)
        #multiply each vector with weight alpha
        sentence_rep = tf.multiply(rnn_outputs, alpha)
        #compute summation of combined weighted outputs for each sentence
        output = tf.math.reduce_sum(sentence_rep, axis=1)

        return output

    def call(self, inputs, pos_inputs, training):

        word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)

        #mask the paddings given for the input
        mask_paddings = tf.cast(inputs!=0, dtype= tf.float32)

        #For word embeddings only
        #out_GRU = self.biGRUbaseline(word_embed, mask = mask_paddings)

        #For word+pos
        word_n_pos = tf.concat([word_embed, pos_embed], 2)
        out_GRU = self.biGRUbaseline(word_n_pos, mask = mask_paddings)


        out_attention = self.attn(out_GRU)
        final_sentence_rep = tf.math.tanh(out_attention) 

        logits = self.decoder(final_sentence_rep)

        return {'logits': logits}


class CNN_Model(models.Model):

    def __init__(self, vocab_size: int, embed_dim: int, hidden_size: int = 128, training: bool = False):
        super(MyAdvancedModel, self).__init__()
        ### TODO(Students) START
        #CNN with one convolution layer
        self.num_classes = len(ID_TO_CLASS)
        self.decoder = layers.Dense(units=self.num_classes)
        self.embeddings = tf.Variable(tf.random.normal((vocab_size, embed_dim)))
        self.convolution = tf.keras.layers.Conv1D(filters=256, kernel_size=2, padding='same', activation='relu')
        self.max_pool = tf.keras.layers.GlobalMaxPool1D()



    def call(self, inputs, pos_inputs, training):
        
        #window processing using embedding_lookup for word and pos embeddings
        cnn_word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        cnn_pos_embed = tf.nn.embedding_lookup(self.embeddings, pos_inputs)
        #concatenate word and pos embeddings
        cnn_word_n_pos = tf.concat([cnn_word_embed, cnn_pos_embed], 2)
        #convolution on the concatenated output
        cnn_layer = self.convolution(cnn_word_n_pos)

        #experiment 2 with word embeddings and dep features
        # cnn_word_embed = tf.nn.embedding_lookup(self.embeddings, inputs)
        # cnn_layer = self.convolution(cnn_word_embed)

        #max-pooling
        max_pooling_val = self.max_pool(cnn_layer)
        #extracting sentence level features
        logits = self.decoder(max_pooling_val)
        
        return {'logits': logits}

