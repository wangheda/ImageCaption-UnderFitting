import tensorflow as tf
import math
FLAGS = tf.app.flags.FLAGS

class ShowAttendTellModel(object):
    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))),name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]),name=name)

    def create_model(self, input_seqs, image_model_output,initializer,
                    mode="train", target_seqs=None, input_mask=None,
                    **unused_params):
        self.n_words = FLAGS.vocab_size
        self.dim_embed = FLAGS.embedding_size
        self.dim_ctx = image_model_output.get_shape()[-1]
        self.dim_hidden = FLAGS.num_lstm_units
        self.ctx_shape = image_model_output.get_shape()[1:3]
        self.n_lstm_steps = input_mask.get_shape()[-1]
        self.batch_size = image_model_output.get_shape()[0]

        #self.Wemb = tf.Variable(tf.random_uniform([self.n_words, self.dim_embed],-1.0,1.0),name='Wemb')

        #self.init_hidden_W = self.init_weight(self.dim_ctx, self.dim_hidden, name='init_hidden_W')
        #self.init_hidden_b = self.init_bias(self.dim_hidden, name='init_hidden_b')

        #self.init_memory_W = self.init_weight(self.dim_ctx, self.hidden, name='init_memory_W')
        #self.init_memory_b = self.init_bias(self.hidden, name='init_memory_b')

        self.lstm_W = self.init_weight(self.dim_embed, self.dim_hidden, name='lstm_W')
        self.lstm_U = self.init_weight(self.dim_hidden, self.dim_hidden, name='lstm_U')
        self.lstm_b = self.init_bias(self.dim_hidden, name='lstm_b')
        self.image_encode_W = self.init_weight(self.dim_ctx, self.dim_hidden, name="image_encode_W")

        self.image_att_W = self.init_weight(self.dim_ctx,self.dim_hidden*4,name='image_encode_W')
        self.hidden_att_W = self.init_weight(self.dim_ctx,self.dim_ctx, name='hidden_att_W')
        self.pre_att_b = self.init_bias(self.dim_ctx, name='pre_att_b')

        self.att_W = self.init_weight(self.dim_ctx, 1, name='att_W')
        self.att_b = self.init_bias(1, name='att_b')

        self.decode_lstm_W = self.init_weight(self.dim_hidden, self.dim_embed, name='decode_lstm_W')
        self.decode_lstm_b = self.init_bias(self.dim_embed, name='decode_lstm_b')

        self.decode_word_W = self.init_weight(self.dim_embed, self.n_words, name='decode_word_W')
        self.decode_word_b = self.init_bias(self.n_words, name='decode_word_b')

        with tf.variable_scope("seq_embedding"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[FLAGS.vocab_size, FLAGS.embedding_size],
                initializer=initializer
            )
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=FLAGS.num_lstm_units, state_is_tuple=True
        )

        if mode == 'train':
            lstm_cell = tf.contrib.rnn.DroupoutWrapper(
                lstm_cell,
                input_keep_prob=FLAGS.lstm_dropout_keep_prob,
                output_keep_prob=FLAGS.lstm_dropout_keep_prob
            )

        context_flat = tf.reshape(image_model_output, [-1, self.dim_ctx])
        context_encode = tf.matmul(context_flat, self.image_att_W)
        context_encode = tf.reshape(context_encode, [-1, self.ctx_shape[0], self.ctx_shape[1]])

        with tf.variable_scope("lstm", intializer=initializer) as lstm_scope:
            zero_state = lstm_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            h, state = lstm_cell(tf.reduce_mean(image_model_output,1), zero_state)

            if mode == "inference":
                tf.concat(axis=1, values=state, name='initial_state')
                state_feed = tf.placeholder(dtype=tf.float32,
                                            shape=[None, sum(lstm_cell.state_size)],
                                            name="state_feed")
                state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

                lstm_outputs, state_tuple = lstm_cell(
                    inputs=tf.squeeze(seq_embeddings, axis=[1]),
                    state=state_tuple
                )
                tf.concat(axis=1, values=state_tuple, name='state')
                lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])
                logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b
                logits = tf.nn.relu(logits)
                logits = tf.nn.dropout(logits, 0.8)

                logits = tf.matmul(logits, self.decode_word_W) + self.decode_word_b

            else:
                logits = tf.zeros([self.n_lstm_steps, self.batch_size, self.n_words])
                for ind in range(self.n_lstm_steps):
                    if ind == 0:
                        word_emb = tf.zeros([self.batch_size, self.dim_embed])
                    else:
                        tf.get_variable_scope().reuse_variables()
                        word_emb = seq_embeddings[:, ind-1]

                    x_t = tf.matmul(word_emb, self.lstm_W) + self.lstm_b

                    labels = tf.expand_dims(input_seqs[:, ind],1)
                    indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                    concated = tf.concat(1, [indices, labels])
                    onehot_labels = tf.sparse_to_dense(concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)

                    context_encode = context_encode + \
                        tf.expand_dims(tf.matmul(h, self.hidden_att_W),1) + \
                        self.pre_att_b
                    context_encode = tf.nn.tanh(context_encode)

                    context_encode_flat = tf.reshape(context_encode, [-1, self.dim_ctx])
                    alpha = tf.matmul(context_encode_flat, self.att_W) + self.att_b
                    alpha = tf.reshape(alpha, [-1, self.ctx_shape[0]])
                    alpha = tf.nn.softmax(alpha)
                    weighted_context = tf.reduce_sum(image_model_output * tf.expand_dims(alpha,2),1)

                    lstm_preactive = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(weighted_context, self.image_encode_W)
                    h, state = lstm_cell(lstm_preactive, state)

                    logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b
                    logits = tf.nn.relu(logits)
                    logits = tf.nn.dropout(logits, 0.8)

                    logit_words = tf.matmul(logits, self.decode_word_W) + self.decode_word_b
                    logit_words = logit_words * input_mask[:, ind]
                    logits[ind,:,:] = logit_words

                logits = tf.reshape(logits, [-1, self.n_words])
        return {"logits": logits}



