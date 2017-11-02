import tensorflow as tf
import math

FLAGS = tf.app.flags.FLAGS
import numpy as np


class ReviewnetworkModel(object):
    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def init_embedding(self, vocab_size, embedding_size):
        embeddings = np.random.rand(vocab_size, embedding_size)
        embeddings[2, :] = np.zeros((embedding_size))
        return embeddings

    def create_model(self, input_seqs, image_model_output, initializer,
                     mode="train", target_seqs=None, input_mask=None,
                     **unused_params):
        self.n_words = FLAGS.vocab_size
        self.dim_embed = FLAGS.embedding_size
        self.dim_ctx_input = 2048
        self.dim_ctx = 512  # dimensiion of LSTM input
        self.dim_hidden = FLAGS.num_lstm_units  # dimension of hidden size in LSTM
        self.ctx_shape = [tf.cast(image_model_output.get_shape()[1], tf.int32), self.dim_ctx]
        self.n_lstm_steps = None
        self.batch_size = 1  # default value

        self.init_hidden_W = self.init_weight(self.dim_ctx_input, self.dim_hidden, name='init_hidden_W')
        # self.init_hidden_b = self.init_bias(self.dim_hidden, name='init_hidden_b')

        self.init_memory_W = self.init_weight(self.dim_ctx_input, self.dim_hidden, name='init_memory_W')
        # self.init_memory_b = self.init_bias(self.hidden, name='init_memory_b')

        self.lstm_W = self.init_weight(self.dim_ctx_input + self.dim_hidden + self.dim_embed, self.dim_ctx,
                                       name='lstm_W')

        self.image_encode_W = self.init_weight(self.dim_ctx_input, self.dim_ctx, name="image_encode_W")

        self.image_att_W = self.init_weight(self.dim_ctx_input, self.dim_ctx, name='image_encode_W')
        self.hidden_att_W = self.init_weight(self.dim_hidden, self.dim_ctx, name='hidden_att_W')
        self.pre_att_b = self.init_bias(self.dim_ctx, name='pre_att_b')

        self.att_W = self.init_weight(self.dim_ctx, 1, name='att_W')
        self.att_b = self.init_bias(1, name='att_b')

        self.image_decode_W = self.init_weight(self.dim_ctx_input, self.dim_embed+self.dim_hidden, name='decode_image_W')
        self.decode_lstm_W = self.init_weight(self.dim_hidden, self.dim_embed, name='decode_lstm_W')
        self.decode_lstm_b = self.init_bias(self.dim_embed, name='decode_lstm_b')

        self.decode_word_W = self.init_weight(self.dim_embed, self.n_words, name='decode_word_W')
        self.decode_word_b = self.init_bias(self.n_words, name='decode_word_b')

        with tf.variable_scope("seq_embedding"):
            '''
            embedding_map = tf.get_variable(
                name="map",
                shape=[FLAGS.vocab_size, FLAGS.embedding_size],
                initializer=initializer
            )
            '''
            embedding_map = tf.Variable(
                self.init_embedding(FLAGS.vocab_size, FLAGS.embedding_size),
                trainable=False,
                dtype=tf.float32,
                name="map")
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)

        context_flat = tf.reshape(image_model_output, [-1, self.dim_ctx_input])
        context_encode = tf.matmul(context_flat, self.image_att_W)
        context_encode = tf.reshape(context_encode, [-1, self.ctx_shape[0], self.ctx_shape[1]])

        self.image = image_model_output
        self.context_encode = context_encode

        with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=FLAGS.num_lstm_units, state_is_tuple=True
            )
            if mode == 'train':
                lstm_cell = tf.contrib.rnn.DropoutWrapper(
                        lstm_cell,
                        input_keep_prob=FLAGS.lstm_dropout_keep_prob,
                        output_keep_prob=FLAGS.lstm_dropout_keep_prob
                )
                self.n_lstm_steps = tf.reduce_max(tf.reduce_sum(input_mask, 1))
                self.batch_size = tf.cast(input_mask.get_shape()[0], tf.int32)

            zero_state = lstm_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            initial_state = tf.reduce_mean(self.image, 1)
            h = tf.nn.tanh(tf.matmul(initial_state, self.init_hidden_W))
            c = tf.nn.tanh(tf.matmul(initial_state, self.init_memory_W))
            state = tf.contrib.rnn.LSTMStateTuple(c, h)
            # to construct LSTM cell kernal here
            _, _ = lstm_cell(h, state)

            #lstm_scope.reuse_variables()

            if mode == "inference":
                tf.concat(axis=1, values=state, name='initial_state')
                state_feed = tf.placeholder(dtype=tf.float32,
                                            shape=[None, sum(lstm_cell.state_size)],
                                            name="state_feed")
                state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

                x_t = tf.squeeze(seq_embeddings, axis=[1])

                context_encode = self.context_encode + \
                                 tf.expand_dims(tf.matmul(state_tuple[1], self.hidden_att_W), 1) + \
                                 self.pre_att_b
                context_encode = tf.nn.tanh(context_encode)

                context_encode_flat = tf.reshape(context_encode, [-1, self.dim_ctx])
                alpha = tf.matmul(context_encode_flat, self.att_W) + self.att_b
                alpha = tf.reshape(alpha, [-1, self.ctx_shape[0]])
                alpha = tf.nn.softmax(alpha)
                weighted_context = tf.reduce_sum(self.image * tf.expand_dims(alpha, 2), 1)
                # weighted_context = tf.tile(weighted_context,

                lstm_preactive = tf.concat(axis=1, values=[state_tuple[1], x_t, weighted_context])
                lstm_preactive = tf.matmul(lstm_preactive, self.lstm_W)

                lstm_outputs, state_tuple = lstm_cell(
                    lstm_preactive,
                    state=state_tuple
                )
                tf.concat(axis=1, values=state_tuple, name='state')

                logits_final = lstm_outputs

            else:
                logits_0 = tf.TensorArray(dtype=tf.float32, size=self.n_lstm_steps)
                i0 = tf.constant(0)

                def cond(ind, m0, h, state):
                    return ind < self.n_lstm_steps

                def body(ind, m0, h, state):
                    tf.get_variable_scope().reuse_variables()
                    x_t = seq_embeddings[:, ind, :]

                    context_encode = self.context_encode + \
                                     tf.expand_dims(tf.matmul(h, self.hidden_att_W), 1) + \
                                     self.pre_att_b
                    context_encode = tf.nn.tanh(context_encode)

                    context_encode_flat = tf.reshape(context_encode, [-1, self.dim_ctx])
                    alpha = tf.matmul(context_encode_flat, self.att_W) + self.att_b
                    alpha = tf.reshape(alpha, [-1, self.ctx_shape[0]])
                    alpha = tf.nn.softmax(alpha)
                    weighted_context = tf.reduce_sum(self.image * tf.expand_dims(alpha, 2), 1)

                    lstm_preactive = tf.concat(axis=1, values=[h, x_t, weighted_context])
                    lstm_preactive = tf.matmul(lstm_preactive, self.lstm_W)
                    h, state = lstm_cell(lstm_preactive, state)

                    m0 = m0.write(ind, h)
                    ind += 1
                    return ind, m0, h, state

                ii, logits_final, _, _ = tf.while_loop(cond, body, [i0, logits_0, h, state])
                logits_final = logits_final.stack()
                logits_final = tf.transpose(logits_final, perm=[1,0,2])

        #print(seq_embeddings.get_shape().as_list())
        #print(logits_final.get_shape().as_list())
        with tf.variable_scope("lstm_review",initializer=initializer) as lstm_review_scope:
            lstm_review = tf.contrib.rnn.LSTMCell(
                num_units=self.dim_embed+self.dim_hidden, state_is_tuple=True
            )
            zero_state = lstm_review.zero_state(self.batch_size, dtype=tf.float32)

            _, initial_state = lstm_review(tf.matmul(tf.reduce_sum(image_model_output,1),self.image_decode_W), zero_state)

            lstm_review_scope.reuse_variables()

            if mode == "inference":
                tf.concat(axis=1, values=initial_state, name="initial_state_review")
                state_review_feed = tf.placeholder(dtype=tf.float32,
                                                   shape=[None, sum(lstm_review.state_size)],
                                                   name='state_review_feed')
                state_review_tuple = tf.split(value=state_review_feed, num_or_size_splits=2, axis=1)

                lstm_outputs, state_review_tuple = lstm_review(
                    inputs=tf.concat(axis=-1, values=[tf.squeeze(seq_embeddings,axis=[1]),logits_final]),
                    state=state_review_tuple
                )
                tf.concat(axis=1, values=state_review_tuple, name="state_review")
            else:
                sequence_length = tf.reduce_sum(input_mask, 1)
                lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_review,
                                                    inputs=tf.concat(axis=-1, values=[seq_embeddings,logits_final]),
                                                    sequence_length=sequence_length,
                                                    initial_state=initial_state,
                                                    dtype=tf.float32,
                                                    scope=lstm_review_scope)
        lstm_outputs = tf.reshape(lstm_outputs,[-1, lstm_review.output_size])

        with tf.variable_scope("logits") as logits_scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=lstm_outputs,
                num_outputs=FLAGS.vocab_size,
                activation_fn=None,
                weights_initializer=initializer,
                scope=logits_scope
            )

        return {"logits": logits}



