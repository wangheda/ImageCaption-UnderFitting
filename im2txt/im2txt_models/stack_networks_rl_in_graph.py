import tensorflow as tf
from tensorflow.python.layers.core import Dense
from ResLSTM import ResLSTMWrapper

FLAGS = tf.app.flags.FLAGS

"""
start_token = 1
end_token = 2
beam_width = 3
max_caption_length = 20
"""

def get_shape(tensor):
  """Returns static shape if available and dynamic shape otherwise."""
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims

class StackNetworkModel(object):

    def create_model(self, input_seqs, image_model_output, initializer,
                 mode="train", target_seqs=None, input_mask=None,
                 global_step=None,
                 **ununsed_params):
        print "image_model_output", image_model_output

        if FLAGS.yet_another_inception:
            if FLAGS.inception_return_tuple:
                image_model_output, middle_layer, ya_image_model_output, ya_middle_layer = image_model_output
            else:
                image_model_output, ya_image_model_output = image_model_output
        else:
            if FLAGS.inception_return_tuple:
                image_model_output, middle_layer = image_model_output
                if FLAGS.l2_normalize_image:
                    middle_layer = tf.nn.l2_normalize(middle_layer, dim=-1)
            else:
                image_model_output = image_model_output

        with tf.variable_scope("seq_embedding"):
            embedding_map = tf.get_variable(
                name="map",
                shape=[FLAGS.vocab_size, FLAGS.embedding_size],
                initializer=initializer
            )
            seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)

        embedding_size = embedding_map.get_shape().as_list()[1]

        with tf.variable_scope("image_embedding") as scope:
            image_embeddings = tf.contrib.layers.fully_connected(
                inputs=image_model_output,
                num_outputs=embedding_size,
                activation_fn=None,
                weights_initializer=initializer,
                biases_initializer=None,
                scope=scope
            )
            if FLAGS.l2_normalize_image:
                print "l2 normalize image"
                image_embeddings = tf.nn.l2_normalize(image_embeddings, dim=-1)
        tf.constant(FLAGS.embedding_size, name="embedding_size")

        self.image_embeddings = image_embeddings

        with tf.variable_scope("lstm_layer_1") as scope_1:
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=FLAGS.num_lstm_units, state_is_tuple=True
            )

            if mode == 'train':
                lstm_cell = tf.contrib.rnn.DropoutWrapper(
                    lstm_cell,
                    input_keep_prob=FLAGS.lstm_dropout_keep_prob,
                    output_keep_prob=FLAGS.lstm_dropout_keep_prob
                )
            if FLAGS.use_attention_wrapper:
                if mode == "inference":
                    middle_layer = tf.contrib.seq2seq.tile_batch(middle_layer, multiplier=FLAGS.beam_width)
                attention_mechanism = getattr(
                    tf.contrib.seq2seq, FLAGS.attention_mechanism)(
                    num_units=FLAGS.num_attention_depth,
                    memory=middle_layer
                )
                lstm_cell = tf.contrib.seq2seq.AttentionWrapper(
                    lstm_cell,
                    attention_mechanism,
                    attention_layer_size=FLAGS.num_attention_depth,
                    output_attention=FLAGS.output_attention
                )
            lstm_cell = ResLSTMWrapper(lstm_cell)

        with tf.variable_scope('lstm_layer_2') as scope_2:
            lstm_review = tf.contrib.rnn.BasicLSTMCell(num_units=FLAGS.num_lstm_units+FLAGS.embedding_size,state_is_tuple=True)
            if mode=='train':
                lstm_review = tf.contrib.rnn.DropoutWrapper(
                    lstm_review,
                    input_keep_prob=FLAGS.lstm_dropout_keep_prob,
                    output_keep_prob=FLAGS.lstm_dropout_keep_prob,
                )
            if FLAGS.use_attention_wrapper:
                attention_mechanism = getattr(tf.contrib.seq2seq, FLAGS.attention_mechanism)(
                    num_units=FLAGS.num_attention_depth,
                    memory=middle_layer
                )
                lstm_review = tf.contrib.seq2seq.AttentionWrapper(
                    lstm_review,
                    attention_mechanism=attention_mechanism,
                    attention_layer_size=FLAGS.num_attention_depth,
                    output_attention=FLAGS.output_attention
                )

        with tf.variable_scope('lstm_layer_multi') as scope_3:
            multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell, lstm_review],state_is_tuple=True)
            batch_size = get_shape(image_embeddings)[0]
            output_layer = Dense(units=FLAGS.vocab_size,
                                 name="output_layer")
            print(multi_cell.state_size)
            print(lstm_cell.state_size)

            if mode == 'train':
                zero_state = multi_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
                _, initial_state = multi_cell(image_embeddings, zero_state)

                if FLAGS.rl_training == True:
                    greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        embedding=embedding_map,
                        start_tokens=tf.fill([batch_size], FLAGS.start_token),
                        end_token=FLAGS.end_token
                    )
                    greedy_decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=multi_cell,
                        helper=greedy_helper,
                        initial_state=initial_state,
                        output_layer=output_layer
                    )
                    greedy_outputs, _, greedy_outputs_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                        decoder=greedy_decoder,
                        output_time_major=False,
                        impute_finished=False,
                        maximum_iterations=FLAGS.max_caption_length
                    )

                    helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
                        embedding=embedding_map,
                        start_tokens=tf.fill([batch_size], FLAGS.start_token),
                        end_token=FLAGS.end_token
                    )
                else:
                    sequence_length = tf.reduce_sum(input_mask, 1)
                    helper = tf.contrib.seq2seq.TrainingHelper(
                        inputs=seq_embeddings,
                        sequence_length=sequence_length
                    )
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=multi_cell,
                    helper=helper,
                    initial_state=initial_state,
                    output_layer=output_layer
                )

            elif mode == "inference":
                image_embeddings = tf.contrib.seq2seq.tile_batch(image_embeddings, multiplier=FLAGS.beam_width)
                zero_state = multi_cell.zero_state(batch_size=batch_size * FLAGS.beam_width, dtype=tf.float32)
                _, initial_state = multi_cell(image_embeddings, zero_state)

                decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=multi_cell,
                    embedding=embedding_map,
                    start_tokens=tf.fill([batch_size], FLAGS.start_token),
                    end_token=FLAGS.end_token,
                    initial_state=initial_state,
                    beam_width=FLAGS.beam_width,
                    output_layer=output_layer,
                    length_penalty_weight=0.0
                )
            else:
                raise Exception("Unknown mode!")

            if mode == 'train':
                if FLAGS.rl_training == True:
                    maximum_iterations = FLAGS.max_caption_length
                else:
                    maximum_iterations = None

            else:
                maximum_iterations = FLAGS.max_caption_length

            outputs, _, outputs_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=maximum_iterations
            )

        if mode == "train":
            if FLAGS.rl_training == True:
                return {"sample_caption_words": outputs.sample_id,
                        "sample_caption_logits": outputs.rnn_output,
                        "sample_caption_lengths": outputs_sequence_lengths,
                        "greedy_caption_words": greedy_outputs.sample_id,
                        "greedy_caption_lengths": greedy_outputs_sequence_lengths}
            else:
                logits = tf.reshape(outputs.rnn_output, [-1, FLAGS.vocab_size])
                return {"logits": logits}
        else:
            return {"bs_results": outputs}
