import tensorflow as tf
import math

FLAGS = tf.app.flags.FLAGS
import numpy as np

from ResLSTM import ResLSTMWrapper
from tensorflow.python.layers.core import Dense

def get_shape(tensor):
  """Returns static shape if available and dynamic shape otherwise."""
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims

class StackNetworkModel(object):
  def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
     return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev / math.sqrt(float(dim_in))), name=name)

  def init_bias(self, dim_out, name=None):
      return tf.Variable(tf.zeros([dim_out]), name=name)

  def create_model(self, input_seqs, image_model_output, initializer,
             mode="train", target_seqs=None, input_mask=None,
             **ununsed_params):
      self.use_concate_image = True

      if FLAGS.yet_another_inception:
        if FLAGS.inception_return_tuple:
          image_model_output, middle_layer, ya_image_model_output, ya_middle_layer = image_model_output
        else:
          image_model_output, ya_image_model_output = image_model_output
      else:
        if FLAGS.inception_return_tuple:
          image_model_output, middle_layer = image_model_output
        else:
          image_model_output = image_model_output

      with tf.variable_scope("seq_embedding"):
        embedding_map = tf.get_variable(
            name='map',
            shape=[FLAGS.vocab_size, FLAGS.embedding_size],
            initializer=initializer
        )
        seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)

      with tf.variable_scope("image_embedding") as scope:
        image_embeddings = tf.contrib.layers.fully_connected(
            inputs=image_model_output,
            num_outputs=FLAGS.embedding_size,
            activation_fn=None,
            weights_initializer=initializer,
            biases_initializer=None,
            scope=scope
        )

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
                num_units = FLAGS.num_attention_depth,
                memory = middle_layer
            )
            lstm_cell = tf.contrib.seq2seq.AttentionWrapper(
                lstm_cell,
                attention_mechanism,
                attention_layer_size=FLAGS.num_attention_depth,
                output_attention=FLAGS.output_attention
            )

        lstm_cell = ResLSTMWrapper(lstm_cell)

      with tf.variable_scope('lstm_layer_2') as scope_2:
        lstm_review = tf.contrib.rnn.BasicLSTMCell(num_units=FLAGS.num_lstm_units+512,state_is_tuple=True)
        if mode=='train':
          if mode == "inference":
            middle_layer = tf.contrib.seq2seq.tile_batch(middle_layer, multiplier=FLAGS.beam_width)
            lstm_review = tf.contrib.rnn.DropoutWrapper(
                lstm_review,
                input_keep_prob=FLAGS.lstm_dropout_keep_prob,
                output_keep_prob=FLAGS.lstm_dropout_keep_prob)
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
        multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell,lstm_review],state_is_tuple=True)
        batch_size = get_shape(image_embeddings)[0]
        print(multi_cell.state_size)
        print(lstm_cell.state_size)

        if mode == 'train':
            zero_state = multi_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            _, initial_state = multi_cell(image_embeddings, zero_state)
        else:
            image_embeddings = tf.contrib.seq2seq.tile_batch(image_embeddings, multiplier=FLAGS.beam_width)
            zero_state = multi_cell.zero_state(batch_size=batch_size*FLAGS.beam_width, dtype=tf.float32)
            _, initial_state = multi_cell(image_embeddings, zero_state)

        print(initial_state)
        output_layer = Dense(units=FLAGS.vocab_size, name="output_layer")

        if mode=='train':

          sequence_length = tf.reduce_sum(input_mask,1)
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

        elif mode == 'inference':
          decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=multi_cell,
              embedding=embedding_map,
              start_tokens=tf.fill([batch_size],FLAGS.start_token),
              end_token=FLAGS.end_token,
              initial_state=initial_state,
              beam_width=FLAGS.beam_width,
              output_layer=output_layer,
              length_penalty_weight=0.0
            )
          maximum_iterations = None if mode == "train" else FLAGS.max_caption_length
          outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            output_time_major=False,
            impute_finished=False,
            maximum_iterations=maximum_iterations)

      if mode == "train":
        logits = tf.reshape(outputs.rnn_output, [-1, FLAGS.vocab_size])
        return {"logits": logits}
      else:
        return {"bs_results": outputs}


