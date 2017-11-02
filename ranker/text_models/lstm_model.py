
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

"""
start_token = 1
end_token = 2
beam_width = 3
max_caption_length = 20
"""

def magic_concat(tensor_or_list, axis=0):
  if type(tensor_or_list) in [list, tuple]:
    return map(magic_concat, tensor_or_list)
  else:
    return tf.concat([tensor_or_list, tensor_or_list], axis=axis)

def get_shape(tensor):
  """Returns static shape if available and dynamic shape otherwise."""
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims

class LstmModel(object):

  def create_model(self, captions, seqlens, initializer=None, 
                   mode="train", global_step=None,
                   **unused_params):

    # Save the embedding size in the graph.
    tf.constant(FLAGS.embedding_size, name="embedding_size")
    embedding_size = FLAGS.embedding_size

    # build_seq_embeddings
    with tf.variable_scope("seq_embedding"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[FLAGS.vocab_size, FLAGS.embedding_size],
          initializer=initializer)

      seq_embeddings = tf.nn.embedding_lookup(embedding_map, captions)

    if FLAGS.lstm_cell_type == "vanilla":
      if FLAGS.num_lstm_layers > 1:
        lstm_cell = tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.BasicLSTMCell(
                        num_units=FLAGS.num_lstm_units, state_is_tuple=True)
                    for i in xrange(FLAGS.num_lstm_layers)], state_is_tuple=True)
      else:
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                    num_units=FLAGS.num_lstm_units, state_is_tuple=True)
    elif FLAGS.lstm_cell_type == "highway":
      if FLAGS.num_lstm_layers > 1:
        lstm_cell = tf.contrib.rnn.MultiRNNCell([
                    tf.contrib.rnn.HighwayWrapper(
                        cell=tf.contrib.rnn.BasicLSTMCell(
                        num_units=FLAGS.num_lstm_units, state_is_tuple=True)
                    )
                    for i in xrange(FLAGS.num_lstm_layers)], state_is_tuple=True)
      else:
        lstm_cell = tf.contrib.rnn.HighwayWrapper(
                    cell=tf.contrib.rnn.BasicLSTMCell(
                        num_units=FLAGS.num_lstm_units, state_is_tuple=True))
    else:
      raise Exception("Unknown lstm_cell_type!")

    if mode == "train":
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=FLAGS.lstm_dropout_keep_prob,
          output_keep_prob=FLAGS.lstm_dropout_keep_prob)

    with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:
      # Feed the image embeddings to set the initial LSTM state.
      output, state = tf.nn.dynamic_rnn(
                          cell=lstm_cell,
                          inputs=seq_embeddings,
                          sequence_length=seqlens,
                          dtype=tf.float32,
                          swap_memory=True,
                      )
    flat_state = magic_concat(state, axis=1)
    flat_output = magic_concat(output, axis=2)
    return flat_state, flat_output
    

