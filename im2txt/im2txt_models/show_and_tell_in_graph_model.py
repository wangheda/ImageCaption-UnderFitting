
import tensorflow as tf
from tensorflow.python.layers.core import Dense

FLAGS = tf.app.flags.FLAGS


class ShowAndTellInGraphModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def create_model(self, input_seqs, image_model_output, initializer, 
                   mode="train", target_seqs=None, input_mask=None, 
                   **unused_params):
    
    # Map image model output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=image_model_output,
          num_outputs=FLAGS.embedding_size,
          activation_fn=None,
          weights_initializer=initializer,
          biases_initializer=None,
          scope=scope)

    # Save the embedding size in the graph.
    tf.constant(FLAGS.embedding_size, name="embedding_size")

    self.image_embeddings = image_embeddings
    # build_seq_embeddings
    with tf.variable_scope("seq_embedding"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[FLAGS.vocab_size, FLAGS.embedding_size],
          initializer=initializer)
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)

    """Builds the model.
    Inputs:
      self.image_embeddings
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=FLAGS.num_lstm_units, state_is_tuple=True)

    if mode == "train":
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=FLAGS.lstm_dropout_keep_prob,
          output_keep_prob=FLAGS.lstm_dropout_keep_prob)

    #output_layer = Dense(units=FLAGS.vocab_size,
    #                     name="output_layer")

    with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:
      # Feed the image embeddings to set the initial LSTM state.
      batch_size = image_embeddings.get_shape()[0]

      zero_state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
      _, initial_state = lstm_cell(image_embeddings, zero_state)

      output_layer = Dense(units=FLAGS.vocab_size,
                                     name="output_layer")
      # Allow the LSTM variables to be reused.
      # lstm_scope.reuse_variables()

      if mode == "inference":
        # In inference mode, use concatenated states for convenient feeding and
        # fetching.
        tf.concat(axis=1, values=initial_state, name="initial_state")

        # Placeholder for feeding a batch of concatenated states.
        state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(lstm_cell.state_size)],
                                    name="state_feed")
        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

        sequence_length = tf.ones([state_feed.get_shape()[0]], tf.int32)
      else:
        sequence_length = tf.reduce_sum(input_mask, 1)

      helper = tf.contrib.seq2seq.TrainingHelper(inputs=seq_embeddings,
                                                 sequence_length=sequence_length)

      decoder = tf.contrib.seq2seq.BasicDecoder(cell=lstm_cell,
                                                helper=helper,
                                                initial_state=initial_state,
                                                output_layer=output_layer)
      if mode == "inference":
        # Run a single LSTM step.
        outputs, state_tuple, _, _ = decoder.step(time=1,
                                                  inputs=tf.squeeze(seq_embeddings, axis=[1]),
                                                  state=state_tuple)

        # Concatentate the resulting state.
        tf.concat(axis=1, values=state_tuple, name="state")
      else:
        # Run the batch of sequence embeddings through the LSTM.
        outputs, _ , _ = tf.contrib.seq2seq.dynamic_decode(decoder=decoder,
                                                           scope=lstm_scope)
      
      logits = tf.reshape(outputs.rnn_output, [-1, FLAGS.vocab_size])
    return logits


