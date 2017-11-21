
import tensorflow as tf
from tensorflow.python.layers.core import Dense

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

class ReviewNetworkModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1605.07912.

  "Review Networks for Caption Generation arXiv:1605.03925v4"
  """

  def create_model(self, input_seqs, image_model_output, initializer, 
                   mode="train", target_seqs=None, input_mask=None,
                   **unused_params):
    
    assert FLAGS.inception_return_tuple
    image_model_output, middle_layer = image_model_output

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

    # add review network here!!!
    batch_size = get_shape(image_embeddings)[0]

    review_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=FLAGS.num_lstm_units, state_is_tuple=True)

    if mode == "train":
      review_cell = tf.contrib.rnn.DropoutWrapper(
          review_cell,
          input_keep_prob=FLAGS.lstm_dropout_keep_prob,
          output_keep_prob=FLAGS.lstm_dropout_keep_prob)

    review_attention_mechanism = getattr(tf.contrib.seq2seq, 
        FLAGS.attention_mechanism)(
            num_units = FLAGS.num_attention_depth,
            memory = middle_layer)

    # Attentive input review: output_attention=False
    review_cell = tf.contrib.seq2seq.AttentionWrapper(
        review_cell,
        review_attention_mechanism,
        attention_layer_size=FLAGS.num_attention_depth,
        output_attention=False)

    review_zero_state = review_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
    _, review_initial_state = review_cell(image_embeddings, review_zero_state)

    # review step don't need inputs
    review_input_seqs = tf.zeros([batch_size, FLAGS.review_steps, FLAGS.embedding_size])
    review_sequence_length = tf.ones([batch_size], tf.int32) * FLAGS.review_steps
    review_helper = tf.contrib.seq2seq.TrainingHelper(
      inputs=review_input_seqs,
      sequence_length=review_sequence_length)
    review_decoder = tf.contrib.seq2seq.BasicDecoder(
      cell=review_cell,
      helper=review_helper,
      initial_state=review_initial_state)

    review_outputs, _ , _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=review_decoder,
        output_time_major=False,
        impute_finished=False)

    fact_vectors = review_outputs.rnn_output

    # add discriminative supervision here
    # to do!
    with tf.variable_scope("discriminative_logits") as scope:
      discriminative_logits = tf.contrib.layers.fully_connected(
          inputs=fact_vectors,
          num_outputs=FLAGS.vocab_size,
          activation_fn=None,
          weights_initializer=initializer,
          scope=scope)

      discriminative_logits = tf.reduce_max(discriminative_logits, axis=1)


    print "review network debug:"
    print "seq_embeddings:", seq_embeddings
    print "review_input_seqs:", review_input_seqs
    print "review_sequence_length:", review_sequence_length
    print "fact_vectors:", fact_vectors
    print "discriminative_logits", discriminative_logits


    # attention decoder part:
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=FLAGS.num_lstm_units, state_is_tuple=True)

    if mode == "train":
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=FLAGS.lstm_dropout_keep_prob,
          output_keep_prob=FLAGS.lstm_dropout_keep_prob)

    ## The parameter of attention should be re-considered
    # num_units and attention_layer_size

    if mode == "inference":
      fact_vectors = tf.contrib.seq2seq.tile_batch(fact_vectors, multiplier=FLAGS.beam_width)

    attention_mechanism = getattr(tf.contrib.seq2seq, 
        FLAGS.attention_mechanism)(
            num_units = FLAGS.num_attention_depth,
            memory = fact_vectors)

    lstm_cell = tf.contrib.seq2seq.AttentionWrapper(
      lstm_cell,
      attention_mechanism,
      attention_layer_size=FLAGS.num_attention_depth,
      output_attention=FLAGS.output_attention)

    with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:
      # Feed the image embeddings to set the initial LSTM state.     
      if mode == "train":
        zero_state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        _, initial_state = lstm_cell(image_embeddings, zero_state)
      elif mode == "inference":
        image_embeddings = tf.contrib.seq2seq.tile_batch(image_embeddings, multiplier=FLAGS.beam_width)
        zero_state = lstm_cell.zero_state(batch_size=batch_size*FLAGS.beam_width, dtype=tf.float32)
        _, initial_state = lstm_cell(image_embeddings, zero_state)
      else:
        raise Exception("Unknown mode!")

      output_layer = Dense(units=FLAGS.vocab_size,
                           name="output_layer")
      # Allow the LSTM variables to be reused.
      # lstm_scope.reuse_variables()

      if mode == "train":
        sequence_length = tf.reduce_sum(input_mask, 1)
        helper = tf.contrib.seq2seq.TrainingHelper(
          inputs=seq_embeddings,
          sequence_length=sequence_length)
        decoder = tf.contrib.seq2seq.BasicDecoder(
          cell=lstm_cell,
          helper=helper,
          initial_state=initial_state,
          output_layer=output_layer)

      elif mode == "inference":
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
          cell=lstm_cell,
          embedding=embedding_map,
          start_tokens=tf.fill([batch_size], FLAGS.start_token),    #[batch_size]
          end_token=FLAGS.end_token,
          initial_state=initial_state, #[batch_size*beam_width]
          beam_width=FLAGS.beam_width,
          output_layer=output_layer,
          length_penalty_weight=0.0)
      else:
        raise Exception("Unknown mode!")

      maximum_iterations = None if mode == "train" else FLAGS.max_caption_length
      outputs, _ , _ = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        output_time_major=False,
        impute_finished=False,
        maximum_iterations=maximum_iterations)

    if mode == "train":
      logits = tf.reshape(outputs.rnn_output, [-1, FLAGS.vocab_size])
      return {"logits": logits, "discriminative_logits": discriminative_logits}
    else:
      return {"bs_results": outputs}
