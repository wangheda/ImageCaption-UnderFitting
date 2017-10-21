
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

class SemanticAttentionModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1603.03925.

  "Image Captioning with Semantic Attention arXiv:1603.03925v1"
  """

  def create_model(self, input_seqs, image_model_output, initializer, 
                   mode="train", target_seqs=None, input_mask=None, multi_label_mask=None,
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

    assert multi_label_mask != None
    # Generate multi-label logits.
    with tf.variable_scope("multi_label_logits") as scope:
      multi_label_logits = tf.contrib.layers.fully_connected(
          inputs=image_embeddings,
          num_outputs=get_shape(multi_label_mask)[0],
          activation_fn=None,
          weights_initializer=initializer,
          biases_initializer=None,
          scope=scope)

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

    # add semantic attention here!!!
    multi_label_probs = tf.sigmoid(multi_label_logits) * multi_label_mask
    _, top_concepts_indices = tf.nn.top_k(multi_label_probs, FLAGS.concepts_top_k)
    attention_memory = tf.nn.embedding_lookup(embedding_map, top_concepts_indices)
    if mode == "inference":
      attention_memory = tf.contrib.seq2seq.tile_batch(attention_memory, multiplier=FLAGS.beam_width)


    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=FLAGS.num_lstm_units, state_is_tuple=True)

    if mode == "train":
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=FLAGS.lstm_dropout_keep_prob,
          output_keep_prob=FLAGS.lstm_dropout_keep_prob)

    ## The parameter of attention should be re-considered
    # num_units and attention_layer_size

    attention_mechanism = getattr(tf.contrib.seq2seq, 
      FLAGS.attention_mechanism)(
        num_units = FLAGS.num_attention_depth,
        memory = attention_memory)

    lstm_cell = tf.contrib.seq2seq.AttentionWrapper(
      lstm_cell,
      attention_mechanism,
      attention_layer_size=FLAGS.num_attention_depth,
      output_attention=FLAGS.output_attention)



    with tf.variable_scope("lstm", initializer=initializer) as lstm_scope:
      # Feed the image embeddings to set the initial LSTM state.
      batch_size = get_shape(image_embeddings)[0]

      zero_state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
      _, initial_state = lstm_cell(image_embeddings, zero_state)

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
          initial_state=tf.contrib.seq2seq.tile_batch(initial_state, multiplier=FLAGS.beam_width), #[batch_size*beam_width]
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
      return {"logits": logits, "multi_label_logits": multi_label_logits}
    else:
      return {"bs_results": outputs, "multi_label_probs": multi_label_probs}

