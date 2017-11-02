
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
                   mode="train", target_seqs=None, input_mask=None,
                   **unused_params):
    
    attributes_mask, idf_weighted_mask = self.build_attributes_mask()

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

    # Generate multi-label logits.
    with tf.variable_scope("attributes_logits") as scope:
      attributes_logits = tf.contrib.layers.fully_connected(
          inputs=image_model_output,
          num_outputs=get_shape(attributes_mask)[0],
          activation_fn=None,
          weights_initializer=initializer,
          #biases_initializer=None,
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
    attributes_probs = tf.sigmoid(attributes_logits) * attributes_mask
    top_attributes_probs, top_attributes_indices = tf.nn.top_k(attributes_probs, FLAGS.attributes_top_k)
    attention_memory = tf.nn.embedding_lookup(embedding_map, top_attributes_indices)
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
      return {"logits": logits, 
              "attributes_logits": attributes_logits, 
              "attributes_mask": attributes_mask,
              "idf_weighted_mask": idf_weighted_mask}
    else:
      return {"bs_results": outputs, "top_n_attributes": (top_attributes_probs, top_attributes_indices)}

  def build_attributes_mask(self, attributes_num=1000):
    input = open(FLAGS.vocab_file)
    vocab = [line.split(" ")[0] for line in input.readlines()]
    input.close()
    input = open(FLAGS.attributes_file)
    attributes = set([line.split(" ")[0] for line in input.readlines()])
    input.close()
    idf_weighted_mask = None
    if FLAGS.use_idf_weighted_attribute_loss:
      idf_mask = []
      input = open(FLAGS.word_idf_file)
      word_idf_dict = {}
      for line in input.readlines():
        word, idf = line.strip().split(" ")
        word_idf_dict[word] = idf
      input.close()
    index = 0
    mask = []
    for word in vocab:
      if word in attributes:
        mask.append(1.0)
        if FLAGS.use_idf_weighted_attribute_loss:
          idf_mask.append(word_idf_dict[word])
        index += 1
        if index == attributes_num:
          break
      else:
        mask.append(0)
        if FLAGS.use_idf_weighted_attribute_loss:
          idf_mask.append(0)

    attributes_mask = tf.Variable(mask,
                                   trainable=False,
                                   name="attributes_mask")
    if FLAGS.use_idf_weighted_attribute_loss:
      idf_weighted_mask = tf.Variable(mask,
                                     trainable=False,
                                     name="idf_weighted_mask")
    return attributes_mask, idf_weighted_mask

