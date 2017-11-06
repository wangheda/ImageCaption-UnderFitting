
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


class ShowAndTellAdvancedModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """

  def create_model(self, input_seqs, image_model_output, initializer, 
                   mode="train", target_seqs=None, input_mask=None, 
                   global_step=None,
                   **unused_params):

    model_outputs = {}

    print "image_model_output", image_model_output

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
    

    # build_seq_embeddings
    with tf.variable_scope("seq_embedding"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[FLAGS.vocab_size, FLAGS.embedding_size],
          initializer=initializer)

      if FLAGS.use_lexical_embedding:
        mapping_types = map(lambda x: x.strip(), FLAGS.lexical_embedding_type.split(","))
        mapping_files = map(lambda x: x.strip(), FLAGS.lexical_mapping_file.split(","))
        mapping_sizes = map(lambda x: int(x.strip()), FLAGS.lexical_embedding_size.split(","))
        for mtype, mfile, msize in zip(mapping_types, mapping_files, mapping_sizes):
          if mtype == "postag":
            lexical_mapping, lexical_size = self.get_lexical_mapping(mfile)
          elif mtype == "char":
            lexical_mapping, lexical_size = self.get_lexical_mapping(mfile)
          else:
            raise Exception("Unknown semantic_attention_type!")

          postfix = "_"+mtype if mtype != "postag" else ""
          lexical_embedding = tf.get_variable(
              name="lexical_map" + postfix,
              shape=[lexical_size, msize],
              initializer=initializer)
          lexical_embedding_map = tf.matmul(lexical_mapping, lexical_embedding)
          embedding_map = tf.concat([embedding_map, lexical_embedding_map], axis=1)
        
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, input_seqs)

    embedding_size = embedding_map.get_shape().as_list()[1]
    # Map image model output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=image_model_output,
          num_outputs=embedding_size,
          activation_fn=None,
          weights_initializer=initializer,
          biases_initializer=None,
          scope=scope)

    # this may be used as auxiliary loss, or as the
    if FLAGS.predict_words_via_image_output:
      with tf.variable_scope("word_prediction"):
        if FLAGS.yet_another_inception:
          word_prediction_image_output = ya_image_model_output
        else:
          word_prediction_image_output = tf.stop_gradient(image_model_output)
        with tf.variable_scope("hidden"):
          word_hidden = tf.contrib.layers.fully_connected(
              inputs=word_prediction_image_output,
              num_outputs=embedding_size,
              activation_fn=tf.nn.relu,
              weights_initializer=initializer,
              biases_initializer=None)
          print word_hidden
        with tf.variable_scope("output"):
          word_predictions = tf.contrib.layers.fully_connected(
              inputs=word_hidden,
              num_outputs=FLAGS.vocab_size,
              activation_fn=tf.nn.sigmoid,
              weights_initializer=initializer,
              biases_initializer=None)
          print word_predictions
        model_outputs["word_predictions"] = word_predictions

    # Save the embedding size in the graph.
    tf.constant(FLAGS.embedding_size, name="embedding_size")

    self.image_embeddings = image_embeddings

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

    if FLAGS.use_attention_wrapper:
      # If mode is inference, copy the middle layer many times
      if mode == "inference":
        middle_layer = tf.contrib.seq2seq.tile_batch(middle_layer, multiplier=FLAGS.beam_width)

      visual_attention_mechanism = getattr(tf.contrib.seq2seq, 
          FLAGS.attention_mechanism)(
              num_units = FLAGS.num_attention_depth,
              memory = middle_layer)


    if FLAGS.use_semantic_attention:
      if FLAGS.use_separate_embedding_for_semantic_attention:
        semantic_embedding_map = tf.get_variable(
            name="semantic_map",
            shape=[FLAGS.vocab_size, FLAGS.embedding_size],
            initializer=initializer)
      else:
        semantic_embedding_map = embedding_map
        
      no_gradient_word_predictions = tf.stop_gradient(word_predictions)

      if FLAGS.semantic_attention_type == "wordhash":
        if FLAGS.weight_semantic_memory_with_hard_prediction:
          word_prior = self.top_k_mask(no_gradient_word_predictions, 
                                       FLAGS.semantic_attention_topk_word)
        elif FLAGS.weight_semantic_memory_with_soft_prediction:
          word_prior = no_gradient_word_predictions

        masked_embedding = tf.einsum("ij,jk->ijk", word_prior, semantic_embedding_map)

        with tf.variable_scope("word_hash"):
          word_hash_map = tf.get_variable(
              name="word_hash_map",
              shape=[FLAGS.vocab_size, FLAGS.semantic_attention_word_hash_depth],
              initializer=initializer)
          word_hasher = tf.nn.softmax(word_hash_map)

        semantic_memory = tf.einsum("ijk,jl->ilk", masked_embedding, word_hasher)
      elif FLAGS.semantic_attention_type == "topk":
        top_probs, top_indices = tf.nn.top_k(no_gradient_word_predictions, 
                                             FLAGS.semantic_attention_topk_word)
        semantic_memory = tf.nn.embedding_lookup(embedding_map, top_indices)
      else:
        raise Exception("Unknown semantic_attention_type!")

      if mode == "inference":
        semantic_memory = tf.contrib.seq2seq.tile_batch(semantic_memory, multiplier=FLAGS.beam_width)

      semantic_attention_mechanism = getattr(tf.contrib.seq2seq, 
          FLAGS.attention_mechanism)(
              num_units = FLAGS.semantic_attention_word_hash_depth,
              memory = semantic_memory)

    if FLAGS.use_attention_wrapper and FLAGS.use_semantic_attention:
      attention_mechanism = [visual_attention_mechanism, semantic_attention_mechanism]
      attention_layer_size = [FLAGS.num_attention_depth, FLAGS.num_attention_depth]
    elif FLAGS.use_attention_wrapper:
      attention_mechanism = visual_attention_mechanism
      attention_layer_size = FLAGS.num_attention_depth
    elif FLAGS.use_semantic_attention:
      attention_mechanism = semantic_attention_mechanism
      attention_layer_size = FLAGS.num_attention_depth
    else:
      attention_mechanism = None
      attention_layer_size = 0

    if attention_mechanism is not None:
      lstm_cell = tf.contrib.seq2seq.AttentionWrapper(
          lstm_cell,
          attention_mechanism,
          attention_layer_size=attention_layer_size,
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
        if FLAGS.use_scheduled_sampling:
          def get_processed_step(step):
            step = tf.maximum(step, FLAGS.scheduled_sampling_starting_step)
            step = tf.minimum(step, FLAGS.scheduled_sampling_ending_step)
            step = tf.maximum(step - FLAGS.scheduled_sampling_starting_step, 0)
            step = tf.cast(step, tf.float32)
            return step
            
          def inverse_sigmoid_decay_fn(step):
            step = get_processed_step(step)
            k = float(FLAGS.inverse_sigmoid_decay_k)
            p = 1.0 - k / (k + tf.exp(step / k))
            return p
          
          def linear_decay_fn(step):
            step = get_processed_step(step)
            slope = (FLAGS.scheduled_sampling_ending_rate - FLAGS.scheduled_sampling_starting_rate) / (FLAGS.scheduled_sampling_ending_step - FLAGS.scheduled_sampling_starting_step)
            a = FLAGS.scheduled_sampling_starting_rate
            p = a + slope * step
            return p

          sampling_fn = {
              "linear": linear_decay_fn, 
              "inverse_sigmoid": inverse_sigmoid_decay_fn
          }

          sampling_probability = sampling_fn[FLAGS.scheduled_sampling_method](global_step)
          tf.summary.scalar("scheduled_sampling/prob", sampling_probability)

          helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=seq_embeddings,
            sequence_length=sequence_length,
            embedding=embedding_map,
            sampling_probability=sampling_probability)
        else:
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
          initial_state=initial_state,
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
      model_outputs["logits"] = logits
    else:
      model_outputs["bs_results"] = outputs

    return model_outputs

  def top_k_mask(self, logits, k = 20):
    values, indices = tf.nn.top_k(logits, k = k)
    sp_idx = tf.where(indices > -1)
    sp_val = tf.gather_nd(indices, sp_idx)
    mask = tf.sparse_to_indicator(
              sp_input = tf.SparseTensor(indices=sp_idx, 
                                         values=sp_val, 
                                         dense_shape=logits.get_shape().as_list()),
              vocab_size = FLAGS.vocab_size)
    mask = tf.cast(mask, dtype=tf.float32) / k
    print("mask", mask)
    return mask

  def get_lexical_mapping(self, lexical_mapping_file):
    mapping = []
    with open(lexical_mapping_file) as F:
      for line in F:
        mapping.append(map(float, line.strip().split()))
    assert len(set(map(len, mapping))) == 1
    lexical_size = len(mapping[0])
    lexical_mapping = tf.constant(mapping, dtype=tf.float32, shape=[FLAGS.vocab_size, lexical_size])
    return lexical_mapping, lexical_size
      
        
