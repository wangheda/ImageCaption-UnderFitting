
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import custom_rnn_cell

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

# padd or truncate the given axis of tensor to max_length
def pad_or_truncate(tensor, lengths, axis, max_length):
  shape = tensor.get_shape().as_list()
  real_shape = tf.shape(tensor)
  target_shape = [l for l in shape]
  target_shape[axis] = max_length

  left_padding, right_padding = [0] * len(shape), [0] * len(shape)
  right_padding[axis] = tf.maximum(max_length - real_shape[axis], 0)
  padded_tensor = tf.pad(tensor, zip(left_padding, right_padding))
  sliced_tensor = tf.slice(padded_tensor, [0] * len(shape), target_shape)
  clipped_lengths = tf.minimum(lengths, max_length)
  return sliced_tensor, clipped_lengths

def scheduled_sampling_prob(global_step):
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
  return sampling_probability


class MultiRefModel(object):
  """Image-to-text implementation based on http://arxiv.org/abs/1411.4555.

  "Show and Tell: A Neural Image Caption Generator"
  Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
  """
  def set_state(self):
    if self.mode == "train":
      if FLAGS.rl_training and FLAGS.rl_training_along_with_mle:
        self.train_rl = True
        self.train_mle = True
      elif FLAGS.rl_training:
        self.train_rl = True
        self.train_mle = False
      else:
        self.train_rl = False
        self.train_mle = True
    else:
      self.train_rl = False
      self.train_mle = False


  def create_model(self, input_seqs, image_model_output, initializer, 
                   mode="train", target_seqs=None, input_mask=None, 
                   global_step=None,
                   **unused_params):
    """
    input_seqs: [batch_size, num_refs, max_ref_length]
    image_model_output: 
          1. [batch_size, image_embedding_size]
          2. [batch_size, image_positions, image_embedding_size]
    """
    self.mode = mode
    self.set_state()

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

    # Save the embedding size in the graph.
    tf.constant(FLAGS.embedding_size, name="embedding_size")

    self.image_embeddings = image_embeddings

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
    elif FLAGS.lstm_cell_type in ["residual", "highway"]: # fast_forward is not ready for use
      wrapper_class_dict = {"residual":     tf.contrib.rnn.ResidualWrapper,
                            "highway":      tf.contrib.rnn.HighwayWrapper,
                            "fast_forward": custom_rnn_cell.FastForwardWrapper}
      wrapper_class = wrapper_class_dict.get(FLAGS.lstm_cell_type)

      if FLAGS.num_lstm_layers > 1:
        lstm_cell = tf.contrib.rnn.MultiRNNCell([
                    wrapper_class(
                        cell=tf.contrib.rnn.BasicLSTMCell(
                        num_units=FLAGS.num_lstm_units, state_is_tuple=True)
                    )
                    for i in xrange(FLAGS.num_lstm_layers)], state_is_tuple=True)
      else:
        lstm_cell = wrapper_class(
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

    if FLAGS.use_attention_wrapper:
      attention_mechanism = visual_attention_mechanism
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
      output_layer = Dense(units=FLAGS.vocab_size,
                           name="output_layer")

      if mode == "train":
        # initial state
        zero_state = lstm_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
        _, initial_state = lstm_cell(image_embeddings, zero_state)

        if self.train_rl:
          # use rl train
          maximum_iterations = FLAGS.max_caption_length

          # 1. generate greedy captions
          rl_greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=embedding_map,
            start_tokens=tf.fill([batch_size], FLAGS.start_token),
            end_token=FLAGS.end_token)
          rl_greedy_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=lstm_cell,
            helper=rl_greedy_helper,
            initial_state=initial_state,
            output_layer=output_layer)
          rl_greedy_outputs, _ , rl_greedy_outputs_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder=rl_greedy_decoder,
            output_time_major=False,
            impute_finished=False,
            maximum_iterations=FLAGS.max_caption_length)

          # 2. generate sample captions
          rl_sampling_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
            embedding=embedding_map,
            start_tokens=tf.fill([batch_size], FLAGS.start_token),
            end_token=FLAGS.end_token)
          rl_sampling_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=lstm_cell,
            helper=rl_sampling_helper,
            initial_state=initial_state,
            output_layer=output_layer)
          rl_sampling_outputs, _ , rl_sampling_outputs_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder=rl_sampling_decoder,
            output_time_major=False,
            impute_finished=False,
            maximum_iterations=maximum_iterations)

        if self.train_mle:
          # pre-process
          maximum_iterations = FLAGS.max_ref_length
          batch_size, num_refs, max_length, emb_size = seq_embeddings.get_shape().as_list()

          # pre-process
          all_mle_seq_embeddings = tf.reshape(seq_embeddings, shape=[num_refs, batch_size, max_length, emb_size])
          all_mle_input_masks = tf.cast(tf.reshape(input_mask, shape=[num_refs, batch_size, max_length]), dtype=tf.int32)

          # mle training
          def mle_training(mle_seq_embeddings, mle_input_masks):
            mle_sequence_length = tf.reduce_sum(mle_input_masks, -1)
            if FLAGS.use_scheduled_sampling:
              sampling_probability = scheduled_sampling_prob(global_step)
              mle_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                inputs=mle_seq_embeddings,
                sequence_length=mle_sequence_length,
                embedding=embedding_map,
                sampling_probability=sampling_probability)
            else:
              mle_helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=mle_seq_embeddings,
                sequence_length=mle_sequence_length)

            mle_decoder = tf.contrib.seq2seq.BasicDecoder(
              cell=lstm_cell,
              helper=mle_helper,
              initial_state=initial_state,
              output_layer=output_layer)

            mle_outputs, _ , mle_outputs_lengths = tf.contrib.seq2seq.dynamic_decode(
              decoder=mle_decoder,
              output_time_major=False,
              impute_finished=False,
              maximum_iterations=maximum_iterations)
            return mle_outputs.rnn_output, mle_outputs_lengths

          all_mle_seq_embeddings = tf.unstack(all_mle_seq_embeddings, axis=0)
          all_mle_input_masks = tf.unstack(all_mle_input_masks, axis=0)
          all_mle_caption_logits = []
          for mle_seq_embeddings, mle_input_masks in zip(all_mle_seq_embeddings, all_mle_input_masks):
            all_mle_caption_logits.append(mle_training(mle_seq_embeddings, mle_input_masks))

          all_mle_caption_logits = tf.stack([pad_or_truncate(logits, lengths, axis=1, max_length=maximum_iterations)[0]
                                             for logits, lengths in all_mle_caption_logits], axis=0)

          mle_caption_logits = tf.reshape(all_mle_caption_logits, 
                                          shape=[batch_size, num_refs, maximum_iterations, FLAGS.vocab_size])

      elif mode == "inference":
        image_embeddings = tf.contrib.seq2seq.tile_batch(image_embeddings, multiplier=FLAGS.beam_width)
        zero_state = lstm_cell.zero_state(batch_size=batch_size*FLAGS.beam_width, dtype=tf.float32)
        _, initial_state = lstm_cell(image_embeddings, zero_state)

        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
          cell=lstm_cell,
          embedding=embedding_map,
          start_tokens=tf.fill([batch_size], FLAGS.start_token),    #[batch_size]
          end_token=FLAGS.end_token,
          initial_state=initial_state,
          beam_width=FLAGS.beam_width,
          output_layer=output_layer,
          length_penalty_weight=0.0)

        maximum_iterations = FLAGS.max_caption_length
        outputs, _ , outputs_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
          decoder=decoder,
          output_time_major=False,
          impute_finished=False,
          maximum_iterations=maximum_iterations)
      else:
        raise Exception("Unknown mode!")

    if mode == "train":
      model_outputs = {}
      if self.train_rl:
        model_outputs.update({"sample_caption_words"   : rl_sampling_outputs.sample_id,
                              "sample_caption_logits"  : rl_sampling_outputs.rnn_output,
                              "sample_caption_lengths" : rl_sampling_outputs_lengths,
                              "greedy_caption_words"   : rl_greedy_outputs.sample_id,
                              "greedy_caption_lengths" : rl_greedy_outputs_lengths})
      if self.train_mle:
        model_outputs.update({"mle_caption_logits"     : mle_caption_logits})
    else:
      model_outputs["bs_results"] = outputs

    return model_outputs

