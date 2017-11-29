
import tensorflow as tf
from tensorflow.python.layers.core import Dense
from top_down_rnn_cell import TopDownRNNCell

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


class TopDownAttentionModel(object):
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

      if input_seqs is not None:
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

    if mode == "inference":
      middle_layer = tf.contrib.seq2seq.tile_batch(middle_layer, multiplier=FLAGS.beam_width)

    att_cell = tf.contrib.rnn.BasicLSTMCell(num_units=FLAGS.num_lstm_units, state_is_tuple=True)
    lm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=FLAGS.num_lstm_units, state_is_tuple=True)
    lstm_cell = TopDownRNNCell(att_cell=att_cell,
                               lm_cell=lm_cell,
                               memory=middle_layer, 
                               attention_size=FLAGS.num_attention_depth, 
                               state_is_tuple=True)

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
        if FLAGS.rl_training == True:
          # use rl train
          # 1. generate greedy captions
          greedy_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            embedding=embedding_map,
            start_tokens=tf.fill([batch_size], FLAGS.start_token),
            end_token=FLAGS.end_token)
          greedy_decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=lstm_cell,
            helper=greedy_helper,
            initial_state=initial_state,
            output_layer=output_layer)
          greedy_outputs, _ , greedy_outputs_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder=greedy_decoder,
            output_time_major=False,
            impute_finished=False,
            maximum_iterations=FLAGS.max_caption_length)

          # 2. generate sample captions
          helper = tf.contrib.seq2seq.SampleEmbeddingHelper(
            embedding=embedding_map,
            start_tokens=tf.fill([batch_size], FLAGS.start_token),
            end_token=FLAGS.end_token)
        else:
          # use cross entropy
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

      if mode == "train":
        if FLAGS.rl_training == True:
          maximum_iterations = FLAGS.max_caption_length
        else:
          maximum_iterations = None
      else:
        maximum_iterations = FLAGS.max_caption_length

      outputs, _ , outputs_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
        decoder=decoder,
        output_time_major=False,
        impute_finished=False,
        maximum_iterations=maximum_iterations)

    if mode == "train":
      if FLAGS.rl_training == True:
        return {"sample_caption_words"   : outputs.sample_id,
                "sample_caption_logits"  : outputs.rnn_output,
                "sample_caption_lengths" : outputs_sequence_lengths,
                "greedy_caption_words"   : greedy_outputs.sample_id,
                "greedy_caption_lengths" : greedy_outputs_sequence_lengths}
      else:
        logits = tf.reshape(outputs.rnn_output, [-1, FLAGS.vocab_size])
        model_outputs["logits"] = logits
    else:
      model_outputs["bs_results"] = outputs

    return model_outputs

