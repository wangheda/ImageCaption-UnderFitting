
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


def cosine_distance(tensor1, tensor2, axis=1):
  dot_product = tf.reduce_mean(tensor1 * tensor2, axis=axis, keep_dims=True)
  length1 = tf.sqrt(tf.reduce_mean(tensor1 * tensor1, axis=axis, keep_dims=True))
  length2 = tf.sqrt(tf.reduce_mean(tensor2 * tensor2, axis=axis, keep_dims=True))
  length = length1 * length2 + 1e-9
  return dot_product / length


class CosModel(object):

  def create_model(self, image_input, text_input, initializer=None, 
                   mode="train", global_step=None,
                   **unused_params):

    model_outputs = {}

    if type(image_input) in [list, tuple]:
      image_input, middle_layer = image_input
      print image_input
      print middle_layer
    
    if type(text_input) in [list, tuple]:
      text_input, sequence_layer = text_input
      print text_input
      print sequence_layer

    if FLAGS.cos_type_activation is not None:
      activation_fn = getattr(tf.nn, FLAGS.cos_type_activation)

    # Map image model output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=image_input,
          num_outputs=FLAGS.embedding_size,
          activation_fn=activation_fn,
          weights_initializer=initializer,
          biases_initializer=None,
          scope=scope)

    with tf.variable_scope("text_embedding") as scope:
      text_embeddings = tf.contrib.layers.fully_connected(
          inputs=text_input,
          num_outputs=FLAGS.embedding_size,
          activation_fn=activation_fn,
          weights_initializer=initializer,
          biases_initializer=None,
          scope=scope)

    output_layer = cosine_distance(image_embeddings, text_embeddings)
    return output_layer
