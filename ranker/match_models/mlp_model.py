
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

class MlpModel(object):

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

    # Map image model output into embedding space.
    with tf.variable_scope("image_embedding") as scope:
      image_embeddings = tf.contrib.layers.fully_connected(
          inputs=image_input,
          num_outputs=FLAGS.embedding_size,
          activation_fn=None,
          weights_initializer=initializer,
          biases_initializer=None,
          scope=scope)

    num_units = map(lambda x: int(x.strip()), FLAGS.mlp_num_units.split(","))
    num_layers = FLAGS.mlp_num_layers

    assert num_layers == len(num_units)

    hidden_layer = image_input
    for num in num_units:
      hidden_layer = tf.contrib.layers.fully_connected(
          inputs=hidden_layer,
          num_outputs=num,
          activation_fn=tf.nn.sigmoid,
          weights_initializer=initializer)

    output_layer = tf.contrib.layers.fully_connected(
        inputs=hidden_layer,
        num_outputs=1,
        activation_fn=None,
        weights_initializer=initializer)

    return output_layer
