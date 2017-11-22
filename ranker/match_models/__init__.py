# flags
import tensorflow as tf
tf.flags.DEFINE_integer("mlp_num_layers", 1,
                        "The num of hidden layers in mlp model")
tf.flags.DEFINE_string("mlp_num_units", "256",
                       "The num of units in hidden layers in mlp model, separated by comma")
tf.flags.DEFINE_string("mlp_type_activation", "tanh",
                       "The type of activation of output layers in mlp model")
tf.flags.DEFINE_string("cos_type_activation", None,
                       "The type of activation of output layers in mlp model")

# models
from mlp_model import MlpModel
from cos_model import CosModel
