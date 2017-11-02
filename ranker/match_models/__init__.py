# flags
import tensorflow as tf
tf.flags.DEFINE_integer("mlp_num_layers", 1,
                        "The num of hidden layers in mlp model")
tf.flags.DEFINE_string("mlp_num_units", "256",
                       "The num of units in hidden layers in mlp model, separated by comma")

# models
from mlp_model import MlpModel
