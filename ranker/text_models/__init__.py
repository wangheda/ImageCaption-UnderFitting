import tensorflow as tf
tf.flags.DEFINE_integer("num_lstm_units", 512, "Number of lstm units.") 
tf.flags.DEFINE_integer("num_lstm_layers", 1, "Number of lstm layers.") 
tf.flags.DEFINE_float("lstm_dropout_keep_prob", 0.7, "Number of lstm layers.") 
tf.flags.DEFINE_string("lstm_cell_type", "vanilla", "Number of lstm layers.") 

from lstm_model import LstmModel
