# flags
import tensorflow as tf
tf.flags.DEFINE_float("lstm_dropout_keep_prob", 0.7,
                        "If < 1.0, the dropout keep probability applied to LSTM variables..")
tf.flags.DEFINE_integer("num_lstm_units", 512,
                        "Lstm output dimension")

# models
from show_and_tell_model import ShowAndTellModel
from show_and_tell_in_graph_model import ShowAndTellInGraphModel

