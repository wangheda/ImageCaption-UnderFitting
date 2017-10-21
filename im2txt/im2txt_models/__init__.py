# flags
import tensorflow as tf
tf.flags.DEFINE_float("lstm_dropout_keep_prob", 0.7,
                        "If < 1.0, the dropout keep probability applied to LSTM variables..")
tf.flags.DEFINE_integer("num_lstm_units", 512,
                        "Lstm output dimension")

tf.flags.DEFINE_integer("start_token", 1,
                        "The start token id")
tf.flags.DEFINE_integer("end_token", 2,
                        "The end token id")
tf.flags.DEFINE_integer("beam_width", 3,
                        "The beam width")
tf.flags.DEFINE_integer("max_caption_length", 20,
                        "The max caption length for beam search decoding")


# models
from show_and_tell_model import ShowAndTellModel
from show_and_tell_in_graph_model import ShowAndTellInGraphModel
from show_attend_tell_model import ShowAttendTellModel
