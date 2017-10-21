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

tf.flags.DEFINE_integer("concepts_top_k", 10,
                        "The number of concepts to generate attention memory")

tf.flags.DEFINE_integer("num_attention_depth", 128,
                        "The num of atttention depth")
tf.flags.DEFINE_string("attention_mechanism", "BahdanauAttention",
                        "The attention mechanism used in attention wrapper.")
tf.flags.DEFINE_boolean("output_attention", False,
                        "If the attention mechanism used in attention wrapper is Bahdanau, "
                        "this value should be false. If the mechanism is Lung, this value should be set true.")




# models
from show_and_tell_model import ShowAndTellModel
from show_and_tell_in_graph_model import ShowAndTellInGraphModel
from semantic_attention_model import SemanticAttentionModel

