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

# flags for advanced model
tf.flags.DEFINE_boolean("use_scheduled_sampling", False,
                        "Whether to use scheduled sampling during training.")
tf.flags.DEFINE_string("scheduled_sampling_method", "inverse_sigmoid",
                        "Scheduled sampling method, options are: inverse_sigmoid, linear.")
tf.flags.DEFINE_float("inverse_sigmoid_decay_k", 21000,
                        "The k in inverse_sigmoid_decay method. This setting will half sampling rate at about 210000 steps.")
tf.flags.DEFINE_integer("scheduled_sampling_starting_step", 0,
                        "Set this up properly in finetuning.")
tf.flags.DEFINE_integer("scheduled_sampling_ending_step", 1000000,
                        "Set this up properly in finetuning.")
tf.flags.DEFINE_float("scheduled_sampling_starting_rate", 0.0,
                        "Set this up properly in finetuning.")
tf.flags.DEFINE_float("scheduled_sampling_ending_rate", 0.5,
                        "Set this up properly in finetuning.")

tf.flags.DEFINE_boolean("use_attention_wrapper", False,
                        "Whether to use scheduled sampling during training.")
tf.flags.DEFINE_integer("num_lstm_layers", 1,
                        "The num of layers in lstm model")
tf.flags.DEFINE_integer("num_attention_depth", 128,
                        "The num of atttention depth")
tf.flags.DEFINE_string("attention_mechanism", "BahdanauAttention",
                        "The attention mechanism used in attention wrapper.")
tf.flags.DEFINE_boolean("output_attention", False,
                        "The attention mechanism used in attention wrapper.")


# models
from show_and_tell_model import ShowAndTellModel
from show_and_tell_in_graph_model import ShowAndTellInGraphModel
from show_and_tell_advanced_model import ShowAndTellAdvancedModel

