
import json
import tensorflow as tf
import numpy as np
import sys

tf.flags.DEFINE_integer("max_vocab_size", 10000,
                       "Don't change this.")
tf.flags.DEFINE_string("document_frequency_file", 
                       "data/document_frequency.json", 
                       "File containing the document frequency infos.")

FLAGS = tf.app.flags.FLAGS

LOG_TENSOR = True

def log_tensor(name, g=None, l=None):
  if LOG_TENSOR:
    if g is None and l is None:
      print >> sys.stderr, name, eval(name, {"self":self})
    else:
      print >> sys.stderr, name, eval(name, g, l)

def get_rank(tensor):
  return len(get_shape_as_list(tensor))

def get_shape_as_list(tensor):
  return tensor.get_shape().as_list()

def get_shape(tensor):
  """Returns static shape if available and dynamic shape otherwise."""
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims

class CiderScorer(object):
  def __init__(self):
    with open(FLAGS.document_frequency_file, 'r') as f:
      df_data = json.load(f)
    df_keys = df_data['df_keys']
    df_values = df_data['df_values']
    ref_len = df_data['ref_len']
    self.df_table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(
                df_keys, df_values,
                key_dtype=tf.int64, value_dtype=tf.float32), default_value = 0.0)
    self.ref_len = tf.constant(ref_len)

  def score(self, hyp_words, hyp_lengths, ref_words, ref_lengths, sigma=6.0):
    """
    parameters
    hyp_words:   [batch, hyp_length]
    hyp_lengths: [batch]
    ref_words:   [batch, num_refs, hyp_length]
    ref_lengths: [batch, num_refs]
    return
    score:       [batch]
    """

    hyp_words = tf.cast(hyp_words, dtype=tf.int64)
    ref_words = tf.cast(ref_words, dtype=tf.int64)
    hyp_lengths = tf.cast(hyp_lengths, dtype=tf.int64)
    ref_lengths = tf.cast(ref_lengths, dtype=tf.int64)

    if get_rank(hyp_words) == 2:
      hyp_words = tf.expand_dims(hyp_words, axis=1)
      hyp_lengths = tf.expand_dims(hyp_lengths, axis=1)
    if get_rank(ref_words) == 2:
      ref_words = tf.expand_dims(ref_words, axis=1)
      ref_lengths = tf.expand_dims(ref_lengths, axis=1)

    log_tensor("hyp_words", l=locals())
    log_tensor("hyp_lengths", l=locals())
    log_tensor("ref_words", l=locals())
    log_tensor("ref_lengths", l=locals())

    def ngram_count(words, lengths, n=4):
      shape = words.get_shape().as_list()
      if len(shape) == 3:
        batch, num_sents, max_length = shape
      else:
        raise NotImplementedError("tensor must be of rank 3")
      
      tmp_ngrams = []
      tmp_lengths = []
      tmp_shifted = []

      words_idx = words + 1
      log_tensor("words_idx", l=locals())

      for i in range(n):
        weights = [FLAGS.max_vocab_size**k for k in range(i,-1,-1)]
        if i == 0:
          tmp_shifted.append(words_idx)
        else:
          tmp_shifted.append(tf.concat([words_idx[:,:,i:], tf.constant(0, dtype=tf.int64, shape=[batch,num_sents,i])], axis=-1))
        tmp_ngram = tf.add_n([x*y for x,y in zip(tmp_shifted, weights)])
        log_tensor("tmp_ngram", l=locals())

        tmp_ngrams.append(tmp_ngram)  # n-gram ids
        tmp_lengths.append(tf.maximum(lengths-i, 0)) # bi-gram ids are shorther by 1, etc

      tmp_ngrams = tf.stack(tmp_ngrams, axis=2)
      tmp_lengths = tf.stack(tmp_lengths, axis=2)
      log_tensor("tmp_ngrams", l=locals())
      log_tensor("tmp_lengths", l=locals())
      return tmp_ngrams, tmp_lengths
    
    def compute_vec_norm_and_freq(ngrams, ngram_lengths):
      """
      parameters
      ngrams        : [batch, num_sents, n, max_length]
      ngram_lengths : [batch, num_sents, n]
      return
      vec           : [batch, num_sents, n, max_length] tfidf values of every ngram
      norm          : [batch, num_sents, n]
      text_freq     : [batch, num_sents, n, max_length]
      """
      shape = ngrams.get_shape().as_list()
      batch, num_sents, n, max_length = shape

      mask = tf.reshape(
                tf.sequence_mask(
                  tf.reshape(ngram_lengths, shape=[-1]), 
                  maxlen=max_length), 
                shape=[batch, num_sents, n, max_length])
      float_mask = tf.cast(mask, dtype=tf.float32)

      square_masks = tf.reshape(float_mask, shape=[batch, num_sents, n, max_length, 1]) \
                     * tf.reshape(float_mask, shape=[batch, num_sents, n, 1, max_length])

      tmp1_ngrams = tf.reshape(ngrams, shape=[batch, num_sents, n, 1, max_length])
      tmp2_ngrams = tf.reshape(ngrams, shape=[batch, num_sents, n, max_length, 1])
      tmp12_equal = tf.cast(tf.equal(tmp1_ngrams, tmp2_ngrams), dtype=tf.float32)

      text_freq = tf.reduce_sum(tmp12_equal * square_masks, axis=-1)
      doc_freq = self.df_table.lookup(ngrams)
      df_values = tf.log(tf.maximum(doc_freq, 1.0))

      tf.summary.histogram("cider/document_freq", doc_freq)
      tf.summary.histogram("cider/text_freq", text_freq)
      tf.summary.histogram("cider/df_values", df_values)

      vec = text_freq * tf.maximum(self.ref_len - df_values, 0.0)
      norm = tf.reduce_sum(vec * vec * float_mask / (text_freq + 1e-12), axis=-1)
      norm = tf.sqrt(norm)
      tf.summary.histogram("cider/vec", vec)
      tf.summary.histogram("cider/norm", norm)
      return vec, norm, text_freq

    def sim(hyp_vec, hyp_norm, hyp_tf, hyp_lengths, hyp_ngrams, hyp_ngram_lengths,
            ref_vec, ref_norm, ref_tf, ref_lengths, ref_ngrams, ref_ngram_lengths,
            sigma=6.0):
      """
      parameters
      vec           : [batch, num_sents, n, max_length] tfidf values of every ngram
      norm          : [batch, num_sents, n]
      tf            : [batch, num_sents, n, max_length]
      lengths       : [batch, num_sents, n]
      ngrams        : [batch, num_sents, n, max_length]
      ngram_lengths : [batch, num_sents, n]
      return 
      score         : [batch]
      """
      batch, num_sents, n, max_hyp_length = hyp_vec.get_shape().as_list()
      _, _, _, max_ref_length = ref_vec.get_shape().as_list()
      
      log_tensor("hyp_vec", l=locals())
      log_tensor("hyp_norm", l=locals())
      log_tensor("hyp_ngrams", l=locals())
      log_tensor("hyp_ngram_lengths", l=locals())
      log_tensor("ref_vec", l=locals())
      log_tensor("ref_norm", l=locals())
      log_tensor("ref_ngrams", l=locals())
      log_tensor("ref_ngram_lengths", l=locals())

      delta = tf.cast(hyp_lengths - ref_lengths, tf.float32)
      log_tensor("delta", l=locals())

      ref_masks = tf.cast(tf.reshape(
                    tf.sequence_mask(
                      tf.reshape(ref_ngram_lengths, shape=[-1]), 
                      maxlen=max_ref_length), 
                    shape=[batch, num_sents, n, max_ref_length]), dtype=tf.float32)
      hyp_masks = tf.cast(tf.reshape(
                    tf.sequence_mask(
                      tf.reshape(hyp_ngram_lengths, shape=[-1]), 
                      maxlen=max_hyp_length), 
                    shape=[batch, num_sents, n, max_hyp_length]), dtype=tf.float32)
      square_masks = tf.reshape(hyp_masks, shape=[batch, num_sents, n, max_hyp_length, 1]) \
                     * tf.reshape(ref_masks, shape=[batch, num_sents, n, 1, max_ref_length])
      freq_masks = tf.reshape(hyp_tf, shape=[batch, num_sents, n, max_hyp_length, 1]) \
                     * tf.reshape(ref_tf, shape=[batch, num_sents, n, 1, max_ref_length])
      equal_masks = tf.cast(tf.equal(
                         tf.reshape(hyp_ngrams, shape=[batch, num_sents, n, max_hyp_length, 1]), 
                         tf.reshape(ref_ngrams, shape=[batch, num_sents, n, 1, max_ref_length])
                                    ), dtype=tf.float32)
      min_vec = tf.reduce_sum(tf.minimum(
                                 tf.reshape(hyp_vec, [batch, num_sents, n, max_hyp_length, 1]),
                                 tf.reshape(ref_vec, [batch, num_sents, n, 1, max_ref_length]))
                                 * equal_masks * square_masks,
                              axis=-1) / (hyp_tf + 1e-12)
      prod = tf.reduce_sum(tf.reshape(min_vec, [batch, num_sents, n, max_hyp_length, 1])
                               * tf.reshape(ref_vec, [batch, num_sents, n, 1, max_ref_length])
                               * equal_masks * square_masks / (freq_masks + 1e-12),
                           axis=[-2,-1])

      val = prod / (hyp_norm * ref_norm + 1e-12)
      log_tensor("val", l=locals())
                           
      mult = np.e ** (-(delta ** 2) / ((sigma ** 2) * 2))
      mask = tf.cast(ref_lengths > 0, dtype=tf.float32)

      scores = val * tf.expand_dims(mult, axis=2) * tf.expand_dims(mask, axis=2)
      tf.summary.histogram("cider/scores", scores)

      log_tensor("scores", l=locals())

      score_avg = tf.reduce_sum(scores, axis=[1,2]) \
                  / (tf.reduce_sum(mask, axis=1) * float(n) + 1e-12)
      score_avg = score_avg * 10.0
      return score_avg

    def tile_on_axis(tensor, axis=1, copies=5):
      shape = tensor.get_shape().as_list()
      multiples = [1] * len(shape)
      multiples[axis] = copies
      return tf.tile(tensor, multiples=multiples)

    ref_ngrams, ref_ngram_lengths = ngram_count(ref_words, ref_lengths)
    ref_vec, ref_norm, ref_text_freq = compute_vec_norm_and_freq(ref_ngrams, ref_ngram_lengths)

    hyp_ngrams, hyp_ngram_lengths = ngram_count(hyp_words, hyp_lengths)
    hyp_vec, hyp_norm, hyp_text_freq = compute_vec_norm_and_freq(hyp_ngrams, hyp_ngram_lengths)

    ref_vec_shape = ref_vec.get_shape().as_list()
    num_refs = ref_vec_shape[1]
    hyp_ngrams, hyp_ngram_lengths, hyp_vec, hyp_norm, hyp_text_freq = map(
            lambda x: tile_on_axis(x, axis=1, copies=num_refs),
            [hyp_ngrams, hyp_ngram_lengths, hyp_vec, hyp_norm, hyp_text_freq])

    sim_score = sim(hyp_vec, hyp_norm, hyp_text_freq, hyp_lengths, hyp_ngrams, hyp_ngram_lengths,
                    ref_vec, ref_norm, ref_text_freq, ref_lengths, ref_ngrams, ref_ngram_lengths,
                    sigma=sigma)
    return sim_score
