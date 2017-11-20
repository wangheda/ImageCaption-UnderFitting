#!/usr/bin/env python
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

LOG_TENSOR = True
LOG_ARRAY = False

import copy
from collections import defaultdict
import numpy as np
import pdb
import math
import sys

import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("vocab_file", "resources/word_counts.txt",
                       "Vocabulary file.")
tf.flags.DEFINE_integer("max_vocab_size", 10000,
                       "Don't change this.")

class CiderScorer(object):
  """
  CIDEr scorer.
  """

  def load_vocab(self):
    vocab_file = FLAGS.vocab_file
    self.vocab = {}
    with open(vocab_file) as F:
      for line in F:
        cols = line.rstrip().split()
        word = cols[0]
        if word == "":
          word = " "
        idx = len(self.vocab)
        self.vocab[word] = idx
      print "len(self.vocab)", len(self.vocab)
    return None

  def word2id(self, sent):
    if self.vocab is None:
      self.load_vocab()
    OOVid = len(self.vocab)
    words = sent.split()
    ids = map(lambda w: self.vocab.get(w, OOVid), words)
    return ids

  def id2ngram(self, word_ids, n=4):
    ngrams = [[] for i in range(n)]
    for i in range(n):
      for j in range(len(word_ids)-i): # idx = word[0] * 10000**3 + word[1] * 10000**2 + word[2] * 10000**1 + word[3]
        # when computing ngram, we want the word id to start from 1, which is different from other scenarios
        idx = (word_ids[j] + 1)
        for k in range(i):
          idx = idx * FLAGS.max_vocab_size + (word_ids[j+k+1] + 1)
        ngrams[i].append(idx)
    return ngrams

  def compute_doc_freq(self):
    if self.document_frequency is None:
      self.document_frequency = defaultdict(float)
    for refs in self.original_refs:
      ref_ngrams = [self.id2ngram(self.word2id(ref)) for ref in refs]
      for idx in set([ngram_idx for ngram_idx_lists in ref_ngrams 
                                for ngram_idx_list in ngram_idx_lists
                                for ngram_idx in ngram_idx_list]):
        self.document_frequency[idx] += 1
    # For df mapping usage
    self.df_keys = map(lambda x: x[0], self.document_frequency.items())
    self.df_values = map(lambda x: x[1], self.document_frequency.items())
    print >> sys.stderr, "document frequency has %d keys" % len(self.df_keys)

  def wordid2nparray(self, word_ids, length=20):
    """
    preprocess prediction
    into numpy array [1, 20], which is then feed into tf program
    other than index array, a length array of shape [1] is also generated.
    this is the same as beam search generated data
    """
    if len(word_ids) >= length:
      actual_length = length
      word_ids = word_ids[:length]
    else:
      actual_length = len(word_ids)
      word_ids = word_ids + [0] * (length - len(word_ids))
    word_array = np.reshape(np.array(word_ids, dtype=np.int64), [1,length])
    actual_length = np.reshape(np.array([actual_length], dtype=np.int64), [1])
    return word_array, actual_length

  def copy(self):
    ''' copy the refs.'''
    raise NotImplementedError("This should not be called")

  def __init__(self, test=None, refs=None, n=4, sigma=6.0, num_refs=5, hyp_length=20, ref_length=30):
    ''' singular instance '''
    self.vocab = None
    self.n = n
    self.num_refs = num_refs
    self.hyp_length = hyp_length 
    self.ref_length = ref_length
    self.sigma = sigma
    self.crefs = []
    self.ctest = []
    self.document_frequency = None
    self.original_test = []
    self.original_refs = []

  def size(self):
    assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
    return len(self.crefs)

  def __iadd__(self, other):
    '''add an instance (e.g., from another sentence).'''
    if type(other) is tuple:
      ## avoid creating new CiderScorer instances
      self.original_test.append(other[0])
      self.original_refs.append(other[1])
    else:
      self.original_test.extend(other.original_test)
      self.original_refs.extend(other.original_refs)
    return self

  def compute_cider(self):

    def log_tensor(name, g=None, l=None):
      if LOG_TENSOR:
        if g is None and l is None:
          print >> sys.stderr, name, eval(name, {"self":self})
        else:
          print >> sys.stderr, name, eval(name, g, l)

    def log_array(name, g=None, l=None):
      if LOG_ARRAY:
        if g is None and l is None:
          print >> sys.stderr, name, eval(name, {"self":self})
        else:
          print >> sys.stderr, name, eval(name, g, l)

    def build_graph():

      def score(hyp_words, hyp_lengths, ref_words, ref_lengths, sigma=6.0):
        """
        parameters
        hyp_words:   [batch, hyp_length]
        hyp_lengths: [batch]
        ref_words:   [batch, num_refs, hyp_length]
        ref_lengths: [batch, num_refs]
        return
        score:       [batch]
        """

        def ngram_count(words, lengths, n=4):
          shape = words.get_shape().as_list()
          if len(shape) == 2:
            num_sents = 1
            batch, max_length = shape
            words = tf.reshape(words, [batch, num_sents, max_length])
            lengths = tf.reshape(lengths, [batch, num_sents])
          elif len(shape) == 3:
            batch, num_sents, max_length = shape
          else:
            raise NotImplementedError("tensor must be of rank 2 or 3")
          
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
            tmp_lengths.append(lengths-i) # bi-gram ids are shorther by 1, etc

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

          text_freq = tf.reduce_sum(tmp12_equal * square_masks, axis=-1) + 1e-9
          doc_freq = self.df_table.lookup(ngrams)
          df_values = tf.log(tf.maximum(doc_freq, 1.0))

          vec = text_freq * tf.maximum(self.ref_len - df_values, 0.0)
          norm = tf.reduce_sum(vec * vec * float_mask / text_freq, axis=-1)
          norm = tf.sqrt(norm)
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

          delta = tf.cast(hyp_lengths - ref_lengths, tf.float32)

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
                                  axis=-1) / (hyp_tf + 1e-9)
          prod = tf.reduce_sum(tf.reshape(min_vec, [batch, num_sents, n, max_hyp_length, 1])
                                   * tf.reshape(ref_vec, [batch, num_sents, n, 1, max_ref_length])
                                   * equal_masks * square_masks / (freq_masks + 1e-9),
                               axis=[-2,-1])
                               
          mult = np.e ** (-(delta ** 2) / ((sigma ** 2) * 2))
          mask = tf.cast(ref_lengths > 0, dtype=tf.float32)

          scores = prod * tf.expand_dims(mult, axis=2) * tf.expand_dims(mask, axis=2)

          score_avg = tf.reduce_sum(scores, axis=[1,2]) \
                      / (tf.reduce_sum(mask, axis=1) + 1e-9) / (float(n) + 1e-9)
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

      self.feed_hyp_words = tf.placeholder(dtype=tf.int64, shape=[1,self.hyp_length])
      self.feed_hyp_lengths = tf.placeholder(dtype=tf.int64, shape=[1])
      log_tensor("self.feed_hyp_words")
      log_tensor("self.feed_hyp_lengths")

      self.feed_ref_words = tf.placeholder(dtype=tf.int64, shape=[1,self.num_refs,self.ref_length])
      self.feed_ref_lengths = tf.placeholder(dtype=tf.int64, shape=[1,self.num_refs])
      log_tensor("self.feed_ref_words")
      log_tensor("self.feed_ref_lengths")

      self.df_table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(self.df_keys, self.df_values), 0)
      log_tensor("self.df_table")

      self.df_table_init = self.df_table.init
      log_tensor("self.df_table_init")

      sim_score = score(self.feed_hyp_words, self.feed_hyp_lengths, self.feed_ref_words, self.feed_ref_lengths)
      return sim_score

    scores = []
    num_refs = self.num_refs
    with tf.Graph().as_default() as g:
      self.sim_score = build_graph()
      with tf.Session() as sess:
        sess.run(self.df_table_init)

        for test, refs in zip(self.original_test, self.original_refs):
          hyp_words, hyp_lengths = self.wordid2nparray(self.word2id(test), length=self.hyp_length)

          refs_words, refs_lengths = [], []
          for ref in refs[:num_refs]:
            ref_words, ref_lengths = self.wordid2nparray(self.word2id(ref), length=self.ref_length)
            refs_words.append(ref_words)
            refs_lengths.append(ref_lengths)
          for i in xrange(num_refs - len(refs)):
            refs_words.append(np.zeros([1,self.ref_length], dtype=np.int64))
            refs_lengths.append(np.zeros([1], dtype=np.int64))
            
          refs_words = np.stack(refs_words, axis=1)
          refs_lengths = np.stack(refs_lengths, axis=1)

          feed_dict = {
              self.feed_hyp_words   : hyp_words,
              self.feed_hyp_lengths : hyp_lengths,
              self.feed_ref_words   : refs_words,
              self.feed_ref_lengths : refs_lengths,
          }

          sim_score = sess.run(self.sim_score, feed_dict=feed_dict)
          sim_score = sim_score.flatten().tolist()
          scores.extend(sim_score)
    return scores
        
  def compute_score(self, option=None, verbose=0):
    self.ref_len = np.log(float(len(self.original_refs)))
    # compute idf
    self.compute_doc_freq()
    # assert to check document frequency
    assert(len(self.original_test) >= max(self.document_frequency.values()))
    # compute cider score
    score = self.compute_cider()
    # debug
    # print score
    return np.mean(np.array(score)), np.array(score)

