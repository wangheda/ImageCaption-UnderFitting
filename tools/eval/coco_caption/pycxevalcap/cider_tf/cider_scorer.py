#!/usr/bin/env python
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

LOG_TENSOR = True
LOG_NUMPY = False

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

  def ngram2nparray(self, many_ngrams, num_refs=5, n=4, length=30):
    """
    preprocess reference
                     [batch_size, num_refs, num_ngrams, length]
    into numpy array [1,5,4,30], which is then feed into tf program
    other than index array, a length array of shape [1,4] is also generated.
    this may seem peculiar, this is acturally stored as key:value format
    this will be pre-computed and saved into tfrecords
    """
    def count_list(idx_list):
      c = defaultdict(float)
      for idx in idx_list:
        c[idx] += 1
      return c

    many_array_keys = []
    many_array_values = []
    many_array_lengths = []

    for ngram_lists in many_ngrams[:num_refs]:
      assert len(ngram_lists) == n, "%d != %d" % (len(ngram_lists), n)
      ngram_keys = []
      ngram_values = []
      ngram_lengths = []

      for idx_list in ngram_lists:
        idx_dict = count_list(idx_list)
        idx_keys = map(lambda x: x[0], idx_dict.items())
        idx_values = map(lambda x: x[1], idx_dict.items())
        idx_length = len(idx_keys)
        if idx_length >= length:
          idx_length = length
          idx_keys = idx_keys[:length]
          idx_values = idx_values[:length]
        else:
          idx_keys = idx_keys + [0] * (length - idx_length)
          idx_values = idx_values + [0.0] * (length - idx_length)
        ngram_keys.append(idx_keys)
        ngram_values.append(idx_values)
        ngram_lengths.append(idx_length)
      
      many_array_keys.append(ngram_keys)
      many_array_values.append(ngram_values)
      many_array_lengths.append(ngram_lengths)

    if len(many_ngrams) < num_refs:
      print >> sys.stderr, "len(many_ngrams) < num_refs %d < %d" % (len(many_ngrams), num_refs)
      for _i in range(num_refs - len(many_ngrams)):
        many_array_keys.append([[0] * length for __i in range(self.n)])
        many_array_values.append([[0.0] * length for __i in range(self.n)])
        many_array_lengths.append([0] * self.n)

    many_array_keys = np.reshape(np.array(many_array_keys, dtype=np.int64), [1,num_refs,n,length])
    many_array_values = np.reshape(np.array(many_array_values, dtype=np.float), [1,num_refs,n,length])
    actual_lengths = np.reshape(np.array(many_array_lengths, dtype=np.int64), [1,num_refs,n])
    return many_array_keys, many_array_values, actual_lengths

 
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
      if LOG_NUMPY:
        if g is None and l is None:
          print >> sys.stderr, name, eval(name, {"self":self})
        else:
          print >> sys.stderr, name, eval(name, g, l)

    def build_graph():
      self.feed_hyp_words = tf.placeholder(dtype=tf.int64, shape=[1,self.hyp_length])
      log_tensor("self.feed_hyp_words")
      self.feed_hyp_lengths = tf.placeholder(dtype=tf.int64, shape=[1])
      log_tensor("self.feed_hyp_lengths")

      self.feed_ref_keys = tf.placeholder(dtype=tf.int64, shape=[1,self.num_refs,self.n,self.ref_length])
      log_tensor("self.feed_ref_keys")
      self.feed_ref_values = tf.placeholder(dtype=tf.float32, shape=[1,self.num_refs,self.n,self.ref_length])
      log_tensor("self.feed_ref_values")
      self.feed_ref_lengths = tf.placeholder(dtype=tf.int64, shape=[1,self.num_refs,self.n])
      log_tensor("self.feed_ref_lengths")

      self.df_table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(self.df_keys, self.df_values), 0)
      log_tensor("self.df_table")

      self.df_table_init = self.df_table.init
      log_tensor("self.df_table_init")

      def ngram_count(words, lengths):
        n = self.n
        num_refs = self.num_refs
        ref_length = self.ref_length
        length = self.hyp_length
        
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
            tmp_shifted.append(tf.concat([words_idx[:,i:], tf.constant(0, dtype=tf.int64, shape=[1,i])], axis=1))
          tmp_ngram = tf.add_n([x*y for x,y in zip(tmp_shifted, weights)])
          print >> sys.stderr, "tmp_ngram", tmp_ngram

          tmp_ngrams.append(tmp_ngram)  # n-gram ids
          tmp_lengths.append(lengths-i) # bi-gram ids are shorther by 1, etc

        tmp_ngrams = tf.stack(tmp_ngrams, axis=1)
        tmp_lengths = tf.stack(tmp_lengths, axis=1)
        print >> sys.stderr, "tmp_ngrams", tmp_ngrams
        return tmp_ngrams, tmp_lengths
        
      def ngram_overlap(ngrams, ngram_lengths, ref_keys):
        n = self.n
        num_refs = self.num_refs
        ref_length = self.ref_length
        length = self.hyp_length

        tmp_ngrams = tf.reshape(ngrams, shape=[1,1,n,1,length])
        print >> sys.stderr, "tmp_ngrams", ngrams
        tmp_masks = tf.reshape(
                      tf.sequence_mask(
                        tf.reshape(ngram_lengths, shape=[-1]), 
                      maxlen=length), 
                    shape=[1,1,n,1,length])
        print >> sys.stderr, "tmp_masks", tmp_masks
        tmp_ref_keys = tf.reshape(ref_keys, shape=[1,num_refs,n,ref_length,1])
        print >> sys.stderr, "tmp_ref_keys", tmp_ref_keys

        tmp_equal = tf.equal(tmp_ngrams, tmp_ref_keys)
        log_tensor("tmp_equal", l=locals())
        tmp_masked = tf.cast(tmp_equal, dtype=tf.float32) * tf.cast(tmp_masks, dtype=tf.float32)
        log_tensor("tmp_masked", l=locals())
        hyp_count = tf.reduce_sum(tmp_masked, axis=-1)
        log_tensor("hyp_count", l=locals())
        return hyp_count

      def compute_norm_for_hyp(ngrams, ngram_lengths):
        n = self.n
        length = self.hyp_length
        mask = tf.sequence_mask(
                   tf.reshape(ngram_lengths, shape=[-1]), 
                   maxlen=length), 
        float_mask = tf.cast(mask, dtype=tf.float32)
        square_masks = tf.reshape(float_mask, shape=[1,n,length,1]) * tf.reshape(float_mask, shape=[1,n,1,length])
        tmp1_ngrams = tf.reshape(ngrams, shape=[1,n,1,length])
        tmp2_ngrams = tf.reshape(ngrams, shape=[1,n,length,1])
        tmp12_equal = tf.cast(tf.equal(tmp1_ngrams, tmp2_ngrams), dtype=tf.float32)
        text_freq = tf.reduce_sum(tmp12_equal * square_masks, axis=-1) + 1e-9
        doc_freq = self.df_table.lookup(ngrams)
        df_values = tf.log(tf.maximum(doc_freq, 1.0))
        vec = text_freq * tf.maximum(self.ref_len - df_values, 0.0)
        vec_mask = tf.reshape(float_mask, shape=[1,n,length])
        norm = tf.reduce_sum(vec * vec * vec_mask / text_freq, axis=-1)
        norm = tf.tile(tf.reshape(norm, shape=[1,1,self.n]), multiples=[1,self.num_refs,1])
        norm = tf.sqrt(norm)
        return norm

      def counts2vec(keys, values, masks):
        text_freq_values = values
        doc_freq_values = self.df_table.lookup(keys)
        df_values = tf.log(tf.maximum(doc_freq_values, 1.0))
        vec = text_freq_values * tf.maximum(self.ref_len - df_values, 0.0)
        float_masks = tf.cast(masks, dtype=tf.float32)
        norm = tf.sqrt(tf.norm(vec * float_masks, axis=-1))
        return vec, norm

      def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, lengths_hyp, lengths_ref, ref_masks):
        delta = tf.reshape(lengths_hyp, shape=[1,1]) - lengths_ref[:,:,0]
        delta = tf.cast(delta, tf.float32)
        mask = tf.cast(ref_masks, dtype=tf.float32)
        prod = tf.reduce_sum(tf.minimum(vec_hyp, vec_ref) * vec_ref * mask, axis=-1) / (norm_hyp * norm_ref + 1e-9)
        mult = np.e ** (-(delta ** 2) / ((self.sigma ** 2) * 2))
        mult = tf.reshape(mult, shape=[1,self.num_refs,1])
        val = prod * mult
        return val

      def score(sim_val, lengths_ref):
        mask = tf.cast(lengths_ref > 0, dtype=tf.float32)
        score_avg = tf.reduce_sum(sim_val * mask, axis=[1,2]) / (tf.reduce_sum(mask, axis=[1,2]) + 1e-9)
        score_avg = score_avg / tf.reduce_sum(mask[:,:,0], axis=1) *  10.0
        return score_avg

      hyp_ngrams, hyp_ngram_lengths = ngram_count(self.feed_hyp_words, self.feed_hyp_lengths)
      hyp_values = ngram_overlap(hyp_ngrams, hyp_ngram_lengths, self.feed_ref_keys)
      self.hyp_values = hyp_values
      log_tensor("hyp_values", l=locals())
      ref_masks = tf.reshape(
                      tf.sequence_mask(
                          tf.reshape(self.feed_ref_lengths, shape=[-1]), 
                          maxlen=self.ref_length),
                      shape=[1,self.num_refs,self.n,self.ref_length])

      lengths_hyp = self.feed_hyp_lengths
      log_tensor("lengths_hyp", l=locals())
      lengths_ref = self.feed_ref_lengths
      log_tensor("lengths_ref", l=locals())
      log_tensor("ref_masks", l=locals())

      vec_ref, norm_ref = counts2vec(self.feed_ref_keys, self.feed_ref_values, ref_masks)
      log_tensor("vec_ref", l=locals())
      log_tensor("norm_ref", l=locals())

      vec_hyp, _ = counts2vec(self.feed_ref_keys, hyp_values, ref_masks)
      log_tensor("vec_hyp", l=locals())
      norm_hyp = compute_norm_for_hyp(hyp_ngrams, hyp_ngram_lengths)
      log_tensor("norm_hyp", l=locals())

      sim_val = sim(vec_hyp, vec_ref, norm_hyp, norm_ref, lengths_hyp, lengths_ref, ref_masks)
      log_tensor("sim_val", l=locals())
      sim_score = score(sim_val, lengths_ref)
      log_tensor("sim_score", l=locals())
      return sim_score

    scores = []
    with tf.Graph().as_default() as g:
      self.sim_score = build_graph()
      with tf.Session() as sess:
        sess.run(self.df_table_init)

        for test, refs in zip(self.original_test, self.original_refs):
          hyp_words, hyp_lengths = self.wordid2nparray(self.word2id(test), length=self.hyp_length)
          refs_ngrams = [self.id2ngram(self.word2id(ref), n=self.n) for ref in refs]
          ref_keys, ref_values, ref_lengths = self.ngram2nparray(refs_ngrams, num_refs=self.num_refs, n=self.n, length=self.ref_length)
        
          feed_dict = {
              self.feed_hyp_words   : hyp_words,
              self.feed_hyp_lengths : hyp_lengths,
              self.feed_ref_keys    : ref_keys,
              self.feed_ref_values  : ref_values,
              self.feed_ref_lengths : ref_lengths,
          }

          log_array("test", l=locals())
          log_array("refs[0]", l=locals())
          log_array("refs[1]", l=locals())
          log_array("refs[2]", l=locals())
          log_array("refs[3]", l=locals())

          log_array("hyp_words", l=locals())
          log_array("hyp_lengths", l=locals())
          log_array("ref_keys", l=locals())
          log_array("ref_values", l=locals())
          log_array("ref_lengths", l=locals())

          hyp_values, sim_score = sess.run([self.hyp_values, self.sim_score], feed_dict=feed_dict)
          sim_score = sim_score.flatten().tolist()[0]

          log_array("hyp_values", l=locals())
          log_array("sim_score", l=locals())
          scores.append(sim_score)
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

