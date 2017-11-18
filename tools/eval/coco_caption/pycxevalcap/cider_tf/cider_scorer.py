#!/usr/bin/env python
# Tsung-Yi Lin <tl483@cornell.edu>
# Ramakrishna Vedantam <vrama91@vt.edu>

import copy
from collections import defaultdict
import numpy as np
import pdb
import math

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
        word, count = line.decode("utf8").rstip().split()
        idx = len(self.vocab) + 1 # starting from 1, make uni-gram's space not overlapping with bi-gram's
        self.vocab[word] = idx
    return None

  def word2id(self, sent)
    if self.vocab is None:
      self.local_vocab()
    OOVid = len(self.local_vocab) + 1 # starting from 1
    words = sent.split()
    ids = map(lambda w: self.vocab.get(w, OOVid), words)
    return ids

  def id2ngram(self, word_ids, n=4):
    ngrams = [[] for i in range(n)]
    for i in range(n):
      for j in range(n-i): # idx = word[0] * 10000**3 + word[1] * 10000**2 + word[2] * 10000**1 + word[3]
        idx = word_ids[j]
        for k in range(i):
          idx = idx * FLAGS.max_vocab_size + word_ids[j+k+1]
        ngrams[i].append(idx)
    return ngrams

  def compute_doc_freq(self):
    if self.document_frequency is None:
      self.document_frequency = defaultdict(float)
    for refs in self.original_refs:
      ref_ngrams = [id2ngram(word2id(ref)) for ref in refs]
      for idx in set([ngram_idx for ngram_idx_lists in ref_ngrams 
                                  for ngram_idx_list in ngram_idx_lists
                                  for ngram_idx in ngram_idx_list]):
        self.document_frequency[idx] += 1
    # For df mapping usage
    self.df_keys = map(lambda x: x[0], self.documnet_frequency)
    self.df_values = map(lambda x: x[1], self.documnet_frequency)
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

    assert len(many_ngrams) == num_refs
    many_array_keys = []
    many_array_values = []
    many_array_lengths = []

    for ngram_lists in many_ngrams:
      assert len(ngram_lists) == n
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
        ngram_lengths.append(idx_lengths)
      
      many_array_keys.append(ngram_keys)
      many_array_values.append(ngram_values)
      many_array_lengths.append(ngram_lengths)

    many_array_keys = np.reshape(np.array(many_array_keys, dtype=np.int64), [1,num_refs,n,length])
    many_array_values = np.reshape(np.array(many_array_values, dtype=np.float), [1,num_refs,n,length])
    actual_length = np.reshape(np.array(many_array_lengths, dtype=np.int64), [1,num_refs,n])
    return many_array_keys, many_array_values, actual_length

 
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
    self.original_test = test
    self.original_refs = refs
    self.ref_len = np.log(float(len(self.original_refs)))

  def size(self):
    assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
    return len(self.crefs)

  def __iadd__(self, other):
    '''add an instance (e.g., from another sentence).'''

    if type(other) is tuple:
      ## avoid creating new CiderScorer instances
      self.cook_append(other[0], other[1])
    else:
      self.ctest.extend(other.ctest)
      self.crefs.extend(other.crefs)

    return self

  def compute_cider(self):

    def build_graph():
      self.feed_hyp_words = tf.placeholder(dtype=tf.int64, shape=[1,self.hyp_length])
      self.feed_hyp_lengths = tf.placeholder(dtype=tf.int64, shape=[1])

      self.feed_ref_keys = tf.placeholder(dtype=tf.int64, shape=[1,self.num_refs,self.n,self.ref_length])
      self.feed_ref_values = tf.placeholder(dtype=tf.float, shape=[1,self.num_refs,self.n,self.ref_length])
      self.feed_ref_lengths = tf.placeholder(dtype=tf.int64, shape=[1,self.num_refs,self.n])

      self.df_table = tf.contrib.lookup.HashTable(
          tf.contrib.lookup.KeyValueTensorInitializer(self.df_keys, self.df_values), 0)
      self.df_table_init = df_table.init

      def ngram_count(words, lengths, ref_keys):
        n = self.n
        num_refs = self.num_refs
        ref_length = self.ref_length
        length = self.hyp_length
        
        tmp_ngrams = []
        tmp_lengths = []

        tmp_shifted = []
        for i in range(n):
          weights = tf.constant([FLAGS.max_vocab_size**k for k in range(i,-1,-1)], dtype=tf.int64, shape=[i+1])
          if i == 0:
            tmp_shifted.append(words)
          else:
            tmp_shifted.append(tf.concat([words[:,i:], tf.constant(0, dtype=tf.int64, shape=[1,i]], axis=1))
          tmp_ngram = tf.einsum("ijk,k->ij", tf.stack(tmp_shifted, axis=2), weights)
          tmp_ngrams.append(tmp_ngram)  # n-gram ids
          tmp_lengths.append(lengths-i) # bi-gram ids are shorther by 1, etc

        tmp_ngrams = tf.stack(tmp_ngrams, axis=1)
        tmp_lengths = tf.stack(tmp_lengths, axis=1)

        tmp_ngrams = tf.reshape(tmp_ngrams, shape=[1,1,n,1,length])
        tmp_masks = tf.reshape(tf.sequence_mask(tmp_lengths, maxlen=length), shape=[1,1,n,1,length])
        tmp_ref_keys = tf.reshape(words, shape=[1,num_refs,n,ref_length,1])

        tmp_masked = tf.logical_and(tmp_words == tmp_ref_keys, tmp_masks)
        hyp_count = tf.reduce_sum(tmp_masked, axis=4)
        return hyp_count

      def counts2vec(keys, values, masks)
        text_freq_values = values
        doc_freq_values = self.df_table.lookup(keys)
        df_values = tf.log(tf.maximum(doc_freq_values, 1.0))
        vec = text_freq_values * tf.maximum(self.ref_len - df_values, 0.0)
        float_masks = tf.cast(masks, dtype=tf.float32)
        norm = tf.norm(tfidf_values * float_masks, axis=-1)
        return vec, norm

      def sim(vec_hyp, vec_ref, norn_hyp, norm_ref, lengths_hyp, lengths_ref):
        # to be continued

    def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
      '''
      Compute the cosine similarity of two vectors.
      :param vec_hyp: array of dictionary for vector corresponding to hypothesis
      :param vec_ref: array of dictionary for vector corresponding to reference
      :param norm_hyp: array of float for vector corresponding to hypothesis
      :param norm_ref: array of float for vector corresponding to reference
      :param length_hyp: int containing length of hypothesis
      :param length_ref: int containing length of reference
      :return: array of score for each n-grams cosine similarity
      '''
      delta = float(length_hyp - length_ref)
      # measure consine similarity
      val = np.array([0.0 for _ in range(self.n)])
      for n in range(self.n):
        # ngram
        for (ngram,count) in vec_hyp[n].iteritems():
          # vrama91 : added clipping
          val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

        if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
          val[n] /= (norm_hyp[n]*norm_ref[n])

        assert(not math.isnan(val[n]))
        # vrama91: added a length based gaussian penalty
        val[n] *= np.e**(-(delta**2)/(2*self.sigma**2))
      return val

    # compute log reference length
    self.ref_len = np.log(float(len(self.crefs)))

    scores = []
    for test, refs in zip(self.ctest, self.crefs):
      # compute vector for test captions
      vec, norm, length = counts2vec(test)
      # compute vector for ref captions
      score = np.array([0.0 for _ in range(self.n)])
      for ref in refs:
          vec_ref, norm_ref, length_ref = counts2vec(ref)
          score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
      # change by vrama91 - mean of ngram scores, instead of sum
      score_avg = np.mean(score)
      # divide by number of references
      score_avg /= len(refs)
      # multiply score by 10
      score_avg *= 10.0
      # append score of an image to the score list
      scores.append(score_avg)
    return scores

  def compute_score(self, option=None, verbose=0):
    # compute idf
    self.compute_doc_freq()
    # assert to check document frequency
    assert(len(self.ctest) >= max(self.document_frequency.values()))
    # compute cider score
    score = self.compute_cider()
    # debug
    # print score
    return np.mean(np.array(score)), np.array(score)
