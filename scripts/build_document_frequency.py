#!/usr/bin/env python


import copy
from collections import defaultdict
import numpy as np
import pdb
import math
import sys
import jieba

import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("vocab_file", "resources/word_counts.txt",
                       "Vocabulary file.")
tf.flags.DEFINE_integer("max_vocab_size", 10000,
                       "Don't change this.")
tf.flags.DEFINE_string("annotation_file", "",
                       "File that contain annotations.")
tf.flags.DEFINE_string("output_file", "",
                       "Output file contain document frequency infos")

class CiderScorer(object):
  """
  Only for stat document_frequency
  copy from cider_tf/cider_scorer.py
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

  def __init__(self, test=None, refs=None, n=4, sigma=6.0, num_refs=5):
    ''' singular instance '''
    self.vocab = None
    self.n = n
    self.num_refs = num_refs
    self.sigma = sigma
    self.document_frequency = None
    self.original_test = []
    self.original_refs = []

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


def main():
    cider_scorer = CiderScorer()
    with open(FLAGS.annotation_file, 'r') as f:
        caption_data = json.load(f)

    for data in caption_data:
        image_name = data['image_id'].split('.')[0]
        captions = data['caption']
        refs = []
        for caption in captions:
            w = jieba.cut(ann['caption'].strip().replace('。',''), cut_all=False)
            p = ' '.join(w)
            refs.append(p)
        cider_scorer += (None, refs)

    cider_scorer.compute_doc_freq()
    df_data = {}
    df_data['df_keys'] = cider_scorer.df_keys
    df_data['df_values'] = cider_scorer.df_values
    df_data['ref_len'] = np.log(float(len(cider_scorer.original_refs)))
    output = open(FLAGS.output_file, 'w')
    json.dump(df_data, output, indent=4)
    output.close()

if __name__ == "__main__":
    tf.app.run()




    

  