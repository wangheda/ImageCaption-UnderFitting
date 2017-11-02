# coding: utf-8
# Copyright 2017 challenger.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Build tfrecord data."""
# __author__ = 'Miao'
# python2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
from collections import namedtuple
from datetime import datetime
import json
import os.path
import random
import sys

reload(sys)
sys.setdefaultencoding('utf8')
import threading
import jieba
import jieba.posseg as pseg
import numpy as np
import tensorflow as tf

# input data
tf.flags.DEFINE_string("train_captions_file", "data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json",
                       "Training captions JSON file.")

# use existing word counts file
tf.flags.DEFINE_string("word_counts_input_file",
                       "",
                       "If defined, use existing word_counts_file.")

# output files
tf.flags.DEFINE_string("output_dir", "data/TFRecord_data", "Output directory for tfrecords.")
tf.flags.DEFINE_string("postags_output_file", "resources/postags.txt",
                       "Output vocabulary file of pos-tag counts.")
tf.flags.DEFINE_string("word2postag_output_file", "resources/word2postag.txt",
                       "Output mapping file of word to pos-tag.")

# words parameters
tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")

FLAGS = tf.flags.FLAGS


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self, vocab, unk_id):
        """Initializes the vocabulary.
        Args:
          vocab: A dictionary of word to word_id.
          unk_id: Id of the special 'unknown' word.
        """
        self._vocab = vocab
        self._unk_id = unk_id

    def word_to_id(self, word):
        """Returns the integer id of a word string."""
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self._unk_id

    def keys(self):
        return self._vocab.keys()

    def values(self):
        return self._vocab.values()

    def __len__(self):
        return self._vocab.__len__()



def load_vocab(vocab_file):
    if not tf.gfile.Exists(vocab_file):
      print("Vocab file %s not found.", vocab_file)
      exit()
    print("Initializing vocabulary from file: %s", vocab_file)

    with tf.gfile.GFile(vocab_file, mode="r") as f:
      reverse_vocab = list(f.readlines())
    reverse_vocab = [line.split()[0].decode('utf-8') for line in reverse_vocab]
    assert FLAGS.start_word in reverse_vocab
    assert FLAGS.end_word in reverse_vocab
    assert FLAGS.unknown_word not in reverse_vocab

    unk_id = len(reverse_vocab)
    vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
    vocab = Vocabulary(vocab_dict, unk_id)
    return vocab


def _process_caption_jieba(caption):
    """Processes a Chinese caption string into a list of tonenized words.
    Args:
      caption: A string caption.
    Returns:
      A list of strings; the tokenized caption.
    """
    tokenized_caption = [FLAGS.start_word]
    tokenized_caption.extend(jieba.cut(caption, cut_all=False))
    tokenized_caption.append(FLAGS.end_word)
    return tokenized_caption


def _load_and_process_captions(captions_file):
    """Loads all captions from a JSON file and processes the captions.
    Args:
      captions_file: Json file containing caption annotations.
    Returns:
      A list of captions
    """
    captions = []
    num_captions = 0
    with open(captions_file, 'r') as f:
        caption_data = json.load(f)
    for data in caption_data:
        image_name = data['image_id'].split('.')[0]
        descriptions = data['caption']

        caption_num = len(descriptions)
        for i in range(caption_num):
            caption_temp = descriptions[i].strip().strip("。").replace('\n', '')
            if caption_temp != '':
                captions.append(caption_temp)
                num_captions += 1

    print("Finished reading %d captions for %d images in %s" %
          (num_captions, len(caption_data), captions_file))
    return captions

def _process_postags(captions, vocab):
    word2postag = {}
    postags = {}

    num_processed = 0
    postags[FLAGS.start_word] = 0
    for caption in captions:

        if num_processed % 1000 == 0:
            print("%d / %d" % (num_processed, len(captions)))
            if num_processed % 10000 == 0:
                print(postags)
        num_processed += 1

        for word, tag in pseg.cut(caption):
            w_id = vocab.word_to_id(word)
            if tag in postags:
                postags[tag] += 1
            else:
                postags[tag] = 1
            if w_id in word2postag:
                w2p = word2postag[w_id]
                if tag in w2p:
                    w2p[tag] += 1
                else:
                    w2p[tag] = 1
            else:
                word2postag[w_id] = {tag: 1}

    # special pos tag
    word2postag[vocab.word_to_id(FLAGS.start_word)] = {FLAGS.start_word: 1}

    prior = {}
    for tag, count in postags.items():
        prior[tag] = count

    # make pos tag dict
    postags = sorted(postags.items(), key=lambda x: x[1], reverse=True)
    with tf.gfile.FastGFile(FLAGS.postags_output_file, "w") as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in postags]))
    print("Wrote postag file:", FLAGS.postags_output_file)
    postags = dict([(y, x) for x, y in enumerate(map(lambda x: x[0], postags))])
    postags = Vocabulary(postags, len(postags))

    sum_prior = float(sum(prior.values()))
    prior = dict([(postags.word_to_id(t), c / sum_prior) for t,c in prior.items()])
    print(prior)
    return postags, word2postag, prior

def main(unused_argv):
    train_captions = _load_and_process_captions(FLAGS.train_captions_file)
    vocab = load_vocab(FLAGS.word_counts_input_file)
    postags, word2postag, prior = _process_postags(train_captions, vocab)

    w2p_matrix = {}

    for w_id in sorted(vocab.values()):
        w2p_matrix[w_id] = [0] * len(postags)
        pt_distribution = word2postag.get(w_id, {})
        for tag, count in pt_distribution.items():
            t_id = postags.word_to_id(tag)
            w2p_matrix[w_id][t_id] = count
        s = float(sum(w2p_matrix[w_id]))
        if s > 0:
          for i in xrange(len(postags)):
              w2p_matrix[w_id][i] /= s
        else:
          for i in xrange(len(postags)):
              w2p_matrix[w_id][i] = prior[i]
          
    with tf.gfile.FastGFile(FLAGS.word2postag_output_file, "w") as f:
        f.write("\n".join([" ".join(map(str, d)) for w, d in sorted(w2p_matrix.items())]))
    print("Wrote word2postag file:", FLAGS.word2postag_output_file)

if __name__ == "__main__":
    tf.app.run()
