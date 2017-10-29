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
tf.flags.DEFINE_string("chars_output_file", "resources/chars.txt",
                       "Output vocabulary file of char counts.")
tf.flags.DEFINE_string("word2char_output_file", "resources/word2char.txt",
                       "Output mapping file of word to char.")

# the minimum word count
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")

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

    def items(self):
        return self._vocab.items()


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

def _process_chars(captions, vocab):
    chars = {}

    num_processed = 0
    chars[FLAGS.start_word] = FLAGS.min_word_count
    chars[FLAGS.end_word] = FLAGS.min_word_count
    chars[FLAGS.unknown_word] = FLAGS.min_word_count
    for caption in captions:

        if num_processed % 1000 == 0:
            print("%d / %d" % (num_processed, len(captions)))
        num_processed += 1

        for ch in caption:
            if ch in chars:
                chars[ch] += 1
            else:
                chars[ch] = 1

    # make char dict
    def mycmp(a, b):
        if a[1] < b[1]:
            return -1
        elif a[1] > b[1]:
            return 1
        else:
            return cmp(a[0], b[0])
    chars = [(w,c) for w, c in chars.items() if c >= FLAGS.min_word_count]
    chars.sort(cmp=mycmp, reverse=True)

    with tf.gfile.FastGFile(FLAGS.chars_output_file, "w") as f:
        f.write("\n".join(["%s %d" % (w.encode("utf8"), c) for w,c in chars]))
    print("Wrote char file:", FLAGS.chars_output_file)
    chars = dict([(y, x) for x, y in enumerate(map(lambda x: x[0], chars))])
    chars = Vocabulary(chars, len(chars))
    return chars

def main(unused_argv):
    train_captions = _load_and_process_captions(FLAGS.train_captions_file)
    vocab = load_vocab(FLAGS.word_counts_input_file)
    chars = _process_chars(train_captions, vocab)

    w2p_matrix = {}

    special_words = set([FLAGS.start_word, FLAGS.end_word, FLAGS.unknown_word])

    for word, w_id in sorted(vocab.items()):
        w2p_matrix[w_id] = [0] * len(chars)
        if word not in special_words:
            for ch in word:
                c_id = chars.word_to_id(ch)
                w2p_matrix[w_id][c_id] = 1
        else:
            c_id = chars.word_to_id(word)
            w2p_matrix[w_id][c_id] = 1

        s = float(sum(w2p_matrix[w_id]))
        if s > 0:
          for i in xrange(len(chars)):
              w2p_matrix[w_id][i] /= s
        else:
          raise Exception("There should be at least on character in word %s" % word)
          
    with tf.gfile.FastGFile(FLAGS.word2char_output_file, "w") as f:
        f.write("\n".join([" ".join(map(str, d)) for w, d in sorted(w2p_matrix.items())]))
    print("Wrote word2char file:", FLAGS.word2char_output_file)

if __name__ == "__main__":
    tf.app.run()
