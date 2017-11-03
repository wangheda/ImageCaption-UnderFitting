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
"""Build word idf score."""
# __author__ = 'Miao'
# python2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os.path
import sys
import math
import codecs

reload(sys)
sys.setdefaultencoding('utf8')
import jieba
import tensorflow as tf

# input data
tf.flags.DEFINE_string("train_captions_file", "data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json",
                       "Training captions JSON file.")

# use existing word counts file
tf.flags.DEFINE_string("word_counts_input_file",
                       "data/word_counts.txt",
                       "If defined, use existing word_counts_file.")

# output files
tf.flags.DEFINE_string("word_idf_output_file", "data/word_idf.txt",
                       "Output vocabulary file of char counts.")

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
    return reverse_vocab


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

def calculate_idf(captions, reverse_vocab):
    word_df_dict = dict(zip(reverse_vocab, [0]*len(reverse_vocab)))
    caption_num = len(captions)
    for caption in captions:
        for word in set(_process_caption_jieba(caption)):
            if word in word_df_dict:
                word_df_dict[word] += 1
    word_idf_dict = {}
    for word, df in word_df_dict.items():
        word_idf_dict[word] = math.log(caption_num / (1.0 + df))
    return word_idf_dict

def main(unused_argv):
    train_captions = _load_and_process_captions(FLAGS.train_captions_file)
    reverse_vocab = load_vocab(FLAGS.word_counts_input_file)
    word_idf_dict = calculate_idf(train_captions, reverse_vocab)

    output = codecs.open(FLAGS.word_idf_output_file, "w", "utf-8")
    for word in reverse_vocab:
        output.write("%s %f\n" %(word, word_idf_dict[word]))
    output.close()
    print("Wrote word_idf file:", FLAGS.word_idf_output_file)

if __name__ == "__main__":
    tf.app.run()
