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
import numpy as np
import tensorflow as tf

# input data
tf.flags.DEFINE_string("image_dir", None, "Image directory.")
tf.flags.DEFINE_string("input_file", None, "File containing image/pos/neg")

# use existing word counts file
tf.flags.DEFINE_string("word_counts_input_file",
                       "data/word_counts.txt",
                       "use existing word_counts_file.")

# output files
tf.flags.DEFINE_string("output_dir", "data/Ranker_TFRecord_data", "Output directory for tfrecords.")
tf.flags.DEFINE_string("output_prefix", "ranker", "Output prefix for tfrecords.")

# words parameters
tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")

tf.flags.DEFINE_integer("maxlen", 30,
                       "Maximum length of captions. The captions will be padded to this length.")
tf.flags.DEFINE_integer("lines_per_image", 20,
                       "Number of captions per image.")

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


class ImageDecoder(object):
    """Helper class for decoding images in TensorFlow."""

    def __init__(self):
        # Create a single TensorFlow Session for all image decoding calls.
        self._sess = tf.Session()

        # TensorFlow ops for JPEG decoding.
        self._encoded_jpeg = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

    def decode_jpeg(self, encoded_jpeg):
        image = self._sess.run(self._decode_jpeg,
                               feed_dict={self._encoded_jpeg: encoded_jpeg})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def pad_or_truncate(input_caption_ids, maxlen):
    seqlens, caption_ids = [], []
    for c in input_caption_ids:
        if len(c) >= maxlen:
            caption_ids.extend(c[:maxlen])
            seqlens.append(maxlen)
        else:
            caption_ids.extend(c + [0] * (maxlen - len(c)))
            seqlens.append(len(c))
    return seqlens, caption_ids

def _to_sequence_example(image_id, image_filename, image_pos_captions, image_neg_captions, decoder, vocab):
    """Builds a SequenceExample proto for an image-caption pair.
    Args:
      image: An ImageMetadata object.
      decoder: An ImageDecoder object.
      vocab: A Vocabulary object.
    Returns:
      A SequenceExample proto.
    """
    with tf.gfile.FastGFile(image_filename, "r") as f:
        encoded_image = f.read()
    try:
        decoder.decode_jpeg(encoded_image)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % image.filename)
        return None

    assert len(image_pos_captions) == FLAGS.lines_per_image
    assert len(image_neg_captions) == FLAGS.lines_per_image

    pos_caption_ids = [[vocab.word_to_id(word) for word in caption] for caption in image_pos_captions]
    neg_caption_ids = [[vocab.word_to_id(word) for word in caption] for caption in image_neg_captions]

    pos_seqlens, pos_caption_ids = pad_or_truncate(pos_caption_ids, FLAGS.maxlen)
    neg_seqlens, neg_caption_ids = pad_or_truncate(neg_caption_ids, FLAGS.maxlen)

    features = tf.train.Features(feature={
        "image/id": _bytes_feature(image_id),
        "image/data": _bytes_feature(encoded_image),
        "image/pos_seqlens": tf.train.Feature(int64_list=tf.train.Int64List(value=pos_seqlens)),
        "image/neg_seqlens": tf.train.Feature(int64_list=tf.train.Int64List(value=neg_seqlens)),
        "image/pos_captions": tf.train.Feature(int64_list=tf.train.Int64List(value=pos_caption_ids)),
        "image/neg_captions": tf.train.Feature(int64_list=tf.train.Int64List(value=neg_caption_ids)),
    })

    example = tf.train.Example(features=features)
    return example

def _write_to_file(filename, strings):
    print("%s : writing %d items into %s." % (datetime.now(), len(strings), filename))
    writer = tf.python_io.TFRecordWriter(filename)
    for s in strings:
        writer.write(s)
    writer.close()

def _process_dataset(image_metadata, vocab):
    # Shuffle the ordering of images. Make the randomization repeatable.
    decoder = ImageDecoder()

    random.seed(12345)
    random.shuffle(image_metadata)

    images_per_file = 500
    strings = []
    counter = 0
    for image_id, filename, pos_captions, neg_captions in image_metadata:
        sequence_example = _to_sequence_example(image_id, filename, pos_captions, neg_captions, decoder, vocab)
        if sequence_example is not None:
            strings.append(sequence_example.SerializeToString())

        if len(strings) >= images_per_file:
            _write_to_file(os.path.join(FLAGS.output_dir, "%s-%d.tfrecord" % (FLAGS.output_prefix, counter)), strings)
            strings = []
            counter += 1

    if len(strings) > 0:
        _write_to_file(os.path.join(FLAGS.output_dir, "%s-%d.tfrecord" % (FLAGS.output_prefix, counter)), strings)

def _process_caption_jieba(caption):
    return jieba.cut(caption, cut_all=False)

def _load_and_process_metadata(input_file, image_dir):
    image_ids = set([])
    image_filenames = {}
    image_pos_captions = {}
    image_neg_captions = {}

    with open(input_file, 'r') as F:
        for line in F:
            image_id, pos, neg = line.strip().split("\t")
            pos = pos.decode("utf8")
            neg = neg.decode("utf8")
            if image_id not in image_ids:
                image_ids.add(image_id)
                image_filenames[image_id] = os.path.join(image_dir, image_id + ".jpg")
                image_pos_captions[image_id] = []
                image_neg_captions[image_id] = []
            image_pos_captions[image_id].append(_process_caption_jieba(pos))
            image_neg_captions[image_id].append(_process_caption_jieba(neg))
                
    image_metadata = []
    for image_id in image_ids:
        filename = image_filenames[image_id]
        pos_captions = image_pos_captions[image_id]
        neg_captions = image_neg_captions[image_id]
        image_metadata.append((image_id, filename, pos_captions, neg_captions))

    print("Finished processing captions for %d images in %s" %
          (len(image_ids), input_file))
    return image_metadata


def main(unused_argv):

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    # Load image metadata from caption files.
    dataset = _load_and_process_metadata(FLAGS.input_file, FLAGS.image_dir)
    # Create vocabulary from the training captions.
    vocab = load_vocab(FLAGS.word_counts_input_file)
    # process dataset
    _process_dataset(dataset, vocab)

if __name__ == "__main__":
    tf.app.run()
