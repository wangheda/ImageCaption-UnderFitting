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
# python2.7
# __author__ = 'WANG, Heda'

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
import base64

reload(sys)
sys.setdefaultencoding('utf8')
import threading
import jieba
import numpy as np
import tensorflow as tf

# input data
tf.flags.DEFINE_string("train_image_dir", "data/ai_challenger_caption_train_20170902/caption_train_images_20170902",
                       "Training image directory.")
tf.flags.DEFINE_string("train_captions_file", "data/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json",
                       "Training captions JSON file.")
tf.flags.DEFINE_string("train_localizations_file", "data/bottom_up_attention/aichallenger_train.tsv.small",
                       "Training captions TSV file.")

tf.flags.DEFINE_string("validate_image_dir", "data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910",
                       "Validation image directory.")
tf.flags.DEFINE_string("validate_localizations_file", "data/bottom_up_attention/aichallenger_validate.tsv.small",
                       "Validating captions TSV file.")

tf.flags.DEFINE_string("test1_image_dir", "data/ai_challenger_caption_test1_20170923/caption_test1_images_20170923",
                       "Test image directory.")
tf.flags.DEFINE_string("test1_localizations_file", "data/bottom_up_attention/aichallenger_test1.tsv.small",
                       "Test captions TSV file.")

tf.flags.DEFINE_string("test2_image_dir", "data/ai_challenger_caption_test_b_20171120/caption_test_b_images_20171120",
                       "Test image directory.")
tf.flags.DEFINE_string("test2_localizations_file", "data/bottom_up_attention/aichallenger_test2.tsv.small",
                       "Test captions TSV file.")


# use existing word counts file
tf.flags.DEFINE_string("word_counts_input_file",
                       "",
                       "If defined, use existing word_counts_file.")

# output files
tf.flags.DEFINE_string("output_dir", "data/Loc_TFRecord_data", "Output directory for tfrecords.")
tf.flags.DEFINE_string("word_counts_output_file",
                       "data/word_counts.txt",
                       "Output vocabulary file of word counts.")

# words parameters
tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")

# the minimum word count
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")

# threads
tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

# sharding parameters
tf.flags.DEFINE_integer("train_shards", 280,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("validate_shards", 8,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test1_shards", 8,
                        "Number of shards in testing TFRecord files.")
tf.flags.DEFINE_integer("test2_shards", 8,
                        "Number of shards in testing TFRecord files.")

tf.flags.DEFINE_boolean("build_flip_caption", False,
                        "Whether to generate flip caption. If True, only build train set,"
                        "If set False, build train and dev set")

tf.flags.DEFINE_integer("max_ref_length", 30,
                        "Maximum caption length.")
tf.flags.DEFINE_integer("num_refs", 5,
                        "Number of references per image.")

tf.flags.DEFINE_string("task", "train",
                       "Options are train/validate/test1/test2.")

FLAGS = tf.flags.FLAGS


ImageMetadata = namedtuple("ImageMetadata",
                           ["id", "filename", "base_filename", "localization", "captions", "flip_captions"])

# functions to flip caption
def find_all(string, query):
    # return all positions
    query_len = len(query)
    positions = []
    beg = 0
    pos = string.find(query, beg)
    while pos != -1:
        positions.append(pos)
        beg = pos + query_len
        pos = string.find(query, beg)
    return positions

def func_flip_caption(caption):
    lr_pos = find_all(caption, u"左右")
    noflip_pos = []
    for pos in lr_pos:
        noflip_pos.append(pos)
        noflip_pos.append(pos + 1)
    l_pos = find_all(caption, u"左")
    l_pos = [pos for pos in l_pos if pos not in noflip_pos]

    r_pos = find_all(caption, u"右")
    r_pos = [pos for pos in r_pos if pos not in noflip_pos]

    if not l_pos and not r_pos:
        return caption

    new_caption = ""
    for i,c in enumerate(caption):
        if i in l_pos:
            new_caption += u"右"
        elif i in r_pos:
            new_caption += u"左"
        else:
            new_caption += c
    return new_caption


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

def _int64_list(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_list(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_list(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])

def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])

def pad_or_truncate(captions, lengths):
    max_length = FLAGS.max_ref_length
    num_refs = FLAGS.num_refs 
    lengths = [min(l, max_length) for l in lengths]
    captions = [c[:l] + [0] * (max_length - l) for c, l in zip(captions, lengths)]
    if len(captions) < num_refs:
        captions = captions + [[0] * max_length for i in xrange(num_refs - len(captions))]
        lengths = lengths + [0] * (num_refs - len(captions))
    flat_captions = []
    for c in captions:
        flat_captions.extend(c)
    assert len(flat_captions) == num_refs * max_length
    assert len(lengths) == num_refs
    return flat_captions, lengths

def _to_sequence_example(image, decoder, vocab):
    """Builds a SequenceExample proto for an image-caption pair.
    Args:
      image: An ImageMetadata object.
      decoder: An ImageDecoder object.
      vocab: A Vocabulary object.
    Returns:
      A SequenceExample proto.
    """
    with tf.gfile.FastGFile(image.filename, "r") as f:
        encoded_image = f.read()

    try:
        decoder.decode_jpeg(encoded_image)
    except (tf.errors.InvalidArgumentError, AssertionError):
        print("Skipping file with invalid JPEG data: %s" % image.filename)
        return

    base_filename = image.base_filename
    localization = image.localization
    feature_list = {
        "image/id": _int64_feature(image.id),
        "image/filename": _bytes_feature(base_filename),
        "image/localization": _float_list(localization),
        "image/data": _bytes_feature(encoded_image),
    }

    if image.captions is not None:
        caption_ids = [[vocab.word_to_id(word) for word in caption] for caption in image.captions]
        caption_lengths = [len(caption) for caption in caption_ids]
        flip_caption_ids = [[vocab.word_to_id(word) for word in caption] for caption in image.flip_captions]
        flip_caption_lengths = [len(caption) for caption in flip_caption_ids]

        caption_ids, caption_lengths = pad_or_truncate(caption_ids, caption_lengths)
        flip_caption_ids, flip_caption_lengths = pad_or_truncate(flip_caption_ids, flip_caption_lengths)

        feature_list.update({
            "image/ref_words": _int64_list(caption_ids),
            "image/ref_lengths": _int64_list(caption_lengths),
            "image/flipped_ref_words": _int64_list(flip_caption_ids),
            "image/flipped_ref_lengths": _int64_list(flip_caption_lengths),
        })
    
    features = tf.train.Features(feature=feature_list)
    example = tf.train.Example(features=features)

    return example


def _process_image_files(thread_index, ranges, name, images, decoder, vocab,
                         num_shards):
    """Processes and saves a subset of images as TFRecord files in one thread.
    Args:
      thread_index: Integer thread identifier within [0, len(ranges)].
      ranges: A list of pairs of integers specifying the ranges of the dataset to
        process in parallel.
      name: Unique identifier specifying the dataset.
      images: List of ImageMetadata.
      decoder: An ImageDecoder object.
      vocab: A Vocabulary object.
      num_shards: Integer number of shards for the output files.
    """
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d.tfrecord" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
            image = images[i]

            sequence_example = _to_sequence_example(image, decoder, vocab)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

            if not counter % 1000:
                print("%s [thread %d]: Processed %d of %d items in thread batch." %
                      (datetime.now(), thread_index, counter, num_images_in_thread))
                sys.stdout.flush()

        writer.close()
        print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def _process_dataset(name, images, vocab, num_shards):
    """Processes a complete data set and saves it as a TFRecord.
    Args:
      name: Unique identifier specifying the dataset.
      images: List of ImageMetadata.
      vocab: A Vocabulary object.
      num_shards: Integer number of shards for the output files.
    """
    # Break up each image into a separate entity for each caption.
    images = [ImageMetadata(image.id, image.filename, image.base_filename, image.localization, image.captions, image.flip_captions)
              for image in images]

    # Shuffle the ordering of images. Make the randomization repeatable.
    random.seed(12345)
    random.shuffle(images)

    # Break the images into num_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]].
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Create a utility for decoding JPEG images to run sanity checks.
    decoder = ImageDecoder()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, images, decoder, vocab, num_shards)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
          (datetime.now(), len(images), name))


def _create_vocab(captions):
    """Creates the vocabulary of word to word_id.
    The vocabulary is saved to disk in a text file of word counts. The id of each
    word in the file is its corresponding 0-based line number.
    Args:
      captions: A list of lists of strings.
    Returns:
      A Vocabulary object.
    """
    print("Creating vocabulary.")
    counter = Counter()
    for c in captions:
        counter.update(c)
    print("Total words:", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    # Write out the word counts file.
    with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

    # Create the vocabulary dictionary.
    reverse_vocab = [x[0] for x in word_counts]
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

def _load_localization_file(localizations_file):
    loc_dict = {}
    with open(localizations_file) as F:
        for line in F:
            filename, width, height, num_boxes, box_str = line.strip().split()
            num_boxes = int(num_boxes)
            assert num_boxes == 36
            box_blob = base64.decodestring(box_str)
            box_array = np.frombuffer(box_blob, dtype=np.float32)
            assert len(box_array) == 4*num_boxes
            for i in xrange(0, len(box_array), 4):
                l1, u1, l2, u2 = box_array[i:i+4]
                assert l1 < l2
                assert u1 < u2
            loc_dict[filename] = box_array
        return loc_dict

def _load_and_process_metadata(captions_file, localizations_file, image_dir):
    """Loads image metadata from a JSON file and processes the captions.
    Args:
      captions_file: Json file containing caption annotations.
      image_dir: Directory containing the image files.
    Returns:
      A list of ImageMetadata.
    """
    loc_dict = _load_localization_file(localizations_file)
    image_id = set([])

    if captions_file is not None:
        id_to_captions = {}
        with open(captions_file, 'r') as f:
            caption_data = json.load(f)
        for data in caption_data:
            image_name = data['image_id'].split('.')[0]
            descriptions = data['caption']
            if image_name not in image_id:
                id_to_captions.setdefault(image_name, [])
                image_id.add(image_name)

            caption_num = len(descriptions)

            for i in range(caption_num):
                caption_temp = descriptions[i].strip().strip("。").replace('\n', '')
                if caption_temp != '':
                    id_to_captions[image_name].append(caption_temp)
        print("Loaded caption metadata for %d images from %s and image_id num is %s" %
              (len(id_to_captions), captions_file, len(image_id)))
    else:
        id_to_captions = None
        for filename in os.listdir(image_dir):
            if filename.endswith(".jpg"):
                image_name = filename.split(".")[0]
                if image_name not in image_id:
                    image_id.add(image_name)

    # Process the captions and combine the data into a list of ImageMetadata.
    print("Proccessing captions.")
    image_metadata = []
    num_captions = 0
    id = 0
    for base_filename in image_id:
        localization = loc_dict[base_filename]
        filename = os.path.join(image_dir, base_filename + '.jpg')
        if id_to_captions is not None:
          captions = [_process_caption_jieba(c) for c in id_to_captions[base_filename]]
          flip_captions = [_process_caption_jieba(func_flip_caption(c)) for c in id_to_captions[base_filename]]
          num_captions += len(captions)
        else:
          captions = None
          flip_captions = None
        image_metadata.append(ImageMetadata(id, filename, base_filename, localization, captions, flip_captions))
        id = id + 1
    print("Finished processing %d captions for %d images in %s" %
          (num_captions, len(image_id), captions_file))
    return image_metadata


def main(unused_argv):
    def _is_valid_num_shards(num_shards):
        """Returns True if num_shards is compatible with FLAGS.num_threads."""
        return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

    assert _is_valid_num_shards(FLAGS.train_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
    assert _is_valid_num_shards(FLAGS.validate_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.validate_shards")
    assert _is_valid_num_shards(FLAGS.test1_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.test1_shards")
    assert _is_valid_num_shards(FLAGS.test2_shards), (
        "Please make the FLAGS.num_threads commensurate with FLAGS.test2_shards")

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    if FLAGS.task == "train":
        # Load image metadata from caption files.
        train_dataset = _load_and_process_metadata(FLAGS.train_captions_file,
                                                   FLAGS.train_localizations_file,
                                                   FLAGS.train_image_dir)

        # Create vocabulary from the training captions.
        vocab = load_vocab(FLAGS.word_counts_input_file)
        _process_dataset("train", train_dataset, vocab, FLAGS.train_shards)

    elif FLAGS.task == "validate":
        # Load image metadata from caption files.
        validate_dataset = _load_and_process_metadata(None,
                                                   FLAGS.validate_localizations_file,
                                                   FLAGS.validate_image_dir)

        # Create vocabulary from the training captions.
        vocab = load_vocab(FLAGS.word_counts_input_file)
        _process_dataset("validate", validate_dataset, vocab, FLAGS.validate_shards)

    elif FLAGS.task == "test1":
        # Load image metadata from caption files.
        test1_dataset = _load_and_process_metadata(None,
                                                   FLAGS.test1_localizations_file,
                                                   FLAGS.test1_image_dir)

        # Create vocabulary from the training captions.
        vocab = load_vocab(FLAGS.word_counts_input_file)
        _process_dataset("test1", test1_dataset, vocab, FLAGS.test1_shards)

    elif FLAGS.task == "test2":
        # Load image metadata from caption files.
        test2_dataset = _load_and_process_metadata(None,
                                                   FLAGS.test2_localizations_file,
                                                   FLAGS.test2_image_dir)

        # Create vocabulary from the training captions.
        vocab = load_vocab(FLAGS.word_counts_input_file)
        _process_dataset("test2", test2_dataset, vocab, FLAGS.test2_shards)

if __name__ == "__main__":
    tf.app.run()
