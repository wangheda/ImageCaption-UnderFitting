"""Provides readers configured for different datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
from tensorflow import logging
from tensorflow import flags
from tensorflow import gfile

from image_models.image_processing import process_image

FLAGS = flags.FLAGS
tf.flags.DEFINE_integer("lines_per_image", 15,
                        "Number of caption lines for each of the images.")
tf.flags.DEFINE_integer("max_len", 30,
                        "The maximum length of caption for each of the images.")


class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()


class ImageCaptionReader(BaseReader):

  def __init__(self,
               mode="train",
               lines_per_image=20,
               max_len=30):
    self.lines_per_image = lines_per_image
    self.max_len = max_len
    self.mode = mode

  def prepare_reader(self, filename_queue, batch_size=16):
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read(filename_queue)

    num_words = self.lines_per_image * self.max_len
    if self.mode == "train":
      feature_map = {
        "image/id": tf.FixedLenFeature([], tf.string),
        "image/data": tf.FixedLenFeature([], tf.string),
        "image/seqlens": tf.FixedLenFeature([self.lines_per_image], tf.int64),
        "image/scores": tf.FixedLenFeature([self.lines_per_image], tf.float32),
        "image/captions": tf.FixedLenFeature([num_words], tf.int64),
      }
    else:
      feature_map = {
        "image/id": tf.FixedLenFeature([], tf.string),
        "image/data": tf.FixedLenFeature([], tf.string),
        "image/seqlens": tf.FixedLenFeature([self.lines_per_image], tf.int64),
        "image/captions": tf.FixedLenFeature([num_words], tf.int64),
      }


    features = tf.parse_single_example(serialized_examples, features=feature_map)
    print(" features", features)

    # [1]
    """
    image_ids = tf.stack([features["image/id"] 
                          for i in range(self.lines_per_image)])
    image_ids = tf.reshape(image_ids, shape=[self.lines_per_image])
    print(" image_id", image_ids)
    """
    image_id = features["image/id"]
    print(" image_id", image_id)
    

    # [height, width, channels]
    """
    encoded_image = features["image/data"]
    images = tf.stack([process_image(encoded_image, 
                          is_training=True, 
                          height=FLAGS.image_height, 
                          width=FLAGS.image_width)
                      for i in range(self.lines_per_image)])
    print(" image", images)
    """
    encoded_image = features["image/data"]
    image = process_image(encoded_image, 
                          is_training=True, 
                          height=FLAGS.image_height, 
                          width=FLAGS.image_width)
    print(" image", image)

    """
    pos_captions = tf.reshape(features["image/pos_captions"], 
                              shape=[self.lines_per_image, self.max_len])
    neg_captions = tf.reshape(features["image/neg_captions"], 
                              shape=[self.lines_per_image, self.max_len])
    pos_seqlens = tf.reshape(features["image/pos_seqlens"], 
                              shape=[self.lines_per_image])
    neg_seqlens = tf.reshape(features["image/neg_seqlens"], 
                              shape=[self.lines_per_image])
    print("pos", pos_captions, pos_seqlens)
    print("neg", neg_captions, neg_seqlens)
    """

    captions = tf.reshape(features["image/captions"], 
                          shape=[self.lines_per_image, self.max_len])
    seqlens = tf.reshape(features["image/seqlens"], 
                         shape=[self.lines_per_image])
    print(" captions", captions)
    print(" seqlens", seqlens)

    if self.mode == "train":
      scores = tf.reshape(features["image/scores"], 
                         shape=[self.lines_per_image])
    else:
      scores = None

    #return image_ids, images, pos_captions, pos_seqlens, neg_captions, neg_seqlens
    return image_id, image, captions, seqlens, scores


def get_input_data_tensors(data_pattern=None,
                           batch_size=16,
                           num_epochs=None,
                           is_training=True,
                           num_readers=1):

  if is_training:
    reader = ImageCaptionReader(lines_per_image=FLAGS.lines_per_image,
                                max_len=FLAGS.max_len,
                                mode="train")
  else:
    reader = ImageCaptionReader(lines_per_image=FLAGS.lines_per_image,
                                max_len=FLAGS.max_len,
                                mode="inference")
  logging.info("Using batch size of " + str(batch_size) + " for training.")
  with tf.name_scope("train_input"):
    files = gfile.Glob(data_pattern)
    print("number of training files:", len(files))
    if not files:
      raise IOError("Unable to find training files. data_pattern='" +
                    data_pattern + "'.")
    logging.info("Number of training files: %s.", str(len(files)))
    if is_training:
      filename_queue = tf.train.string_input_producer(
          files, num_epochs=num_epochs, shuffle=True)
    else:
      filename_queue = tf.train.string_input_producer(
          files, num_epochs=num_epochs, shuffle=False)

    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    if is_training:
      return tf.train.shuffle_batch_join(
          training_data,
          batch_size=batch_size,
          capacity=batch_size * 8,
          min_after_dequeue=batch_size,
          allow_smaller_final_batch=False,
          enqueue_many=True)
    else:
      return tf.train.batch_join(
          training_data,
          batch_size=batch_size,
          capacity=batch_size * 8,
          allow_smaller_final_batch=False,
          enqueue_many=True)

