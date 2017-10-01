# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides readers configured for different datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import utils
import tensorflow as tf
from tensorflow import logging
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_bool("use_image_distortion", True,
    "Whether to augmenting images before apply them.")
flags.DEFINE_bool("image_distortion_flip_horizontal", False,
    "Whether to flip images horizontally before apply them.")

class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()


def distort_image(image, thread_id):
  """Perform random distortions on an image.

  Args:
    image: A float32 Tensor of shape [height, width, 3] with values in [0, 1).
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.

  Returns:
    distorted_image: A float32 Tensor of shape [height, width, 3] with values in
      [0, 1].
  """
  # Randomly flip horizontally.
  with tf.name_scope("flip_horizontal", values=[image]):
    image = tf.image.random_flip_left_right(image)

  # Randomly distort the colors based on thread id.
  color_ordering = thread_id % 2
  with tf.name_scope("distort_color", values=[image]):
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.032)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
      image = tf.image.random_brightness(image, max_delta=32. / 255.)
      image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      image = tf.image.random_hue(image, max_delta=0.032)

    # The random_* ops do not necessarily clamp.
    image = tf.clip_by_value(image, 0.0, 1.0)

  return image


def process_image(encoded_image,
                  is_training,
                  height,
                  width,
                  resize_height=346,
                  resize_width=346,
                  thread_id=0,
                  image_format="jpeg"):
  """Decode an image, resize and apply random distortions.

  In training, images are distorted slightly differently depending on thread_id.

  Args:
    encoded_image: String Tensor containing the image.
    is_training: Boolean; whether preprocessing for training or eval.
    height: Height of the output image.
    width: Width of the output image.
    resize_height: If > 0, resize height before crop to final dimensions.
    resize_width: If > 0, resize width before crop to final dimensions.
    thread_id: Preprocessing thread id used to select the ordering of color
      distortions. There should be a multiple of 2 preprocessing threads.
    image_format: "jpeg" or "png".

  Returns:
    A float32 Tensor of shape [height, width, 3] with values in [-1, 1].

  Raises:
    ValueError: If image_format is invalid.
  """
  # Helper function to log an image summary to the visualizer. Summaries are
  # only logged in thread 0.
  def image_summary(name, image):
    if not thread_id:
      tf.summary.image(name, tf.expand_dims(image, 0))

  # Decode image into a float32 Tensor of shape [?, ?, 3] with values in [0, 1).
  with tf.name_scope("decode", values=[encoded_image]):
    if image_format == "jpeg":
      image = tf.image.decode_jpeg(encoded_image, channels=3)
    elif image_format == "png":
      image = tf.image.decode_png(encoded_image, channels=3)
    else:
      raise ValueError("Invalid image format: %s" % image_format)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  image_summary("original_image", image)

  # Resize image.
  assert (resize_height > 0) == (resize_width > 0)
  if resize_height:
    image = tf.image.resize_images(image,
                                   size=[resize_height, resize_width],
                                   method=tf.image.ResizeMethod.BILINEAR)

  # Crop to final dimensions.
  if is_training:
    image = tf.random_crop(image, [height, width, 3])
  else:
    # Central crop, assuming resize_height > height, resize_width > width.
    image = tf.image.resize_image_with_crop_or_pad(image, height, width)

  image_summary("resized_image", image)

  # Randomly distort the image.
  if is_training:
    image = distort_image(image, thread_id)

  image_summary("final_image", image)

  # Rescale to [-1,1] instead of [0, 1]
  image = tf.subtract(image, 0.5)
  image = tf.multiply(image, 2.0)
  return image
  
class ImageCaptionReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """

  def __init__(self,
               width=1918,
               height=1280,
               channels=3):
    """Construct a CarvanaFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
    """
    self.width = width
    self.height = height
    self.channels = channels

  def prepare_reader(self, filename_queue, batch_size=16):
    """Creates a single reader thread for .

    Args:
      filename_queue: A tensorflow queue of filename locations.

    Returns:
      A tuple of video indexes, features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read(filename_queue)

    feature_map = {"id": tf.FixedLenFeature([], tf.string),
                   "image": tf.FixedLenFeature([], tf.string),
                   "mask": tf.FixedLenFeature([], tf.string)}

    features = tf.parse_single_example(serialized_examples, features=feature_map)
    print >> sys.stderr, " features", features

    image_id = features["id"]
    image_data = features["image"]
    image_mask = features["mask"]
    print >> sys.stderr, " image_id", image_id
    print >> sys.stderr, " image_data", image_data
    print >> sys.stderr, " image_mask", image_mask

    # reshape to rank1
    image_id = tf.reshape(image_id, shape=[1])

    # [height, width, channels]
    image_data = tf.image.decode_jpeg(image_data, channels=3)
    # image_data.set_shape(self.height * self.width * self.channels)
    image_data = tf.reshape(image_data, shape=[self.height, self.width, self.channels])
    print >> sys.stderr, " image_data", image_data

    # [height, width]
    image_mask = tf.decode_raw(image_mask, tf.uint8)
    image_mask.set_shape(self.height * self.width)
    image_mask = tf.reshape(image_mask, shape=[self.height, self.width])
    image_mask = tf.greater(image_mask, 0)
    print >> sys.stderr, " image_mask", image_mask

    # image augmentation
    if hasattr(FLAGS, "use_data_augmentation") and FLAGS.use_data_augmentation:
      image_data, image_mask = image_augmentation(image_data, image_mask)

    image_data = tf.reshape(image_data, shape=[1, self.height, self.width, self.channels])
    image_mask = tf.reshape(image_mask, shape=[1, self.height, self.width])
    return image_id, image_data, image_mask


class CarvanaPredictionFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """

  def __init__(self,
               width=1918,
               height=1280,
               channels=3):
    """Construct a CarvanaFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
    """
    self.width = width
    self.height = height
    self.channels = channels

  def prepare_reader(self, filename_queue, batch_size=16):
    """Creates a single reader thread for .

    Args:
      filename_queue: A tensorflow queue of filename locations.

    Returns:
      A tuple of video indexes, features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read(filename_queue)

    feature_map = {"id": tf.FixedLenFeature([], tf.string),
                   "image": tf.FixedLenFeature([], tf.string),
                   "mask": tf.FixedLenFeature([], tf.string)}

    features = tf.parse_single_example(serialized_examples, features=feature_map)
    print >> sys.stderr, " features", features

    image_id = features["id"]
    image_data = features["image"]
    image_mask = features["mask"]
    print >> sys.stderr, " image_id", image_id
    print >> sys.stderr, " image_data", image_data
    print >> sys.stderr, " image_mask", image_mask

    # reshape to rank1
    image_id = tf.reshape(image_id, shape=[1])

    # [height, width, channels]
    image_data = tf.decode_raw(image_data, tf.uint8)
    image_data.set_shape(self.height * self.width * self.channels)
    image_data = tf.reshape(image_data, shape=[self.height, self.width, self.channels])
    print >> sys.stderr, " image_data", image_data

    # [height, width]
    image_mask = tf.decode_raw(image_mask, tf.uint8)
    image_mask.set_shape(self.height * self.width)
    image_mask = tf.reshape(image_mask, shape=[self.height, self.width])
    image_mask = tf.greater(image_mask, 0)
    print >> sys.stderr, " image_mask", image_mask

    # image augmentation
    if hasattr(FLAGS, "use_data_augmentation") and FLAGS.use_data_augmentation:
      image_data, image_mask = image_augmentation(image_data, image_mask)

    image_data = tf.reshape(image_data, shape=[1, self.height, self.width, self.channels])
    image_mask = tf.reshape(image_mask, shape=[1, self.height, self.width])
    return image_id, image_data, image_mask


class CarvanaTestFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """

  def __init__(self,
               width=1918,
               height=1280,
               channels=3):
    """Construct a CarvanaFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
    """
    self.width = width
    self.height = height
    self.channels = channels

  def prepare_reader(self, filename_queue, batch_size=16):
    """Creates a single reader thread for .

    Args:
      filename_queue: A tensorflow queue of filename locations.

    Returns:
      A tuple of video indexes, features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_examples = reader.read(filename_queue)

    feature_map = {"id": tf.FixedLenFeature([], tf.string),
                   "image": tf.FixedLenFeature([], tf.string)}

    features = tf.parse_single_example(serialized_examples, features=feature_map)
    print >> sys.stderr, " features", features

    image_id = features["id"]
    image_data = features["image"]
    print >> sys.stderr, " image_id", image_id
    print >> sys.stderr, " image_data", image_data

    # reshape to rank1
    image_id = tf.reshape(image_id, shape=[1])

    # [height, width, channels]
    image_data = tf.image.decode_jpeg(image_data, channels=3)
    # image_data.set_shape(self.height * self.width * self.channels)
    image_data = tf.reshape(image_data, shape=[1, self.height, self.width, self.channels])
    print >> sys.stderr, " image_data", image_data

    return image_id, image_data

