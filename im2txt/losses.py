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

"""Provides definitions for non-regularized training or test losses."""

import sys
import numpy as np
import tensorflow as tf
from tensorflow import flags
import tf_cider

FLAGS = flags.FLAGS

LOG_TENSOR = True

def log_tensor(name, g=None, l=None):
  if LOG_TENSOR:
    if g is None and l is None:
      print >> sys.stderr, name, eval(name, {"self":self})
    else:
      print >> sys.stderr, name, eval(name, g, l)

class BaseLoss(object):
  """Inherit from this class when implementing new losses."""

  def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
    """Calculates the average loss of the examples in a mini-batch.

     Args:
      unused_predictions: a 2-d tensor storing the prediction scores, in which
        each row represents a sample in the mini-batch and each column
        represents a class.
      unused_labels: a 2-d tensor storing the labels, which has the same shape
        as the unused_predictions. The labels must be in the range of 0 and 1.
      unused_params: loss specific parameters.

    Returns:
      A scalar loss tensor.
    """
    raise NotImplementedError()


class CrossEntropyLoss(BaseLoss):
  def calculate_loss(self, predictions, labels, weights=None,
                     **unused_params):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,
                                                   logits=predictions)
    if weights is not None:
      loss = tf.div(tf.reduce_sum(loss * weights),
                    tf.reduce_sum(weights) + epsilon)
    else:
      loss = tf.reduce_mean(loss)
    return loss

class SparseSoftmaxCrossEntropyLoss(BaseLoss):
  def calculate_loss(self, predictions, labels, weights=None, 
                     epsilon=1e-9, **unused_params):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                          logits=predictions)
    if weights is not None:
      loss = tf.div(tf.reduce_sum(loss * weights),
                    tf.reduce_sum(weights) + epsilon)
    else:
      loss = tf.reduce_mean(loss)
    return loss

class SelfCriticalLoss(BaseLoss):
  def __init__(self):
    self.cider_scorer = tf_cider.CiderScorer()

  def calculate_loss(self,
                     target_caption_words, 
                     target_caption_lengths, 
                     greedy_caption_words, 
                     greedy_caption_lengths, 
                     sample_caption_words, 
                     sample_caption_lengths, 
                     sample_caption_logits, 
                     epsilon=1e-9, **unused_params):

    cider_scorer = self.cider_scorer

    log_tensor("greedy_caption_words", l=locals())
    log_tensor("greedy_caption_lengths", l=locals())
    log_tensor("sample_caption_logits", l=locals())
    log_tensor("sample_caption_words", l=locals())
    log_tensor("sample_caption_lengths", l=locals())
    log_tensor("target_caption_words", l=locals())
    log_tensor("target_caption_lengths", l=locals())

    greedy_score = cider_scorer.score(greedy_caption_words,
                                      greedy_caption_lengths,
                                      target_caption_words,
                                      target_caption_lengths)
    sample_score = cider_scorer.score(sample_caption_words,
                                      sample_caption_lengths,
                                      target_caption_words,
                                      target_caption_lengths)

    tf.summary.histogram("losses/greedy_score", greedy_score)
    tf.summary.histogram("losses/sample_score", sample_score)
    tf.summary.histogram("losses/greedy_caption_lengths", greedy_caption_lengths)
    tf.summary.histogram("losses/sample_caption_lengths", sample_caption_lengths)

    # reward = -1 * reward
    reward = greedy_score - sample_score
    reward = tf.stop_gradient(reward)

    # extract the logprobs of each word in sample_captions
    sample_probs = tf.nn.softmax(sample_caption_logits)

    # get sample_probs of every 
    batch_size, max_sample_length, _ = sample_probs.get_shape().as_list()
    sample_caption_mask = tf.sequence_mask(sample_caption_lengths, 
                                           maxlen=max_sample_length)
    sample_caption_mask = tf.cast(sample_caption_mask, dtype=tf.float32)
    sample_batch_index = tf.tile(tf.reshape(tf.range(0, batch_size), 
                                            shape=[batch_size,1]), 
                                 multiples=[1, max_sample_length])
    sample_seq_index = tf.tile(tf.reshape(tf.range(0, max_sample_length), 
                                          shape=[1, max_sample_length]), 
                               multiples=[batch_size, 1])
    sample_gather_index = tf.stack([sample_batch_index, 
                                    sample_seq_index, 
                                    sample_caption_words], axis=2)

    sample_caption_probs = tf.gather_nd(sample_probs, sample_gather_index)

    rl_loss = tf.expand_dims(reward, 1) * tf.log(sample_caption_probs)
    rl_loss = tf.div(tf.reduce_sum(rl_loss * sample_caption_mask),
                     tf.reduce_sum(sample_caption_mask),
                     name="rl_loss")
    tf.summary.scalar("losses/rl_loss", rl_loss)

    log_tensor("reward", l=locals())
    log_tensor("sample_probs", l=locals())
    log_tensor("sample_batch_index", l=locals())
    log_tensor("sample_seq_index", l=locals())
    log_tensor("sample_gather_index", l=locals())
    log_tensor("sample_caption_probs", l=locals())
    log_tensor("rl_loss", l=locals())
    return rl_loss


