# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Class for generating captions from an image-to-text model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import heapq
import math

import tensorflow as tf
FLAGS = tf.flags.FLAGS

import numpy as np


class Caption(object):
  """Represents a complete or partial caption."""

  def __init__(self, sentence, state, logprob, score, metadata=None, review_state=None):
    """Initializes the Caption.

    Args:
      sentence: List of word ids in the caption.
      state: Model state after generating the previous word.
      logprob: Log-probability of the caption.
      score: Score of the caption.
      metadata: Optional metadata associated with the partial sentence. If not
        None, a list of strings with the same length as 'sentence'.
    """
    self.sentence = sentence
    self.state = state
    self.review_state = review_state
    self.logprob = logprob
    self.score = score
    self.metadata = metadata

  def __cmp__(self, other):
    """Compares Captions by score."""
    assert isinstance(other, Caption)
    if self.score == other.score:
      return 0
    elif self.score < other.score:
      return -1
    else:
      return 1
  
  # For Python 3 compatibility (__cmp__ is deprecated).
  def __lt__(self, other):
    assert isinstance(other, Caption)
    return self.score < other.score
  
  # Also for Python 3 compatibility.
  def __eq__(self, other):
    assert isinstance(other, Caption)
    return self.score == other.score


class TopN(object):
  """Maintains the top n elements of an incrementally provided set."""

  def __init__(self, n):
    self._n = n
    self._data = []

  def size(self):
    assert self._data is not None
    return len(self._data)

  def push(self, x):
    """Pushes a new element."""
    assert self._data is not None
    if len(self._data) < self._n:
      heapq.heappush(self._data, x)
    else:
      heapq.heappushpop(self._data, x)

  def extract(self, sort=False):
    """Extracts all elements from the TopN. This is a destructive operation.

    The only method that can be called immediately after extract() is reset().

    Args:
      sort: Whether to return the elements in descending sorted order.

    Returns:
      A list of data; the top n elements provided to the set.
    """
    assert self._data is not None
    data = self._data
    self._data = None
    if sort:
      data.sort(reverse=True)
    return data

  def reset(self):
    """Returns the TopN to an empty state."""
    self._data = []


class CaptionGenerator(object):
  """Class to generate captions from an image-to-text model."""

  def __init__(self,
               model,
               vocab,
               beam_size=3,
               max_caption_length=20,
               length_normalization_factor=0.0):
    """Initializes the generator.

    Args:
      model: Object encapsulating a trained image-to-text model. Must have
        methods feed_image() and inference_step(). For example, an instance of
        InferenceWrapperBase.
      vocab: A Vocabulary object.
      beam_size: Beam size to use when generating captions.
      max_caption_length: The maximum caption length before stopping the search.
      length_normalization_factor: If != 0, a number x such that captions are
        scored by logprob/length^x, rather than logprob. This changes the
        relative scores of captions depending on their lengths. For example, if
        x > 0 then longer captions will be favored.
    """
    self.vocab = vocab
    self.model = model

    self.beam_size = beam_size
    self.max_caption_length = max_caption_length
    self.length_normalization_factor = length_normalization_factor

  def beam_search(self, sess, encoded_image):
    """Runs beam search caption generation on a single image.

    Args:
      sess: TensorFlow Session object.
      encoded_image: An encoded image string.

    Returns:
      A list of Caption sorted by descending score.
    """
    if self.model.support_ingraph():
      predicted_ids, scores = sess.run(
        [self.model.predicted_ids, self.model.scores], 
        feed_dict={"image_feed:0": encoded_image})
      predicted_ids = np.transpose(predicted_ids, (0,2,1))   
      scores = np.transpose(scores, (0,2,1))

      final_captions = []
      for caption in predicted_ids[0]:
        partial_caption = []
        for word_id in caption:
          if word_id != FLAGS.end_token:
            if word_id >= 0 and word_id != FLAGS.start_token:
              partial_caption.append(word_id)
          else:
            break
        final_captions.append(partial_caption)

    else:
      # The original out-graph inference
      # Feed in the image to get the initial state.
      initial_state, initial_review_state = self.model.feed_image(sess, encoded_image, use_attention=True)

      initial_beam = Caption(
          sentence=[self.vocab.start_id],
          state=initial_state[0],
          logprob=0.0,
          score=0.0,
          metadata=[""],
          review_state=initial_review_state[0])
      partial_captions = TopN(self.beam_size)
      partial_captions.push(initial_beam)
      complete_captions = TopN(self.beam_size)

      # Run beam search.
      for _ in range(self.max_caption_length - 1):
        partial_captions_list = partial_captions.extract()
        partial_captions.reset()
        input_feed = np.array([c.sentence[-1] for c in partial_captions_list])
        state_feed = np.array([c.state for c in partial_captions_list])
        review_state_feed = np.array([c.review_state for c in partial_captions_list])

        softmax, new_states, metadata, new_review_states = self.model.inference_step(sess,
                                                                  input_feed,
                                                                  state_feed,
                                                                  encoded_image=encoded_image,
                                                                  use_attention=True,
                                                                  state_review_feed=review_state_feed)

        for i, partial_caption in enumerate(partial_captions_list):
          word_probabilities = softmax[i]
          state = new_states[i]
          review_state = new_review_states[i]
          # For this partial caption, get the beam_size most probable next words.
          words_and_probs = list(enumerate(word_probabilities))
          words_and_probs.sort(key=lambda x: -x[1])
          words_and_probs = words_and_probs[0:self.beam_size]
          # Each next word gives a new partial caption.
          for w, p in words_and_probs:
            if p < 1e-12:
              continue  # Avoid log(0).
            sentence = partial_caption.sentence + [w]
            logprob = partial_caption.logprob + math.log(p)
            score = logprob
            if metadata:
              metadata_list = partial_caption.metadata + [metadata[i]]
            else:
              metadata_list = None
            if w == self.vocab.end_id:
              if self.length_normalization_factor > 0:
                score /= len(sentence)**self.length_normalization_factor
              beam = Caption(sentence, state, logprob, score, metadata_list, review_state)
              complete_captions.push(beam)
            else:
              beam = Caption(sentence, state, logprob, score, metadata_list, review_state)
              partial_captions.push(beam)
        if partial_captions.size() == 0:
          # We have run out of partial candidates; happens when beam_size = 1.
          break

      # If we have no complete captions then fall back to the partial captions.
      # But never output a mixture of complete and partial captions because a
      # partial caption could have a higher score than all the complete captions.
      if not complete_captions.size():
        complete_captions = partial_captions

      final_captions = [item.sentence[1:-1] for item in complete_captions.extract(sort=True)]

    return final_captions

  def predict_attributes(self, sess, encoded_image):
    """Predict attributes on a single image.

    Args:
      sess: TensorFlow Session object.
      encoded_image: An encoded image string.

    Returns:
      A list of attributes and probabilities sorted by descending score,.
    """
    top_n_attributes = sess.run(self.model.top_n_attributes, feed_dict={"image_feed:0": encoded_image})
    attr_probs, attr_ids = top_n_attributes
    return attr_ids[0], attr_probs[0]
