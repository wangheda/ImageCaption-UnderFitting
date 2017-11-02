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

"""Model wrapper class for performing inference with a Im2TxtModel."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import im2txt_model
from inference_utils import inference_wrapper_base



class InferenceWrapper(inference_wrapper_base.InferenceWrapperBase):
  """Model wrapper class for performing inference with a Im2TxtModel."""

  def __init__(self):
    super(InferenceWrapper, self).__init__()

  def build_model(self):
    model = im2txt_model.Im2TxtModel(mode="inference")
    model.build()
    self.model = model
    if hasattr(model, "predicted_ids"):
      self.predicted_ids = model.predicted_ids
    if hasattr(model, "scores"):
      self.scores = model.scores
    if hasattr(model, "top_n_attributes"):
      self.top_n_attributes = model.top_n_attributes
    return model

  def feed_image(self, sess, encoded_image, use_attention=False):

    if use_attention:
        initial_state, initial_state_review = sess.run(fetches=["lstm/initial_state:0", "lstm_review/initial_state_review:0"],
                                 feed_dict={"image_feed:0": encoded_image})
        return initial_state, initial_state_review
    else:
        initial_state = sess.run(fetches=["lstm/initial_state:0"],
                                 feed_dict={"image_feed:0": encoded_image})
        return initial_state, None

  def inference_step(self, sess, input_feed, state_feed, encoded_image=None, use_attention=False, state_review_feed=None):
    # the image_feed need to be used if attention model is used
    if use_attention:
        softmax_output, state_output, state_review_output = sess.run(
            fetches=["softmax:0", "lstm/state:0","lstm_review/state_review:0"],
            feed_dict={
                "image_feed:0": encoded_image,
                "input_feed:0": input_feed,
                "lstm/state_feed:0": state_feed,
                "lstm_review/state_review_feed:0": state_review_feed,
            })
        return softmax_output, state_output, None, state_review_output
    else:
        softmax_output, state_output = sess.run(
            fetches=["softmax:0", "lstm/state:0"],
            feed_dict={
                "input_feed:0": input_feed,
                "lstm/state_feed:0": state_feed
            })
        return softmax_output, state_output, None, None

  def support_ingraph(self):
    return self.model.support_ingraph
