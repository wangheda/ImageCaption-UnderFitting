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

"""Contains a collection of util functions for training and evaluating.
"""

import numpy
import tensorflow as tf
from tensorflow import logging

def MakeSummary(name, value):
  """Creates a tf.Summary proto with the given name and value."""
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  return summary

def AddGlobalStepSummary(summary_writer,
                         global_step_val,
                         global_step_info_dict,
                         summary_scope="Eval"):
  """Add the global_step summary to the Tensorboard.

  Args:
    summary_writer: Tensorflow summary_writer.
    global_step_val: a int value of the global step.
    global_step_info_dict: a dictionary of the evaluation metrics calculated for
      a mini-batch.
    summary_scope: Train or Eval.

  Returns:
    A string of this global_step summary
  """
  this_f1_score = global_step_info_dict["f1_score"]
  this_f2_score = global_step_info_dict["f2_score"]
  this_hit_at_one = global_step_info_dict["hit_at_one"]
  this_perr = global_step_info_dict["perr"]
  this_loss = global_step_info_dict["loss"]
  examples_per_second = global_step_info_dict.get("examples_per_second", -1)

  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Hit@1", this_hit_at_one),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Perr", this_perr),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_F1", this_f1_score),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_F2", this_f2_score),
      global_step_val)
  summary_writer.add_summary(
      MakeSummary("GlobalStep/" + summary_scope + "_Loss", this_loss),
      global_step_val)

  if examples_per_second != -1:
    summary_writer.add_summary(
        MakeSummary("GlobalStep/" + summary_scope + "_Example_Second",
                    examples_per_second), global_step_val)

  summary_writer.flush()
  info = ("global_step {0} | Batch Hit@1: {1:.3f} | Batch PERR: {2:.3f} | Batch F1: {2:.3f} | Batch F2: {2:.3f} | Batch Loss: {3:.3f} "
          "| Examples_per_sec: {4:.3f}").format(
              global_step_val, this_hit_at_one, this_perr, this_f1_score, this_f2_score, this_loss,
              examples_per_second)
  return info


def GetListOfFeatureNames(feature_names):
  """Extract the list of feature names
     from string of comma separated values.

  Args:
    feature_names: string containing comma separated list of feature names

  Returns:
    List of the feature names
    Elements in the list are strings.
  """
  list_of_feature_names = [
      feature_names.strip() for feature_names in feature_names.split(',')]
  return list_of_feature_names

def GetListOfFeatureSizes(feature_sizes):
  """Extract the list of the dimensionality of each feature
     from string of comma separated values.

  Args:
    feature_sizes: string containing comma separated list of feature sizes

  Returns:
    List of the dimensionality of each feature.
    Elements in the first list are integers.
  """
  list_of_feature_sizes = [
      int(feature_sizes) for feature_sizes in feature_sizes.split(',')]
  return list_of_feature_sizes


def clip_gradient_norms(gradients_to_variables, max_norm):
  clipped_grads_and_vars = []
  for grad, var in gradients_to_variables:
    if grad is not None:
      if isinstance(grad, tf.IndexedSlices):
        tmp = tf.clip_by_norm(grad.values, max_norm)
        grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
      else:
        grad = tf.clip_by_norm(grad, max_norm)
    clipped_grads_and_vars.append((grad, var))
  return clipped_grads_and_vars

def clip_variable_norms(variables, max_norm, scale=1.0):
  clipped_vars = []
  if scale != 1.0:
    for var in variables:
      if var is not None:
        if isinstance(var, tf.IndexedSlices):
          tmp = tf.clip_by_norm(var.values * scale, max_norm)
          var = tf.IndexedSlices(tmp, var.indices, var.dense_shape)
        else:
          var = tf.clip_by_norm(var * scale, max_norm)
      clipped_vars.append(var)
  else:
    for var in variables:
      if var is not None:
        if isinstance(var, tf.IndexedSlices):
          tmp = tf.clip_by_norm(var.values, max_norm)
          var = tf.IndexedSlices(tmp, var.indices, var.dense_shape)
        else:
          var = tf.clip_by_norm(var, max_norm)
      clipped_vars.append(var)
  return clipped_vars


def AddEpochSummary(summary_writer,
                    epoch_info_dict,
                    summary_scope="Eval"):

  epoch_id = epoch_info_dict["epoch_id"]
  avg_loss = epoch_info_dict["avg_loss"]
  mean_iou = epoch_info_dict["mean_iou"]

  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Avg_Loss", avg_loss),
          epoch_id)
  summary_writer.add_summary(
      MakeSummary("Epoch/" + summary_scope + "_Mean_IOU", mean_iou),
          epoch_id)
  summary_writer.flush()

  info = ("epoch/eval number {0} "
          "| Mean IOU: {1:.5f} | Avg_Loss: {2:3f}").format(
          epoch_id, mean_iou, avg_loss)
  return info
