"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import tensorflow

from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest

class ResLSTMWrapper(rnn_cell_impl.RNNCell):
  def __init__(self,cell,
               couple_carry_trainsform_gates=True,
               carry_bias_init=1.0):
    self._cell = cell
    self._couple_carry_transform_gates = couple_carry_trainsform_gates
    self._carry_bias_init = carry_bias_init

  @property
  def state_size(self):
    return self._cell.state_size
        
  def output_size(self):
    return self._cell.output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def _res(self, inp, out):
    input_size = inp.get_shape().with_rank(2)[1].value
    return tensorflow.concat(axis=-1, values=[inp, out])

  def __call__(self, inputs, state, scope=None):
    outputs, new_state = self._cell(inputs, state, scope=scope)
    nest.assert_same_structure(inputs, outputs)
    def assert_shape_match(inp, out):
      inp.get_shape().assert_is_compatible_with(out.get_shape())

    nest.map_structure(assert_shape_match, inputs, outputs)
    res_outputs = nest.map_structure(self._res, inputs, outputs)
    return (res_outputs, new_state)
