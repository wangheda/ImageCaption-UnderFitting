import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import *

class TopDownRNNCell(RNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, att_cell, lm_cell, memory, attention_size, state_is_tuple=True):
    """Create a RNN cell composed sequentially of a number of RNNCells.
    Args:
      att_cell: an RNNCell
      lm_cell: an RNNCell
      memory: [batch, k, size] tensor
      state_is_tuple: If True, accepted and returned states are n-tuples, where
        `n = len(cells)`.  If False, the states are all
        concatenated along the column axis.  This latter behavior will soon be
        deprecated.
    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    super(TopDownRNNCell, self).__init__()
    cells = [att_cell, lm_cell]
    if not nest.is_sequence(cells):
      raise TypeError(
          "cells must be a list or tuple, but saw: %s." % cells)

    self._att_cell = att_cell
    self._lm_cell = lm_cell
    self._cells = cells
    self._memory = memory
    self._avg_memory = tf.reduce_mean(memory, axis=1)
    self._state_is_tuple = state_is_tuple
    self._V = self._memory.get_shape().as_list()[-1]
    if state_is_tuple:
      self._M = self._lm_cell.state_size.h
    else:
      self._M = self._lm_cell._num_units
    self._H = attention_size

    with vs.variable_scope("top_down_cell"):
      self._W_va = tf.get_variable("W_va", shape=[self._V, self._H], dtype=tf.float32)
      self._W_ha = tf.get_variable("W_ha", shape=[self._M, self._H], dtype=tf.float32)
      self._W_a = tf.get_variable("W_a", shape=[self._H], dtype=tf.float32)

    self._Wv = tf.einsum("ijk,kl->ijl", self._memory, self._W_va)

    if not state_is_tuple:
      if any(nest.is_sequence(c.state_size) for c in self._cells):
        raise ValueError("Some cells return tuples of states, but the flag "
                         "state_is_tuple is not set.  State sizes are: %s"
                         % str([c.state_size for c in self._cells]))

  @property
  def state_size(self):
    if self._state_is_tuple:
      return tuple(cell.state_size for cell in self._cells)
    else:
      return sum([cell.state_size for cell in self._cells])

  @property
  def output_size(self):
    return self._cells[-1].output_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._state_is_tuple:
        return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)
      else:
        # We know here that state_size of each cell is not a tuple and
        # presumably does not contain TensorArrays or anything else fancy
        return super(TopDownRNNCell, self).zero_state(batch_size, dtype)

  def call(self, inputs, state):
    """Run this multi-layer cell on inputs, starting from state."""
    new_states = []

    with vs.variable_scope("att_cell"):
      if self._state_is_tuple:
        if not nest.is_sequence(state):
          raise ValueError(
              "Expected state to be a tuple of length %d, but received: %s" %
              (len(self.state_size), state))
        cur_att_state = state[0]
        cur_lm_state = state[1]
        lm_state_as_input = cur_lm_state[1] # h-state
      else:
        cur_state_pos = 0
        cur_att_state = array_ops.slice(state, [0, cur_state_pos],
                                    [-1, att_cell.state_size])
        cur_state_pos += self._att_cell.state_size
        cur_lm_state = array_ops.slice(state, [0, cur_state_pos],
                                    [-1, self._lm_cell.state_size])
        lm_state_as_input = array_ops.slice(cur_lm_state, [0, 0],
                                    [-1, self._lm_cell._num_units])
        cur_state_pos += lm_cell.state_size

      cur_att_inp = tf.concat([lm_state_as_input, self._avg_memory, inputs], axis=1)
      cur_lm_inp, new_att_state = self._att_cell(cur_att_inp, cur_att_state)
      new_states.append(new_att_state)

    with vs.variable_scope("attend"):
      if self._state_is_tuple:
        att_state_as_input = new_att_state[1]
      else:
        att_state_as_input = array_ops.slice(new_att_state, [0, 0],
                                    [-1, self._att_cell._num_units])
      activation = self._Wv + tf.expand_dims(tf.einsum("ij,jk->ik", att_state_as_input, self._W_ha), axis=1)
      att = tf.einsum("ijk,k->ij", tf.nn.tanh(activation), self._W_a)
      att = tf.nn.softmax(att, dim=-1)
      att_memory = tf.einsum("ijk,ij->ik", self._memory, att)

    with vs.variable_scope("lm_cell"):
      if self._state_is_tuple:
        if not nest.is_sequence(state):
          raise ValueError(
              "Expected state to be a tuple of length %d, but received: %s" %
              (len(self.state_size), state))
        cur_att_state = state[0]
        cur_lm_state = state[1]
      else:
        cur_state_pos = 0
        cur_state_pos += self._att_cell.state_size
        cur_lm_state = array_ops.slice(state, [0, cur_state_pos],
                                    [-1, self._lm_cell.state_size])
        cur_state_pos += self._lm_cell.state_size
      cur_lm_inp = tf.concat([att_memory, att_state_as_input], axis=1)
      lm_output, new_lm_state = self._lm_cell(cur_lm_inp, cur_lm_state)
      new_states.append(new_lm_state)

    new_states = (tuple(new_states) if self._state_is_tuple else
                  array_ops.concat(new_states, 1))

    return lm_output, new_states


