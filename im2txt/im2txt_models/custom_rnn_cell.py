from tensorflow.python.ops.rnn_cell_impl import *

class StackedWrapper(RNNCell):
  """RNNCell wrapper that ensures cell inputs are concatenated to the outputs."""

  def __init__(self, cell, stack_fn=None):
    """Constructs a `StackedWrapper` for `cell`.
    Args:
      cell: An instance of `RNNCell`.
      stack_fn: (Optional) The function to map raw cell inputs and raw cell
        outputs to the actual cell outputs of the stack network.
        Defaults to calling nest.map_structure on (lambda i, o: i + o), inputs
        and outputs.
    """
    self._cell = cell
    self._stack_fn = stack_fn

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size + self._cell.input_size

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      return self._cell.zero_state(batch_size, dtype)

  def __call__(self, inputs, state, scope=None):
    """Run the cell and then apply the residual_fn on its inputs to its outputs.
    Args:
      inputs: cell inputs.
      state: cell state.
      scope: optional cell scope.
    Returns:
      Tuple of cell outputs and new state.
    Raises:
      TypeError: If cell inputs and outputs have different structure (type).
      ValueError: If cell inputs and outputs have different structure (value).
    """
    outputs, new_state = self._cell(inputs, state, scope=scope)
    # Ensure shapes match
    def default_stack_fn(inputs, outputs):
      return nest.map_structure(lambda inp, out: tf.concat([inp, out], axis=-1), inputs, outputs)
    res_outputs = (self._stack_fn or default_stack_fn)(inputs, outputs)
    return (res_outputs, new_state)


