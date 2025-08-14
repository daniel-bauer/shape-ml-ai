import numpy as np


def sigmoid(z): 
  """
  Compute the sigmoid (for each element) of input vector z.
  """
  return 1/(1 + np.exp(-z))


class Layer:

  def __init__(self, in_dim, out_dim, activation=sigmoid):    
    self.weights = np.ones(in_dim, out_dim)
    self.activation = activation

  def forward(self, input):
    assert input.shape[-1] == self.weights(0)
    return self.activation(np.matmul(self.weights, self.input))



  def backprop(self, gradient):
    """
    Update the weights for this layer. Compute gradient, then propagate back.
    """
