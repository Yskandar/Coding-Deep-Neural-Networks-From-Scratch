import numpy as np
from .layers import *
from .layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dims=100, num_classes=10,
               dropout=0, weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dims: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg

    # Initialization
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims)
    self.params['b1'] = np.zeros(hidden_dims)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dims, num_classes)
    self.params['b2'] = np.zeros(num_classes)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N, C) giving classification scores, where
      scores[i, c] is the classification score for X[i] and class c.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None

    out, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
    scores, cache2 = affine_forward(out, self.params['W2'], self.params['b2'])
    
    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    # Let us implement the backward pass  
    L2_reg = np.sum(self.params['W2'] ** 2) + np.sum(self.params['W1'] ** 2)  # regularization
    s_loss, s_grad = softmax_loss(scores, y)  # getting the initial upstream derivative
    loss =  s_loss + 0.5*self.reg*L2_reg  # computing the total loss
    dout, dW2, db2 = affine_backward(s_grad, cache2)  # computing the following derivatives
    dout, dW1, db1 = affine_relu_backward(dout, cache1)
    dreg_dW1 = 2*self.params['W1']*0.5*self.reg
    dreg_dW2 = 2*self.params['W2']*0.5*self.reg

    # Updating the gradients
    grads['W1'] = dW1 + dreg_dW1
    grads['W2'] = dW2 + dreg_dW2
    grads['b1'] = db1
    grads['b2'] = db2
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}
    self.hidden_dims = hidden_dims

    
    # First layer
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
    self.params['b1'] = np.zeros(hidden_dims[0])
    if self.use_batchnorm:
      self.params['gamma1'] = np.ones((hidden_dims[0],))
      self.params['beta1'] = np.zeros((hidden_dims[0],))

    for i in range(1, len(hidden_dims)):  # Rest of the hidden layers

      self.params['W{}'.format(i+1)] = weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])
      self.params['b{}'.format(i+1)] = np.zeros(hidden_dims[i])
      if self.use_batchnorm:
        self.params['gamma{}'.format(i+1)] = np.ones((hidden_dims[i],))
        self.params['beta{}'.format(i+1)] = np.zeros((hidden_dims[i],))


    # Last layer
    self.params['W{}'.format(i+2)] = weight_scale * np.random.randn(hidden_dims[i], num_classes)
    self.params['b{}'.format(i+2)] = np.zeros(num_classes)

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). We pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. WE pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in np.arange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    
    # Let us implement the forward pass of the FC
    outs = [X]
    caches = []
    
    if not self.use_batchnorm:
      
      if not self.use_dropout:
        for i in range(len(self.hidden_dims)):  # going through the hidden layers
          out, cache = affine_relu_forward(outs[-1], self.params['W{}'.format(i+1)], self.params['b{}'.format(i+1)])
          outs.append(out)  # storing the output of the current layer
          caches.append(cache)  # storing the cache
      
      if self.use_dropout:
        for i in range(len(self.hidden_dims)):  # going through the hidden layers
          out, cache = affine_relu_forward(outs[-1], self.params['W{}'.format(i+1)], self.params['b{}'.format(i+1)])
          out, dp_cache = dropout_forward(out, self.dropout_param)
          outs.append(out)  # storing the output of the current layer
          cache = (cache, dp_cache) # (affine relu cache, dropout cache) 
          caches.append(cache)  # storing the cache        


    if self.use_batchnorm:
      
      if not self.use_dropout:
        for i in range(len(self.hidden_dims)):  # going through the hidden layers
          out, cache = affine_batchnorm_relu_forward(outs[-1], self.params['W{}'.format(i+1)], self.params['b{}'.format(i+1)],
                                                    self.params['gamma{}'.format(i+1)], self.params['beta{}'.format(i+1)], self.bn_params[i])
          outs.append(out)  # storing the output of the current layer
          caches.append(cache)  # storing the cache
      
      if self.use_dropout:
        for i in range(len(self.hidden_dims)):  # going through the hidden layers       
          out, cache = affine_batchnorm_relu_forward(outs[-1], self.params['W{}'.format(i+1)], self.params['b{}'.format(i+1)],
                                                    self.params['gamma{}'.format(i+1)], self.params['beta{}'.format(i+1)], self.bn_params[i])
          out, dp_cache = dropout_forward(out, self.dropout_param)
          outs.append(out)  # storing the output of the current layer
          cache = (cache, dp_cache) # (affine batchnorm relu cache, dropout cache) 
          caches.append(cache)  # storing the cache

    scores, cache = affine_forward(outs[-1], self.params['W{}'.format(i+2)], self.params['b{}'.format(i+2)])  # handling the last layer

    
    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}

    # Let us implement the backward pass of the FCnet
    L2_reg = np.sum([np.sum(self.params['W{}'.format(i)] ** 2) for i in range(1, len(self.hidden_dims) + 2)])  # regularization term
    s_loss, s_grad = softmax_loss(scores, y)  # softmax loss
    loss =  s_loss + 0.5*self.reg*L2_reg
    douts = [s_grad]

    # getting and storing the upstream derivative of the last layer + the relative derivatives
    dout, grads['W{}'.format(self.num_layers)], grads['b{}'.format(self.num_layers)] = affine_backward(s_grad, cache)
    
    # adding the impact of the reg term on the derivative
    grads['W{}'.format(self.num_layers)] += 2*self.params['W{}'.format(self.num_layers)]*0.5*self.reg
    douts.append(dout)


    if not self.use_batchnorm:

      if not self.use_dropout:
        # getting and storing the upstream derivative of the hidden layers + the relative derivatives
        for i in range(len(self.hidden_dims)-1, -1, -1):
          dout, grads['W{}'.format(i+1)], grads['b{}'.format(i+1)] = affine_relu_backward(douts[-1], caches[i])  # backward pass
          douts.append(dout)  # storing the upstream derivative
          grads['W{}'.format(i+1)] += 2*self.params['W{}'.format(i+1)]*0.5*self.reg  # regularization impact
      
      if self.use_dropout:
        # getting and storing the upstream derivative of the hidden layers + the relative derivatives
        for i in range(len(self.hidden_dims)-1, -1, -1):
          ar_cache, dp_cache = caches[i]  # (affine relu cache, dropout cache) 
          dout2 = dropout_backward(douts[-1], dp_cache)  # backward pass - droupout
          dout, grads['W{}'.format(i+1)], grads['b{}'.format(i+1)] = affine_relu_backward(dout2, ar_cache)  # backward pass - affine relu
          douts.append(dout)  # storing the upstream derivative
          grads['W{}'.format(i+1)] += 2*self.params['W{}'.format(i+1)]*0.5*self.reg  # regularization impact
    
    
    if self.use_batchnorm:
      
      if not self.use_dropout:
        # getting and storing the upstream derivative of the hidden layers + the relative derivatives
        for i in range(len(self.hidden_dims)-1, -1, -1):
          dout, grads['W{}'.format(i+1)], grads['b{}'.format(i+1)], grads['gamma{}'.format(i+1)], grads['beta{}'.format(i+1)]  = affine_batchnorm_relu_backward(douts[-1], caches[i])  # backward pass
          douts.append(dout)  # storing the upstream derivative
          grads['W{}'.format(i+1)] += 2*self.params['W{}'.format(i+1)]*0.5*self.reg  # regularization impact
      
      if self.use_dropout:
        # getting and storing the upstream derivative of the hidden layers + the relative derivatives
        for i in range(len(self.hidden_dims)-1, -1, -1):
          abr_cache, dp_cache = caches[i]  # (affine batchnorm relu cache, dropout cache) 
          dout2 = dropout_backward(douts[-1], dp_cache)  # backward pass - droupout
          dout, grads['W{}'.format(i+1)], grads['b{}'.format(i+1)], grads['gamma{}'.format(i+1)], grads['beta{}'.format(i+1)]  = affine_batchnorm_relu_backward(dout2, abr_cache)  # backward pass - affine batchnorm relu
          douts.append(dout)  # storing the upstream derivative
          grads['W{}'.format(i+1)] += 2*self.params['W{}'.format(i+1)]*0.5*self.reg  # regularization impact      
    
    
    return loss, grads
