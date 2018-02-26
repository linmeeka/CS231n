import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    #? 
    conv_param1 = {'stride': 1, 'pad': (filter_size - 1) / 2}
    pool_param1 = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    
    C,H,W=input_dim
    HH=WW=filter_size
    pad_conv,stride_conv=conv_param1['pad'],conv_param1['stride']
    pool_H,pool_W,stride_pool=pool_param1['pool_height'],pool_param1['pool_width'],pool_param1['stride']
    
    conv_H=1+(H+2*pad_conv-HH)/stride_conv
    conv_W=1+(W+2*pad_conv-WW)/stride_conv
    out_H=1+(conv_H-pool_H)/stride_pool
    out_W=1+(conv_W-pool_W)/stride_pool
    D=out_H*out_W*num_filters  #C?
    
    W1=np.random.randn(num_filters,C,HH,WW)*weight_scale
    b1=np.zeros(num_filters)
    W2=np.random.randn(D,hidden_dim)*weight_scale
    b2=np.zeros(hidden_dim)
    W3=np.random.randn(hidden_dim,num_classes)*weight_scale
    b3=np.zeros(num_classes)
    
    self.params['W1']=W1
    self.params['b1']=b1
    self.params['W2']=W2
    self.params['b2']=b2
    self.params['W3']=W3
    self.params['b3']=b3
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    #conv - relu - 2x2 max pool - affine - relu - affine - softmax

    out_conv,cache_conv=conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    out_fc,cache_fc=affine_relu_forward(out_conv, W2, b2)
    scores,cache_score=affine_forward(out_fc, W3, b3)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss,dscore=softmax_loss(scores,y)
    d_fc,dW3,db3=affine_backward(dscore, cache_score)
    d_conv,dW2,db2=affine_relu_backward(d_fc, cache_fc)
    dx,dW1,db1=conv_relu_pool_backward(d_conv, cache_conv)
    
    reg=self.reg
    loss+=0.5*reg*(np.sum(W1**2)+np.sum(W2**2)+np.sum(W3**2))
    dW3+=reg*W3
    dW2+=reg*W2
    dW1+=reg*W1
    
    grads['W3']=dW3
    grads['W2']=dW2
    grads['W1']=dW1
    grads['b3']=db3
    grads['b2']=db2
    grads['b1']=db1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
