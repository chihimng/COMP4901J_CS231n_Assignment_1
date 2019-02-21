import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
    scores = X[i].dot(W)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)
    loss -= np.log(probs[y[i]])
    for j in xrange(num_classes):
      # math magic: http://cs231n.github.io/neural-networks-case-study/#grad
      # dL_i/df_k = p_k - (y_i == k ? 1 : 0)
      # then backpropate by X[i] * dL_i
      if j == y[i]:
        dW[:,j] += X[i] * (probs[j] - 1)
      else:
        dW[:,j] += X[i] * probs[j]
  loss = loss / num_train + reg * np.sum(W ** 2)
  dW = dW / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  
  scores = X.dot(W)
  stable_scores = scores - np.max(scores) # avoid overflow, see http://cs231n.github.io/linear-classify/ (Numeric Stability)
  exp_scores = np.exp(stable_scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  loss -= np.log(probs[range(num_train),y])
  loss = np.sum(loss) / num_train + reg * np.sum(W ** 2)

  # math magic: http://cs231n.github.io/neural-networks-case-study/#grad
  # dL_i/df_k = p_k - (y_i == k ? 1 : 0)
  # then backpropagate by X[i] * dL_i
  dScores = probs
  dScores[range(num_train), y] -= 1
  dScores /= num_train
  dW = X.T.dot(dScores) + reg * 2 * W # 2 is from d(W^2)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

