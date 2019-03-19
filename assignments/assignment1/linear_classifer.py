import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    predictions -= np.max(predictions)
    return np.exp(predictions) / np.sum(np.exp(predictions))


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy

    eps = 1e-9
    target = np.zeros(probs.shape[0])
    target[target_index] = 1
    log_like = np.log(probs + eps)
    loss = - np.sum(target * log_like)

    return loss


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy

    predictions = np.array(predictions)
    n_samples = predictions.shape[0]
    try:
        target_index[0]
        axis = 1
    except:
        axis = 0

    s = np.max(predictions, axis=axis)
    if axis == 1:
        s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(predictions - s)
    div = np.sum(e_x, axis=axis)
    if axis ==1:
        div = div[:, np.newaxis] # dito
    softmax_x = e_x / div

    eps = 1e-9
    if axis == 1:
        target = np.zeros(predictions.shape[1])
        target[target_index[0]] = 1
    elif axis == 0:
        target = np.zeros(3)
        target[target_index] = 1

    log_like = np.log(softmax_x + eps)
    loss = - np.sum(target * log_like) / n_samples

    dprediction = (softmax_x - target) / n_samples

    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient

    loss = reg_strength * np.linalg.norm(W) ** 2
    grad = 2 * reg_strength * W

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''

    predictions = np.dot(X, W)
    # TODO implement prediction and gradient over W

    loss, dprediction = softmax_with_cross_entropy(predictions, target_index)

    dW = np.dot(X.T, dprediction)

    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)

            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            losses = []
            for batch in batches_indices:
                x_batch = X[batch]
                y_batch = y[batch]

                loss_func, grad_func = linear_softmax(x_batch, self.W, y_batch)
                loss_reg, grad_reg = l2_regularization(self.W, reg)

                print('func', loss_func)
                print('reg', loss_reg)

                loss_cur = loss_func + loss_reg
                losses.append(loss_cur)
                self.W -= learning_rate * grad_func + grad_reg
                print(grad_reg)

            # end
            loss = np.mean(losses)
            print("Epoch %i, loss: %f" % (epoch, loss))
            loss_history.append(loss)

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)

        # TODO Implement class prediction
        raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
