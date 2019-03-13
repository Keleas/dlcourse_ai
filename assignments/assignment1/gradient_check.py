import numpy as np


def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    print(fx, analytic_grad)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        # print(np.array(x[ix]) + delta)
        # print(f(np.array(x[ix]) + delta))
        analytic_grad_at_ix = analytic_grad[ix]

        def _inc(x, delta):
            return x+delta
        inc_delta = np.vectorize(_inc)

        numeric_grad_at_ix = 0
        arg_x_1 = inc_delta(x, delta)
        arg_x_2 = inc_delta(x, -delta)
        fx_1, _ = f(arg_x_1)
        fx_2, _ = f(arg_x_2)
        numeric_grad_at_ix = (fx_1 - fx_2) / (2*delta)
        print(fx_1)

        # TODO compute value of numeric gradient of f to idx
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True

        

        
