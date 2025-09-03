# FNN_architecture.py
import jax
import jax.numpy as jnp
from quantization import fake_quantize
import config as conf



def init_params(key, layer_sizes):
    """
    Initialize weights and biases for a feedforward NN.
    Xavier/He initialization used.
    """
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1) 
    
    '''
    JAX doesn’t let you “re-use” the same random key — every random op consumes entropy. To keep randomness reproducible (and parallelizable), 
    you must split keys.
    Example:
    key = jax.random.PRNGKey(42)       # master seed
    key1, key2, key3 = jax.random.split(key, 3)   # split into 3 new independent keys
    Each keyX is another PRNGKey you can pass into random functions.
    So in your code: You need 3 layers → 3 sets of random weights → 3 independent keys.
    '''
    
    for m, n, k in zip(layer_sizes[:-1], layer_sizes[1:], keys): #zip combines multiple lists element by element.
        W = jax.random.normal(k, (m, n)) * jnp.sqrt(2.0 / m)
        '''
        random.normal(k, shape) generates random numbers from a standard normal distribution (mean=0, std=1) using random key k.
        shape=(m, n) makes it a 2D matrix.
        Then multiplying by jnp.sqrt(2.0/m) scales the variance (He initialization).
        '''
        b = jnp.zeros(n)
        params.append((W, b))
    return params


def tanh_activation (activations):
    return jnp.tanh(activations)

def RELU_activation (activations):
    return jnp.maximum(activations, 0)

def _linear_qat(x, W, b):
    if conf.QAT_ENABLE:
        x = fake_quantize(x, conf.A_BITS)     # quantize activations in
        W = fake_quantize(W, conf.W_BITS)     # quantize weights
        y = jnp.dot(x, W) + b
        y = fake_quantize(y, conf.A_BITS)     # quantize activations out
        return y
    else:
        return jnp.dot(x, W) + b

#Recall params is a list of (W,b) pairs for each layer; x is a set of batch_size datapoint inputs so array of dimension batch_size*28*28
def forward(params, x):
    """
    Forward pass through the NN.
    """
    activations = x.reshape(-1, 28*28)  #x.reshape(-1, 28*28) → reshape (batch_size, 28, 28) into (batch_size, 784) so each row is one flattened image. By default, .reshape() uses row-major (C-style) order
    for W, b in params[:-1]:
        activations = _linear_qat(activations, W, b) #Q(W)*Q(x) + Q(b)
        activations = tanh_activation(activations) #tanh nonlinear activation
        # activations = RELU_activation(activations) #RELU nonlinear activation
        if conf.QAT_ENABLE:
            activations = fake_quantize(activations, conf.A_BITS)
    W, b = params[-1]
    logits = _linear_qat(activations, W, b)  #No activation applied to final layer, because softmax is applied later in loss_fn
    return logits

def loss_fn(params, x, y):
    """
    Cross-entropy loss.
    """
    logits = forward(params, x) #Basically with the weights given in params and dataset of size 'batch_size' what is output?
    labels = jax.nn.one_hot(y, 10) 
    '''Eg: If y = [2, 0, 9], then
    [[0,0,1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1]]
    '''
    loss = -jnp.mean(jnp.sum(labels * jax.nn.log_softmax(logits), axis=1))
    '''
    (log of softmax outputs value in range (-inf, 0))!
    Cross-entropy loss:
        L = - (1/N) * sum_{i=1}^N log( p_theta(y_i | x_i) )
        where:
        N = batch size
        y_i = true class label for example i
        p_theta(y_i | x_i) = predicted probability (softmax) of the true class
    
    Compute log of softmax of the unactivated output. Then multiply with labels, where multiply is element wise multiplication of each correspoinding
    datapoint. As labels is a one-shot vector, this simply picks the log of softmax of correct label and outputs 0 for the other entries, then the sum
    just returns the log of softmax of the correct label. Then the mean takes the mean over the batch_size datatpoints.
    '''
    return loss

def accuracy(params, x, y):
    """
    Compute accuracy.
    """
    preds = jnp.argmax(forward(params, x), axis=1) #So here no need for softmax, as you are taking argmax either way.
    return jnp.mean(preds == y) #Basically sum of 0-1 erros on each datapoint