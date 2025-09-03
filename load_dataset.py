# load_dataset.py (without TensorFlow dependency)
from tensorflow.keras.datasets import mnist
import numpy as np

def load_dataset(split, batch_size=128, shuffle=True):
    '''
    MNIST = Modified National Institute of Standards and Technology dataset. It's the “Hello World” dataset for machine learning.
    It contains 70,000 grayscale images of handwritten digits 0-9: 60,000 training examples and 10,000 testing examples.
    Each image:
    Size = 28 * 28 pixels, and each pixel is just 1D i.e. an integer from 0 to 255
    Values = integers 0-255 (pixel intensity, 0 = black, 255 = white).
    Each label:
    An integer 0-9, telling which digit the image is.
    '''


    (x_train, y_train), (x_test, y_test) = mnist.load_data() 
    '''x_train is a numpy array of 60,000 elements, where each element is a 2d numpy of 28*28 elements from 0-255.
       y_train is a numpy array of 60,000 elements, where each element is a digit from 0-9.
       x_test, y_test are similar but havee 10,000 elements instead of 60,000.
        '''

    if split == "train":
        data = (x_train, y_train)
    else:
        data = (x_test, y_test)

    images, labels = data
    images = images.astype(np.float32) / 255.0 - 0.5 # Type converts each element from uint8 to float32, then normalizes by dividing by 255.0 and -0.5 is mean-centre

    # yield in mini-batches; describe yield function TODO: Understand yield function!
    n = len(images)
    idxs = np.arange(n)
    if shuffle:
        np.random.shuffle(idxs)

    for i in range(0, n, batch_size):
        batch_idx = idxs[i:i+batch_size]
        yield images[batch_idx], labels[batch_idx] 
'''
Fancy Indexing in numpy: In NumPy, if you index an array with another array/list of indices, it selects those rows from indexing array. 
Eg: 
A = np.array([10, 20, 30, 40, 50])
print(A[[1, 3, 4]])
# [20 40 50]
'''

''' 
yield command makes a function generator instead of a simple function i.e.
if this function is called an assigned a value the function DOES not execute at that time.
For eg, in main.py: train_ds = load_dataset("train") DOES NOT EXECUTE THIS FUNCTION as it is a function generator - rather it returns a generator
object i.e. value of local variables, code address etc.
Then when train_ds is called repeatedly, does this function run, and pause after returning each batch i.e.
The moment you write for x, y in train_ds:, Python starts iterating over the generator.
1. This is when load_dataset actually begins executing.
2. Each time the generator hits yield, it returns a batch (images[batch_idx], labels[batch_idx]) to the training loop.
3. The generator pauses at that line, then resumes the next time the loop asks for another batch.
So the multiple calls are not in your code explicitly — they come from the for loop machinery in Python, 
which repeatedly calls the generator until it is exhausted.
'''