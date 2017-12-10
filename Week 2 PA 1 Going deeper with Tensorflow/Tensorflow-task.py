
# coding: utf-8

# # Going deeper with Tensorflow
# 
# In this video, we're going to study the tools you'll use to build deep learning models. Namely, [Tensorflow](https://www.tensorflow.org/).
# 
# If you're running this notebook outside the course environment, you'll need to install tensorflow:
# * `pip install tensorflow` should install cpu-only TF on Linux & Mac OS
# * If you want GPU support from offset, see [TF install page](https://www.tensorflow.org/install/)

# In[1]:

import sys
sys.path.append("..")
import grading


# # Visualization

# Plase note that if you are running on the Coursera platform, you won't be able to access the tensorboard instance due to the network setup there. If you run the notebook locally, you should be able to access TensorBoard on http://127.0.0.1:7007/

# In[ ]:

get_ipython().system(' killall tensorboard')
import os
os.system("tensorboard --logdir=/tmp/tboard --port=7007 &");


# In[2]:

import tensorflow as tf
s = tf.InteractiveSession()


# # Warming up
# For starters, let's implement a python function that computes the sum of squares of numbers from 0 to N-1.

# In[3]:

import numpy as np
def sum_sin(N):
    return np.sum(np.arange(N)**2)


# In[4]:

get_ipython().run_cell_magic('time', '', 'sum_sin(10**8)')


# # Tensoflow teaser
# 
# Doing the very same thing

# In[5]:

# An integer parameter
N = tf.placeholder('int64', name="input_to_your_function")

# A recipe on how to produce the same result
result = tf.reduce_sum(tf.range(N)**2)


# In[6]:

result


# In[7]:

get_ipython().run_cell_magic('time', '', 'result.eval({N: 10**8})')


# In[8]:

writer = tf.summary.FileWriter("/tmp/tboard", graph=s.graph)


# # How does it work?
# 1. Define placeholders where you'll send inputs
# 2. Make symbolic graph: a recipe for mathematical transformation of those placeholders
# 3. Compute outputs of your graph with particular values for each placeholder
#   * `output.eval({placeholder:value})`
#   * `s.run(output, {placeholder:value})`
# 
# So far there are two main entities: "placeholder" and "transformation"
# * Both can be numbers, vectors, matrices, tensors, etc.
# * Both can be int32/64, floats, booleans (uint8) of various size.
# 
# * You can define new transformations as an arbitrary operation on placeholders and other transformations
#  * `tf.reduce_sum(tf.arange(N)**2)` are 3 sequential transformations of placeholder `N`
#  * There's a tensorflow symbolic version for every numpy function
#    * `a+b, a/b, a**b, ...` behave just like in numpy
#    * `np.mean` -> `tf.reduce_mean`
#    * `np.arange` -> `tf.range`
#    * `np.cumsum` -> `tf.cumsum`
#    * If if you can't find the op you need, see the [docs](https://www.tensorflow.org/api_docs/python).
#    
# `tf.contrib` has many high-level features, may be worth a look.

# In[9]:

with tf.name_scope("Placeholders_examples"):
    # Default placeholder that can be arbitrary float32
    # scalar, vertor, matrix, etc.
    arbitrary_input = tf.placeholder('float32')

    # Input vector of arbitrary length
    input_vector = tf.placeholder('float32', shape=(None,))

    # Input vector that _must_ have 10 elements and integer type
    fixed_vector = tf.placeholder('int32', shape=(10,))

    # Matrix of arbitrary n_rows and 15 columns
    # (e.g. a minibatch your data table)
    input_matrix = tf.placeholder('float32', shape=(None, 15))
    
    # You can generally use None whenever you don't need a specific shape
    input1 = tf.placeholder('float64', shape=(None, 100, None))
    input2 = tf.placeholder('int32', shape=(None, None, 3, 224, 224))

    # elementwise multiplication
    double_the_vector = input_vector*2

    # elementwise cosine
    elementwise_cosine = tf.cos(input_vector)

    # difference between squared vector and vector itself plus one
    vector_squares = input_vector**2 - input_vector + 1


# In[10]:

my_vector =  tf.placeholder('float32', shape=(None,), name="VECTOR_1")
my_vector2 = tf.placeholder('float32', shape=(None,))
my_transformation = my_vector * my_vector2 / (tf.sin(my_vector) + 1)


# In[11]:

print(my_transformation)


# In[12]:

dummy = np.arange(5).astype('float32')
print(dummy)
my_transformation.eval({my_vector:dummy, my_vector2:dummy[::-1]})


# In[13]:

writer.add_graph(my_transformation.graph)
writer.flush()


# TensorBoard allows writing scalars, images, audio, histogram. You can read more on tensorboard usage [here](https://www.tensorflow.org/get_started/graph_viz).

# # Summary
# * Tensorflow is based on computation graphs
# * The graphs consist of placehlders and transformations

# # Mean squared error
# 
# Your assignment is to implement mean squared error in tensorflow.

# In[16]:

with tf.name_scope("MSE"):
    y_true = tf.placeholder("float32", shape=(None,), name="y_true")
    y_predicted = tf.placeholder("float32", shape=(None,), name="y_predicted")
    # Your code goes here
    # You want to use tf.reduce_mean
    # mse = tf.<...>
    mse = tf.reduce_mean(tf.squared_difference(y_true, y_predicted)) 
def compute_mse(vector1, vector2):
    return mse.eval({y_true: vector1, y_predicted: vector2})


# In[17]:

writer.add_graph(mse.graph)
writer.flush()


# Tests and result submission. Please use the credentials obtained from the Coursera assignment page.

# In[18]:

import submit


# In[19]:

submit.submit_mse(compute_mse, "ssq6554@126.com", "adoEuwVXGdMn5aDs")


# # Variables
# 
# The inputs and transformations have no value outside function call. This isn't too comfortable if you want your model to have parameters (e.g. network weights) that are always present, but can change their value over time.
# 
# Tensorflow solves this with `tf.Variable` objects.
# * You can assign variable a value at any time in your graph
# * Unlike placeholders, there's no need to explicitly pass values to variables when `s.run(...)`-ing
# * You can use variables the same way you use transformations 
#  

# In[20]:

# Creating a shared variable
shared_vector_1 = tf.Variable(initial_value=np.ones(5),
                              name="example_variable")


# In[21]:

# Initialize variable(s) with initial values
s.run(tf.global_variables_initializer())

# Evaluating shared variable (outside symbolicd graph)
print("Initial value", s.run(shared_vector_1))

# Within symbolic graph you use them just
# as any other inout or transformation, not "get value" needed


# In[22]:

# Setting a new value
s.run(shared_vector_1.assign(np.arange(5)))

# Getting that new value
print("New value", s.run(shared_vector_1))


# # tf.gradients - why graphs matter
# * Tensorflow can compute derivatives and gradients automatically using the computation graph
# * True to its name it can manage matrix derivatives
# * Gradients are computed as a product of elementary derivatives via the chain rule:
# 
# $$ {\partial f(g(x)) \over \partial x} = {\partial f(g(x)) \over \partial g(x)}\cdot {\partial g(x) \over \partial x} $$
# 
# It can get you the derivative of any graph as long as it knows how to differentiate elementary operations

# In[23]:

my_scalar = tf.placeholder('float32')

scalar_squared = my_scalar**2

# A derivative of scalar_squared by my_scalar
derivative = tf.gradients(scalar_squared, [my_scalar, ])


# In[24]:

derivative


# In[25]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

x = np.linspace(-3, 3)
x_squared, x_squared_der = s.run([scalar_squared, derivative[0]],
                                 {my_scalar:x})

plt.plot(x, x_squared,label="$x^2$")
plt.plot(x, x_squared_der, label=r"$\frac{dx^2}{dx}$")
plt.legend();


# # Why that rocks

# In[26]:

my_vector = tf.placeholder('float32', [None])
# Compute the gradient of the next weird function over my_scalar and my_vector
# Warning! Trying to understand the meaning of that function may result in permanent brain damage
weird_psychotic_function = tf.reduce_mean(
    (my_vector+my_scalar)**(1+tf.nn.moments(my_vector,[0])[1]) + 
    1./ tf.atan(my_scalar))/(my_scalar**2 + 1) + 0.01*tf.sin(
    2*my_scalar**1.5)*(tf.reduce_sum(my_vector)* my_scalar**2
                      )*tf.exp((my_scalar-4)**2)/(
    1+tf.exp((my_scalar-4)**2))*(1.-(tf.exp(-(my_scalar-4)**2)
                                    )/(1+tf.exp(-(my_scalar-4)**2)))**2

der_by_scalar = tf.gradients(weird_psychotic_function, my_scalar)
der_by_vector = tf.gradients(weird_psychotic_function, my_vector)


# In[27]:

# Plotting the derivative
scalar_space = np.linspace(1, 7, 100)

y = [s.run(weird_psychotic_function, {my_scalar:x, my_vector:[1, 2, 3]})
     for x in scalar_space]

plt.plot(scalar_space, y, label='function')

y_der_by_scalar = [s.run(der_by_scalar,
                         {my_scalar:x, my_vector:[1, 2, 3]})
                   for x in scalar_space]

plt.plot(scalar_space, y_der_by_scalar, label='derivative')
plt.grid()
plt.legend();


# # Almost done - optimizers
# 
# While you can perform gradient descent by hand with automatic grads from above, tensorflow also has some optimization methods implemented for you. Recall momentum & rmsprop?

# In[28]:

y_guess = tf.Variable(np.zeros(2, dtype='float32'))
y_true = tf.range(1, 3, dtype='float32')
loss = tf.reduce_mean((y_guess - y_true + tf.random_normal([2]))**2) 
#loss = tf.reduce_mean((y_guess - y_true)**2) 
optimizer = tf.train.MomentumOptimizer(0.01, 0.5).minimize(
    loss, var_list=y_guess)


# In[29]:

from matplotlib import animation, rc
import matplotlib_utils
from IPython.display import HTML

fig, ax = plt.subplots()
y_true_value = s.run(y_true)
level_x = np.arange(0, 2, 0.02)
level_y = np.arange(0, 3, 0.02)
X, Y = np.meshgrid(level_x, level_y)
Z = (X - y_true_value[0])**2 + (Y - y_true_value[1])**2
ax.set_xlim(-0.02, 2)
ax.set_ylim(-0.02, 3)
s.run(tf.global_variables_initializer())
ax.scatter(*s.run(y_true), c='red')
contour = ax.contour(X, Y, Z, 10)
ax.clabel(contour, inline=1, fontsize=10)
line, = ax.plot([], [], lw=2)

def init():
    line.set_data([], [])
    return (line,)

guesses = [s.run(y_guess)]

def animate(i):
    s.run(optimizer)
    guesses.append(s.run(y_guess))
    line.set_data(*zip(*guesses))
    return (line,)

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=400, interval=20, blit=True)


# In[ ]:

try:
    HTML(anim.to_html5_video())
# In case the build-in renderers are unaviable, fall back to
# a custom one, that doesn't require external libraries
except RuntimeError:
    anim.save(None, writer=matplotlib_utils.SimpleMovieWriter(0.001))


# # Logistic regression
# Your assignment is to implement the logistic regression
# 
# Plan:
# * Use a shared variable for weights
# * Use a matrix placeholder for `X`
#  
# We shall train on a two-class MNIST dataset
# * please note that target `y` are `{0,1}` and not `{-1,1}` as in some formulae

# In[31]:

from sklearn.datasets import load_digits
mnist = load_digits(2)

X, y = mnist.data, mnist.target

print("y [shape - %s]:" % (str(y.shape)), y[:10])
print("X [shape - %s]:" % (str(X.shape)))


# In[32]:

print('X:\n',X[:3,:10])
print('y:\n',y[:10])
plt.imshow(X[0].reshape([8,8]));


# It's your turn now!
# Just a small reminder of the relevant math:
# 
# $$
# P(y=1|X) = \sigma(X \cdot W + b)
# $$
# $$
# \text{loss} = -\log\left(P\left(y_\text{predicted} = 1\right)\right)\cdot y_\text{true} - \log\left(1 - P\left(y_\text{predicted} = 1\right)\right)\cdot\left(1 - y_\text{true}\right)
# $$
# 
# $\sigma(x)$ is available via `tf.nn.sigmoid` and matrix multiplication via `tf.matmul`

# In[33]:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)


# __Your code goes here.__ For the training and testing scaffolding to work, please stick to the names in comments.

# In[102]:

# Model parameters - weights and bias
# weights = tf.Variable(...) shape should be (X.shape[1], 1)
# b = tf.Variable(...)

weights = tf.Variable(initial_value=np.random.randn(X.shape[1], 1)*0.01, name="weights", dtype="float32")
b = tf.Variable(initial_value=0, name="b", dtype="float32")
print(weights)
print(b)


# In[103]:

# Placeholders for the input data
# input_X = tf.placeholder(...)
# input_y = tf.placeholder(...)

input_X = tf.placeholder(tf.float32, name="input_X")
input_y = tf.placeholder(tf.float32, name="input_y")
print(input_X)
print(input_y)


# In[104]:

# The model code

# Compute a vector of predictions, resulting shape should be [input_X.shape[0],]
# This is 1D, if you have extra dimensions, you can  get rid of them with tf.squeeze .
# Don't forget the sigmoid.
# predicted_y = <predicted probabilities for input_X>
predicted_y = tf.squeeze(tf.nn.sigmoid(tf.add(tf.matmul(input_X, weights), b)))
print(predicted_y)

# Loss. Should be a scalar number - average loss over all the objects
# tf.reduce_mean is your friend here
# loss = <logistic loss (scalar, mean over sample)>
loss = -tf.reduce_mean(tf.log(predicted_y)*input_y + tf.log(1-predicted_y)*(1-input_y))
print(loss)

# See above for an example. tf.train.*Optimizer
# optimizer = <optimizer that minimizes loss>
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
print(optimizer)


# A test to help with the debugging

# In[105]:

validation_weights = 1e-3 * np.fromiter(map(lambda x:
        s.run(weird_psychotic_function, {my_scalar:x, my_vector:[1, 0.1, 2]}),
                                   0.15 * np.arange(1, X.shape[1] + 1)),
                                   count=X.shape[1], dtype=np.float32)[:, np.newaxis]
# Compute predictions for given weights and bias
prediction_validation = s.run(
    predicted_y, {
    input_X: X,
    weights: validation_weights,
    b: 1e-1})

# Load the reference values for the predictions
validation_true_values = np.loadtxt("validation_predictons.txt")

assert prediction_validation.shape == (X.shape[0],),       "Predictions must be a 1D array with length equal to the number "        "of examples in input_X"
assert np.allclose(validation_true_values, prediction_validation)
loss_validation = s.run(
        loss, {
            input_X: X[:100],
            input_y: y[-100:],
            weights: validation_weights+1.21e-3,
            b: -1e-1})
assert np.allclose(loss_validation, 0.728689)


# In[106]:

from sklearn.metrics import roc_auc_score
s.run(tf.global_variables_initializer())
for i in range(5):
    s.run(optimizer, {input_X: X_train, input_y: y_train})
    loss_i = s.run(loss, {input_X: X_train, input_y: y_train})
    print("loss at iter %i:%.4f" % (i, loss_i))
    print("train auc:", roc_auc_score(y_train, s.run(predicted_y, {input_X:X_train})))
    print("test auc:", roc_auc_score(y_test, s.run(predicted_y, {input_X:X_test})))


# ### Coursera submission

# In[107]:

grade_submitter = grading.Grader("BJCiiY8sEeeCnhKCj4fcOA")


# In[108]:

test_weights = 1e-3 * np.fromiter(map(lambda x:
    s.run(weird_psychotic_function, {my_scalar:x, my_vector:[1, 2, 3]}),
                               0.1 * np.arange(1, X.shape[1] + 1)),
                               count=X.shape[1], dtype=np.float32)[:, np.newaxis]


# First, test prediction and loss computation. This part doesn't require a fitted model.

# In[109]:

prediction_test = s.run(
    predicted_y, {
    input_X: X,
    weights: test_weights,
    b: 1e-1})


# In[110]:

assert prediction_test.shape == (X.shape[0],),       "Predictions must be a 1D array with length equal to the number "        "of examples in X_test"


# In[111]:

grade_submitter.set_answer("0ENlN", prediction_test)


# In[112]:

loss_test = s.run(
    loss, {
        input_X: X[:100],
        input_y: y[-100:],
        weights: test_weights+1.21e-3,
        b: -1e-1})
# Yes, the X/y indices mistmach is intentional


# In[113]:

grade_submitter.set_answer("mMVpM", loss_test)


# In[114]:

grade_submitter.set_answer("D16Rc", roc_auc_score(y_test, s.run(predicted_y, {input_X:X_test})))


# Please use the credentials obtained from the Coursera assignment page.

# In[115]:

grade_submitter.submit("ssq6554@126.com", "zfkj43piwD65Symi")


# In[ ]:



