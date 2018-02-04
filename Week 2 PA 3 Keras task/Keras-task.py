
# coding: utf-8

# # Getting deeper with Keras
# * Tensorflow is a powerful and flexible tool, but coding large neural architectures with it is tedious.
# * There are plenty of deep learning toolkits that work on top of it like Slim, TFLearn, Sonnet, Keras.
# * Choice is matter of taste and particular task
# * We'll be using Keras

# In[1]:

import sys
sys.path.append("..")
import grading


# In[2]:

# use preloaded keras datasets and models
get_ipython().system(' mkdir -p ~/.keras/datasets')
get_ipython().system(' mkdir -p ~/.keras/models')
get_ipython().system(' ln -s $(realpath ../readonly/keras/datasets/*) ~/.keras/datasets/')
get_ipython().system(' ln -s $(realpath ../readonly/keras/models/*) ~/.keras/models/')


# In[3]:

import numpy as np
from preprocessed_mnist import load_dataset
import keras
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
y_train,y_val,y_test = map(keras.utils.np_utils.to_categorical,[y_train,y_val,y_test])


# In[7]:

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.imshow(X_train[0]);
print(y_train[0])


# In[6]:

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print('\n')
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)


# ## The pretty keras

# In[8]:

import tensorflow as tf
s = tf.InteractiveSession()


# In[35]:

import keras
from keras.models import Sequential
import keras.layers as ll

model = Sequential(name="mlp")

model.add(ll.InputLayer([28, 28]))

model.add(ll.Flatten())

# network body
model.add(ll.Dense(75))
model.add(ll.Activation('relu'))

model.add(ll.Dense(125))
model.add(ll.Activation('relu'))

model.add(ll.Dense(75))
model.add(ll.Activation('relu'))

# output layer: 10 neurons for each class with softmax
model.add(ll.Dense(10, activation='softmax'))

# categorical_crossentropy is your good old crossentropy
# but applied for one-hot-encoded vectors
model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])


# In[36]:

model.summary()


# ### Model interface
# 
# Keras models follow __Scikit-learn__'s interface of fit/predict with some notable extensions. Let's take a tour.

# In[37]:

# fit(X,y) ships with a neat automatic logging.
#          Highly customizable under the hood.
model.fit(X_train, y_train,
          validation_data=(X_val, y_val), epochs=10);


# In[38]:

# estimate probabilities P(y|x)
model.predict_proba(X_val[:2])


# In[39]:

# Save trained weights
model.save("weights.h5")


# In[40]:

print("\nLoss, Accuracy = ", model.evaluate(X_test, y_test))


# ### Whoops!
# So far our model is staggeringly inefficient. There is something wring with it. Guess, what?

# In[41]:

# Test score...
test_predictions = model.predict_proba(X_test).argmax(axis=-1)
test_answers = y_test.argmax(axis=-1)

test_accuracy = np.mean(test_predictions==test_answers)

print("\nTest accuracy: {} %".format(test_accuracy*100))

assert test_accuracy>=0.92,"Logistic regression can do better!"
assert test_accuracy>=0.975,"Your network can do better!"
print("Great job!")


# In[42]:

answer_submitter = grading.Grader("0ybD9ZxxEeea8A6GzH-6CA")
answer_submitter.set_answer("N56DR", test_accuracy)


# In[43]:

answer_submitter.submit("ssq6554@126.com", "ElEuNjK3zprBIYx3")


# ## Keras + tensorboard
# 
# Remember the interactive graphs from Tensorboard one notebook ago? 
# 
# Thing is, Keras can use tensorboard to show you a lot of useful information about the learning progress. Just take a look!

# In[44]:

get_ipython().system(' rm -r /tmp/tboard/**')


# In[45]:

from keras.callbacks import TensorBoard
model.fit(X_train, y_train, validation_data=(X_val, y_val), 
          epochs=10,
          callbacks=[TensorBoard("/tmp/tboard")])


# # Tips & tricks
# 
# Here are some tips on what you could do. Don't worry, to reach the passing threshold you don't need to try all the ideas listed here, feel free to stop once you reach the 0.975 accuracy mark.
# 
#  * __Network size__
#    * More neurons, 
#    * More layers, ([docs](https://keras.io/))
# 
#    * Nonlinearities in the hidden layers
#      * tanh, relu, leaky relu, etc
#    * Larger networks may take more epochs to train, so don't discard your net just because it could didn't beat the baseline in 5 epochs.
# 
# 
#  * __Early Stopping__
#    * Training for 100 epochs regardless of anything is probably a bad idea.
#    * Some networks converge over 5 epochs, others - over 500.
#    * Way to go: stop when validation score is 10 iterations past maximum
#      
# 
#  * __Faster optimization__
#    * rmsprop, nesterov_momentum, adam, adagrad and so on.
#      * Converge faster and sometimes reach better optima
#      * It might make sense to tweak learning rate/momentum, other learning parameters, batch size and number of epochs
# 
# 
#  * __Regularize__ to prevent overfitting
#    * Add some L2 weight norm to the loss function, theano will do the rest
#      * Can be done manually or via - https://keras.io/regularizers/
#    
#    
#  * __Data augmemntation__ - getting 5x as large dataset for free is a great deal
#    * https://keras.io/preprocessing/image/
#    * Zoom-in+slice = move
#    * Rotate+zoom(to remove black stripes)
#    * any other perturbations
#    * Simple way to do that (if you have PIL/Image): 
#      * ```from scipy.misc import imrotate,imresize```
#      * and a few slicing
#    * Stay realistic. There's usually no point in flipping dogs upside down as that is not the way you usually see them.
