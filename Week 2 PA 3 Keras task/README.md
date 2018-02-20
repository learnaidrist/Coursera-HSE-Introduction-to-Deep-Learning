# Getting deeper with Keras
```
Epoch 10/10
50000/50000 [==============================] - 7s - loss: 0.0285 - acc: 0.9908 - val_loss: 0.1104 - val_acc: 0.9731

9216/10000 [==========================>...] - ETA: 0s
Loss, Accuracy =  [0.10402157519632602, 0.9758]
```

# Description
- Tensorflow is a powerful and flexible tool, but coding large neural architectures with it is tedious.
- There are plenty of deep learning toolkits that work on top of it like Slim, TFLearn, Sonnet, Keras.
- Choice is matter of taste and particular task
- We'll be using Keras

# Goal
build DNN via Keras

# File Description
- `.ipynb` file is the solution of Week 2 program assignment 3
  - `Keras-task.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `Keras-task.html`
- `.py` python verson
  - `Keras-task.py`
- `.zip` of `md` file
  - `Keras-task.zip`
- file
  - `Keras-task`

# Snapshot
- **Recommend** read `.ipynb` file via [nbviewer](https://nbviewer.jupyter.org/)
- open `md` in file `Keras-task`
- open `.html` file via brower for quick look.

# Structure
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_5 (InputLayer)         (None, 28, 28)            0         
_________________________________________________________________
flatten_5 (Flatten)          (None, 784)               0         
_________________________________________________________________
dense_16 (Dense)             (None, 75)                58875     
_________________________________________________________________
activation_12 (Activation)   (None, 75)                0         
_________________________________________________________________
dense_17 (Dense)             (None, 125)               9500      
_________________________________________________________________
activation_13 (Activation)   (None, 125)               0         
_________________________________________________________________
dense_18 (Dense)             (None, 75)                9450      
_________________________________________________________________
activation_14 (Activation)   (None, 75)                0         
_________________________________________________________________
dense_19 (Dense)             (None, 10)                760       
=================================================================
Total params: 78,585
Trainable params: 78,585
Non-trainable params: 0
_________________________________________________________________
```


