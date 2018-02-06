# Your first CNN on CIFAR-10 via Keras
train/test -- 0.8/0.78

# Goal
- define your first CNN architecture for CIFAR-10 dataset
- train it from scratch
- visualize learnt filters

# File Description
- `.ipynb` file is the solution of Week 3 program assignment 1
  - `week3_task1_first_cnn_cifar10_clean.ipynb`
- `.html` file is the html version of `.ipynb` file.
  - `week3_task1_first_cnn_cifar10_clean.html`
- `.py` python verson
  - `week3_task1_first_cnn_cifar10_clean.py`
- `.zip` of `md` file
  - `week3_task1_first_cnn_cifar10_clean.zip`
- file
  - `week3_task1_first_cnn_cifar10_clean`
# Snapshot
- **Recommend** read `.ipynb` file via [nbviewer](https://nbviewer.jupyter.org/)
- open `md` in file `week3_task1_first_cnn_cifar10_clean`
- open `.html` file via brower for quick look.

# What you've done
- defined CNN architecture
- trained your model
- evaluated your model
- visualised learnt filters

# Architecture
```
  Layer (type)                 Output Shape              Param #   
  =================================================================
  conv1 (Conv2D)               (None, 32, 32, 16)        448       
  _________________________________________________________________
  leaky_re_lu_1 (LeakyReLU)    (None, 32, 32, 16)        0         
  _________________________________________________________________
  conv2 (Conv2D)               (None, 32, 32, 32)        4640      
  _________________________________________________________________
  leaky_re_lu_2 (LeakyReLU)    (None, 32, 32, 32)        0         
  _________________________________________________________________
  max_pool_1 (MaxPooling2D)    (None, 16, 16, 32)        0         
  _________________________________________________________________
  dropout_1 (Dropout)          (None, 16, 16, 32)        0         
  _________________________________________________________________
  conv3 (Conv2D)               (None, 16, 16, 32)        9248      
  _________________________________________________________________
  leaky_re_lu_3 (LeakyReLU)    (None, 16, 16, 32)        0         
  _________________________________________________________________
  conv4 (Conv2D)               (None, 16, 16, 64)        18496     
  _________________________________________________________________
  leaky_re_lu_4 (LeakyReLU)    (None, 16, 16, 64)        0         
  _________________________________________________________________
  max_pool_2 (MaxPooling2D)    (None, 8, 8, 64)          0         
  _________________________________________________________________
  dropout_2 (Dropout)          (None, 8, 8, 64)          0         
  _________________________________________________________________
  flatten_1 (Flatten)          (None, 4096)              0         
  _________________________________________________________________
  fc1 (Dense)                  (None, 256)               1048832   
  _________________________________________________________________
  dropout_3 (Dropout)          (None, 256)               0         
  _________________________________________________________________
  dense_1 (Dense)              (None, 10)                2570      
  _________________________________________________________________
  activation_1 (Activation)    (None, 10)                0         
  =================================================================
  Total params: 1,084,234
  Trainable params: 1,084,234
  Non-trainable params: 0
```
