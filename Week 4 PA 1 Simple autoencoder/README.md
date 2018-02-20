# Denoising Autoencoders And Where To Find Them

<img src="images/encoder.png" width=50% />

PCA MSE: 0.00665396321636
Convolutional autoencoder MSE: 0.00549629178011

human faces from the [lfw dataset](http://vis-www.cs.umass.edu/lfw/)

# Goal
train deep autoencoders and apply them to faces and similar images search.

# Relevant data links
- http://www.cs.columbia.edu/CAVE/databases/pubfig/download/lfw_attributes.txt
- http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
- http://vis-www.cs.umass.edu/lfw/lfw.tgz

# File Description
- `.ipynb` file is the solution of Week 4 program assignment 1
  - `Autoencoders-task_v2.ipynb`
- `.py` python verson
  - `Autoencoders-task_v2.py`
- file
  - `Autoencoders-task_v2`
  
  
# Snapshot
- **Recommend** read `.ipynb` file via [nbviewer](https://nbviewer.jupyter.org/)
- open `md` in file `Autoencoders-task_v2`

# Architecture

Encoder

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 32, 32, 3)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 16, 16, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 8, 128)         73856     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 4, 4, 128)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 4, 256)         295168    
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 2, 2, 256)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1024)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 32)                32800     
=================================================================
Total params: 421,216
Trainable params: 421,216
Non-trainable params: 0
_________________________________________________________________
```

Decoder
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         (None, 32)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 1024)              33792     
_________________________________________________________________
reshape_1 (Reshape)          (None, 2, 2, 256)         0         
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 4, 4, 128)         295040    
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 8, 8, 64)          73792     
_________________________________________________________________
conv2d_transpose_3 (Conv2DTr (None, 16, 16, 32)        18464     
_________________________________________________________________
conv2d_transpose_4 (Conv2DTr (None, 32, 32, 3)         867       
=================================================================
Total params: 421,955
Trainable params: 421,955
Non-trainable params: 0
_________________________________________________________________
```
