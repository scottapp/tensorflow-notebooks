## CNN classification using the cats and dogs dataset with data augmentation

This notebook is originally from a lesson on the Coursera website.  All rights belong to the original authors

## Load dependent libraries


```python
import os
import zipfile
import random
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
```

## Download dataset


```python
!wget --no-check-certificate \
     "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip" \
-O "/tmp/cats-and-dogs.zip"
```

    --2022-04-07 09:09:47--  https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
    Resolving download.microsoft.com (download.microsoft.com)... 173.222.180.188, 2600:1417:1800:285::e59, 2600:1417:1800:298::e59, ...
    Connecting to download.microsoft.com (download.microsoft.com)|173.222.180.188|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 824894548 (787M) [application/octet-stream]
    Saving to: ‘/tmp/cats-and-dogs.zip’
    
    /tmp/cats-and-dogs. 100%[===================>] 786.68M  6.47MB/s    in 2m 21s  
    
    2022-04-07 09:12:09 (5.56 MB/s) - ‘/tmp/cats-and-dogs.zip’ saved [824894548/824894548]
    



```python
local_zip = '/tmp/cats-and-dogs.zip'
zip_ref   = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('./tmp')
zip_ref.close()
```


```python
print(len(os.listdir('./tmp/PetImages/Cat/')))
print(len(os.listdir('./tmp/PetImages/Dog/')))
```

    12501
    12501



```python
try:
    os.mkdir('./tmp/cats-v-dogs')
    os.mkdir('./tmp/cats-v-dogs/training')
    os.mkdir('./tmp/cats-v-dogs/testing')
    os.mkdir('./tmp/cats-v-dogs/training/cats')
    os.mkdir('./tmp/cats-v-dogs/training/dogs')
    os.mkdir('./tmp/cats-v-dogs/testing/cats')
    os.mkdir('./tmp/cats-v-dogs/testing/dogs')
except OSError:
    pass
```

## Preprocess data


```python
def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = SOURCE + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    testing_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    testing_set = shuffled_set[:testing_length]

    for filename in training_set:
        this_file = SOURCE + filename
        destination = TRAINING + filename
        copyfile(this_file, destination)

    for filename in testing_set:
        this_file = SOURCE + filename
        destination = TESTING + filename
        copyfile(this_file, destination)


CAT_SOURCE_DIR = "./tmp/PetImages/Cat/"
TRAINING_CATS_DIR = "./tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "./tmp/cats-v-dogs/testing/cats/"

DOG_SOURCE_DIR = "./tmp/PetImages/Dog/"
TRAINING_DOGS_DIR = "./tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "./tmp/cats-v-dogs/testing/dogs/"

split_size = .9
split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
```

    666.jpg is zero length, so ignoring.
    11702.jpg is zero length, so ignoring.


## Display some images of cats and dogs


```python
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameters for our graph; we'll output images in a 5x5 configuration
nrows = 5
ncols = 5

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 5x5 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 5, nrows * 5)

cat_fnames = os.listdir('./tmp/cats-v-dogs/training/cats/')
dog_fnames = os.listdir('./tmp/cats-v-dogs/training/dogs/')

fnames = ['%s/%s' % ('./tmp/cats-v-dogs/training/cats/', x) for x in cat_fnames[0:15]]
fnames = fnames + ['%s/%s' % ('./tmp/cats-v-dogs/training/dogs/', x) for x in dog_fnames[0:10]]

i = 0
for fname in fnames:
    sp = plt.subplot(nrows, ncols, i + 1)
    i += 1
    sp.axis('Off')
    img = mpimg.imread(fname)
    plt.imshow(img)

plt.show()
print(img.shape)
```


    
![png](output_12_0.png)
    


    (500, 429, 3)



```python
print(len(os.listdir('./tmp/cats-v-dogs/training/cats/')))
print(len(os.listdir('./tmp/cats-v-dogs/training/dogs/')))
print(len(os.listdir('./tmp/cats-v-dogs/testing/cats/')))
print(len(os.listdir('./tmp/cats-v-dogs/testing/dogs/')))
```

    11250
    11250
    1250
    1250


## Build and compile model


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 148, 148, 16)      448       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 74, 74, 16)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 72, 72, 32)        4640      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 36, 36, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 34, 34, 64)        18496     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 17, 17, 64)        0         
    _________________________________________________________________
    flatten (Flatten)            (None, 18496)             0         
    _________________________________________________________________
    dense (Dense)                (None, 512)               9470464   
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 513       
    =================================================================
    Total params: 9,494,561
    Trainable params: 9,494,561
    Non-trainable params: 0
    _________________________________________________________________


## Init the ImageDataGenerator for traning and validation


```python
TRAINING_DIR = "./tmp/cats-v-dogs/training/"

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

VALIDATION_DIR = "./tmp/cats-v-dogs/testing/"

validation_datagen = ImageDataGenerator(rescale=1./255,
                                        rotation_range=40,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        shear_range=0.2,
                                        zoom_range=0.2,
                                        horizontal_flip=True,
                                        fill_mode='nearest')

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=100,
                                                              class_mode='binary',
                                                              target_size=(150, 150))
```

    Found 22498 images belonging to 2 classes.
    Found 2499 images belonging to 2 classes.


## Fit the model


```python
history = model.fit(train_generator,
                    epochs=15,
                    verbose=1,
                    validation_data=validation_generator)
```

    Epoch 1/15
    132/225 [================>.............] - ETA: 4:08 - loss: 1.5070 - accuracy: 0.5298

    /opt/conda/lib/python3.9/site-packages/PIL/TiffImagePlugin.py:793: UserWarning: Truncated File Read
      warnings.warn(str(msg))


    225/225 [==============================] - 664s 3s/step - loss: 1.2241 - accuracy: 0.5437 - val_loss: 0.6564 - val_accuracy: 0.6086
    Epoch 2/15
    225/225 [==============================] - 561s 2s/step - loss: 0.6387 - accuracy: 0.6402 - val_loss: 0.5658 - val_accuracy: 0.7135
    Epoch 3/15
    225/225 [==============================] - 656s 3s/step - loss: 0.5902 - accuracy: 0.6864 - val_loss: 0.5288 - val_accuracy: 0.7411
    Epoch 4/15
    225/225 [==============================] - 647s 3s/step - loss: 0.5691 - accuracy: 0.7056 - val_loss: 0.5276 - val_accuracy: 0.7223
    Epoch 5/15
    225/225 [==============================] - 638s 3s/step - loss: 0.5337 - accuracy: 0.7311 - val_loss: 0.4886 - val_accuracy: 0.7643
    Epoch 6/15
    225/225 [==============================] - 690s 3s/step - loss: 0.5219 - accuracy: 0.7395 - val_loss: 0.4728 - val_accuracy: 0.7823
    Epoch 7/15
    225/225 [==============================] - 564s 3s/step - loss: 0.4999 - accuracy: 0.7575 - val_loss: 0.4563 - val_accuracy: 0.7867
    Epoch 8/15
    225/225 [==============================] - 524s 2s/step - loss: 0.4980 - accuracy: 0.7633 - val_loss: 0.4749 - val_accuracy: 0.7767
    Epoch 9/15
    225/225 [==============================] - 518s 2s/step - loss: 0.4887 - accuracy: 0.7681 - val_loss: 0.4222 - val_accuracy: 0.8099
    Epoch 10/15
    225/225 [==============================] - 503s 2s/step - loss: 0.4737 - accuracy: 0.7738 - val_loss: 0.4516 - val_accuracy: 0.7859
    Epoch 11/15
    225/225 [==============================] - 503s 2s/step - loss: 0.4637 - accuracy: 0.7785 - val_loss: 0.4901 - val_accuracy: 0.7847
    Epoch 12/15
    225/225 [==============================] - 502s 2s/step - loss: 0.4577 - accuracy: 0.7854 - val_loss: 0.4040 - val_accuracy: 0.8259
    Epoch 13/15
    225/225 [==============================] - 501s 2s/step - loss: 0.4351 - accuracy: 0.7970 - val_loss: 0.4024 - val_accuracy: 0.8155
    Epoch 14/15
    225/225 [==============================] - 505s 2s/step - loss: 0.4377 - accuracy: 0.7978 - val_loss: 0.4452 - val_accuracy: 0.7815
    Epoch 15/15
    225/225 [==============================] - 509s 2s/step - loss: 0.4282 - accuracy: 0.8059 - val_loss: 0.3819 - val_accuracy: 0.8367



```python
%matplotlib inline

import matplotlib.image  as mpimg
import matplotlib.pyplot as plt

#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.figure()

#------------------------------------------------
# Plot training and validation loss per epoch
#------------------------------------------------
plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.figure()
```




    <Figure size 432x288 with 0 Axes>




    
![png](output_20_1.png)
    



    
![png](output_20_2.png)
    



    <Figure size 432x288 with 0 Axes>



```python

```
