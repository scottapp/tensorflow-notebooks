## CNN classification using the sign mnist dataset with data augmentation

The dataset can be downloaded at https://www.kaggle.com/datasets/datamunge/sign-language-mnist

## Load dependent libraries


```python
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd
```


```python
tf.__version__
```




    '2.4.1'



## Preprocess data


```python
def get_data(filename):
    with open(filename) as training_file:
        reader = csv.reader(training_file, delimiter=',')
        temp_images = []
        temp_labels = []
        next(reader)
        for row in reader:
            temp_labels.append(row[0])
            image_data = row[1:785]
            image_data_as_array = np.array_split(image_data, 28)
            temp_images.append(image_data_as_array)                
        images = np.array(temp_images).astype('float')
        labels = np.array(temp_labels).astype('float')
    return images, labels


training_images, training_labels = get_data('data/sign_mnist_train.csv')
testing_images, testing_labels = get_data('data/sign_mnist_test.csv')
```

## Some info about the dataset


```python
print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

num_classes = None
num_classes = len(np.unique(training_labels))
print(np.unique(training_labels))
print(num_classes)
```

    (27455, 28, 28)
    (27455,)
    (7172, 28, 28)
    (7172,)
    [ 0.  1.  2.  3.  4.  5.  6.  7.  8. 10. 11. 12. 13. 14. 15. 16. 17. 18.
     19. 20. 21. 22. 23. 24.]
    24


## Take a look the images


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

i = 0
for img in training_images[0:25]:
    sp = plt.subplot(nrows, ncols, i + 1)
    i += 1
    sp.axis('Off')
    #img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()
```


    
![png](output_9_0.png)
    


## Initialize ImageDataGenerator for training


```python
training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

print(training_images.shape)
print(testing_images.shape)
```

    (27455, 28, 28, 1)
    (7172, 28, 28, 1)


## Build Model


```python
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(26, activation=tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 26, 26, 64)        640       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 11, 11, 64)        36928     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
    _________________________________________________________________
    flatten (Flatten)            (None, 1600)              0         
    _________________________________________________________________
    dense (Dense)                (None, 128)               204928    
    _________________________________________________________________
    dense_1 (Dense)              (None, 26)                3354      
    =================================================================
    Total params: 245,850
    Trainable params: 245,850
    Non-trainable params: 0
    _________________________________________________________________


    2022-04-09 03:23:46.768271: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


## Fit Model


```python
history = model.fit(train_datagen.flow(training_images, training_labels, batch_size=32),
                    epochs=15,
                    validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=32))
                    
model.evaluate(testing_images, testing_labels)
```

    2022-04-09 03:24:03.196655: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:116] None of the MLIR optimization passes are enabled (registered 2)
    2022-04-09 03:24:03.197101: I tensorflow/core/platform/profile_utils/cpu_utils.cc:112] CPU Frequency: 1497600000 Hz


    Epoch 1/15
    858/858 [==============================] - 49s 56ms/step - loss: 3.0538 - accuracy: 0.0891 - val_loss: 2.0495 - val_accuracy: 0.3113
    Epoch 2/15
    858/858 [==============================] - 49s 58ms/step - loss: 2.2542 - accuracy: 0.2918 - val_loss: 1.4040 - val_accuracy: 0.5046
    Epoch 3/15
    858/858 [==============================] - 39s 45ms/step - loss: 1.8520 - accuracy: 0.4068 - val_loss: 1.0282 - val_accuracy: 0.6433
    Epoch 4/15
    858/858 [==============================] - 57s 66ms/step - loss: 1.6010 - accuracy: 0.4860 - val_loss: 0.9982 - val_accuracy: 0.6571
    Epoch 5/15
    858/858 [==============================] - 43s 51ms/step - loss: 1.4231 - accuracy: 0.5352 - val_loss: 0.7287 - val_accuracy: 0.7526
    Epoch 6/15
    858/858 [==============================] - 34s 40ms/step - loss: 1.2719 - accuracy: 0.5882 - val_loss: 0.7194 - val_accuracy: 0.7487
    Epoch 7/15
    858/858 [==============================] - 34s 40ms/step - loss: 1.1688 - accuracy: 0.6147 - val_loss: 0.6685 - val_accuracy: 0.7769
    Epoch 8/15
    858/858 [==============================] - 35s 41ms/step - loss: 1.0967 - accuracy: 0.6388 - val_loss: 0.6164 - val_accuracy: 0.7981
    Epoch 9/15
    858/858 [==============================] - 33s 39ms/step - loss: 1.0215 - accuracy: 0.6632 - val_loss: 0.5344 - val_accuracy: 0.8122
    Epoch 10/15
    858/858 [==============================] - 37s 43ms/step - loss: 0.9729 - accuracy: 0.6736 - val_loss: 0.4464 - val_accuracy: 0.8572
    Epoch 11/15
    858/858 [==============================] - 27s 32ms/step - loss: 0.8971 - accuracy: 0.6960 - val_loss: 0.4612 - val_accuracy: 0.8470
    Epoch 12/15
    858/858 [==============================] - 25s 29ms/step - loss: 0.8674 - accuracy: 0.7087 - val_loss: 0.4954 - val_accuracy: 0.8348
    Epoch 13/15
    858/858 [==============================] - 24s 28ms/step - loss: 0.8316 - accuracy: 0.7253 - val_loss: 0.3905 - val_accuracy: 0.8581
    Epoch 14/15
    858/858 [==============================] - 24s 28ms/step - loss: 0.7818 - accuracy: 0.7375 - val_loss: 0.4901 - val_accuracy: 0.8272
    Epoch 15/15
    858/858 [==============================] - 35s 41ms/step - loss: 0.7627 - accuracy: 0.7466 - val_loss: 0.3578 - val_accuracy: 0.8738
    225/225 [==============================] - 2s 9ms/step - loss: 216.8945 - accuracy: 0.6139





    [216.8944854736328, 0.613915205001831]




```python
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
```


    
![png](output_16_0.png)
    



    
![png](output_16_1.png)
    



```python

```
