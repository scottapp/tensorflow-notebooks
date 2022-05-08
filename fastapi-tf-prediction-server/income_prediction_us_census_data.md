# Income Bracket Prediction Based On US Census Dataset


```python
import os
from six.moves import urllib
import tempfile

import numpy as np
import pandas as pd
import tensorflow as tf

print(__import__('sys').version)
print(tf.__version__)
print(tf.keras.__version__)

```

    3.9.10 | packaged by conda-forge | (main, Feb  1 2022, 21:24:11) 
    [GCC 9.4.0]
    2.6.2
    2.6.0



```python
DATA_DIR = os.path.join(tempfile.gettempdir(), 'census_data')

# Download options.
DATA_URL = 'https://storage.googleapis.com/cloud-samples-data/ai-platform/census/data'
TRAINING_FILE = 'adult.data.csv'
EVAL_FILE = 'adult.test.csv'
TRAINING_URL = '%s/%s' % (DATA_URL, TRAINING_FILE)
EVAL_URL = '%s/%s' % (DATA_URL, EVAL_FILE)

### For interpreting data ###

# These are the features in the dataset.
# Dataset information: https://archive.ics.uci.edu/ml/datasets/census+income
_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CATEGORICAL_TYPES = {
  'workclass': pd.api.types.CategoricalDtype(categories=[
    'Federal-gov', 'Local-gov', 'Never-worked', 'Private', 'Self-emp-inc',
    'Self-emp-not-inc', 'State-gov', 'Without-pay'
  ]),
  'marital_status': pd.api.types.CategoricalDtype(categories=[
    'Divorced', 'Married-AF-spouse', 'Married-civ-spouse',
    'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'
  ]),
  'occupation': pd.api.types.CategoricalDtype([
    'Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial',
    'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct',
    'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv',
    'Sales', 'Tech-support', 'Transport-moving'
  ]),
  'relationship': pd.api.types.CategoricalDtype(categories=[
    'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried',
    'Wife'
  ]),
  'race': pd.api.types.CategoricalDtype(categories=[
    'Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'
  ]),
  'native_country': pd.api.types.CategoricalDtype(categories=[
    'Cambodia', 'Canada', 'China', 'Columbia', 'Cuba', 'Dominican-Republic',
    'Ecuador', 'El-Salvador', 'England', 'France', 'Germany', 'Greece',
    'Guatemala', 'Haiti', 'Holand-Netherlands', 'Honduras', 'Hong', 'Hungary',
    'India', 'Iran', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Laos', 'Mexico',
    'Nicaragua', 'Outlying-US(Guam-USVI-etc)', 'Peru', 'Philippines', 'Poland',
    'Portugal', 'Puerto-Rico', 'Scotland', 'South', 'Taiwan', 'Thailand',
    'Trinadad&Tobago', 'United-States', 'Vietnam', 'Yugoslavia'
  ]),
  'income_bracket': pd.api.types.CategoricalDtype(categories=[
    '<=50K', '>50K'
  ])
}

# This is the label (target) we want to predict.
_LABEL_COLUMN = 'income_bracket'

### Hyperparameters for training ###

# This the training batch size
BATCH_SIZE = 128

# This is the number of epochs (passes over the full training data)
NUM_EPOCHS = 20

# Define learning rate.
LEARNING_RATE = .01

```


```python
def _download_and_clean_file(filename, url):
  """Downloads data from url, and makes changes to match the CSV format.

  The CSVs may use spaces after the comma delimters (non-standard) or include
  rows which do not represent well-formed examples. This function strips out
  some of these problems.

  Args:
    filename: filename to save url to
    url: URL of resource to download
  """
  temp_file, _ = urllib.request.urlretrieve(url)
  with tf.io.gfile.GFile(temp_file, 'r') as temp_file_object:
    with tf.io.gfile.GFile(filename, 'w') as file_object:
      for line in temp_file_object:
        line = line.strip()
        line = line.replace(', ', ',')
        if not line or ',' not in line:
          continue
        if line[-1] == '.':
          line = line[:-1]
        line += '\n'
        file_object.write(line)
  tf.io.gfile.remove(temp_file)


def download(data_dir):
  """Downloads census data if it is not already present.

  Args:
    data_dir: directory where we will access/save the census data
  """
  tf.io.gfile.makedirs(data_dir)

  training_file_path = os.path.join(data_dir, TRAINING_FILE)
  if not tf.io.gfile.exists(training_file_path):
    _download_and_clean_file(training_file_path, TRAINING_URL)

  eval_file_path = os.path.join(data_dir, EVAL_FILE)
  if not tf.io.gfile.exists(eval_file_path):
    _download_and_clean_file(eval_file_path, EVAL_URL)

  return training_file_path, eval_file_path

```


```python
training_file_path, eval_file_path = download(DATA_DIR)
```


```python
!cp /tmp/census_data/adult.data.csv data/adult.data.csv
!cp /tmp/census_data/adult.test.csv data/adult.test.csv
```


```python
print(training_file_path)
print(eval_file_path)
```

    /tmp/census_data/adult.data.csv
    /tmp/census_data/adult.test.csv



```python
train_df = pd.read_csv(training_file_path, names=_CSV_COLUMNS, na_values='?')
eval_df = pd.read_csv(eval_file_path, names=_CSV_COLUMNS, na_values='?')
```


```python
train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education_num</th>
      <th>marital_status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>gender</th>
      <th>capital_gain</th>
      <th>capital_loss</th>
      <th>hours_per_week</th>
      <th>native_country</th>
      <th>income_bracket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>




```python
UNUSED_COLUMNS = ['fnlwgt', 'education', 'gender']


def preprocess(dataframe):
  """Converts categorical features to numeric. Removes unused columns.

  Args:
    dataframe: Pandas dataframe with raw data

  Returns:
    Dataframe with preprocessed data
  """
  dataframe = dataframe.drop(columns=UNUSED_COLUMNS)

  # Convert integer valued (numeric) columns to floating point
  numeric_columns = dataframe.select_dtypes(['int64']).columns
  dataframe[numeric_columns] = dataframe[numeric_columns].astype('float32')

  # Convert categorical columns to numeric
  cat_columns = dataframe.select_dtypes(['object']).columns
  dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.astype(_CATEGORICAL_TYPES[x.name]))
  dataframe[cat_columns] = dataframe[cat_columns].apply(lambda x: x.cat.codes)
  return dataframe

prepped_train_df = preprocess(train_df)
prepped_eval_df = preprocess(eval_df)

```


```python
prepped_train_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>education_num</th>
      <th>marital_status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>capital_gain</th>
      <th>capital_loss</th>
      <th>hours_per_week</th>
      <th>native_country</th>
      <th>income_bracket</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39.0</td>
      <td>6</td>
      <td>13.0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>2174.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50.0</td>
      <td>5</td>
      <td>13.0</td>
      <td>2</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38.0</td>
      <td>3</td>
      <td>9.0</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53.0</td>
      <td>3</td>
      <td>7.0</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28.0</td>
      <td>3</td>
      <td>13.0</td>
      <td>2</td>
      <td>9</td>
      <td>5</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>40.0</td>
      <td>4</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Split train and test data with labels.
# The pop() method will extract (copy) and remove the label column from the dataframe
train_x, train_y = prepped_train_df, prepped_train_df.pop(_LABEL_COLUMN)
eval_x, eval_y = prepped_eval_df, prepped_eval_df.pop(_LABEL_COLUMN)

# Reshape label columns for use with tf.data.Dataset
train_y = np.asarray(train_y).astype('float32').reshape((-1, 1))
eval_y = np.asarray(eval_y).astype('float32').reshape((-1, 1))

```


```python
def standardize(dataframe):
  """Scales numerical columns using their means and standard deviation to get
  z-scores: the mean of each numerical column becomes 0, and the standard
  deviation becomes 1. This can help the model converge during training.

  Args:
    dataframe: Pandas dataframe

  Returns:
    Input dataframe with the numerical columns scaled to z-scores
  """
  dtypes = list(zip(dataframe.dtypes.index, map(str, dataframe.dtypes)))
  # Normalize numeric columns.
  for column, dtype in dtypes:
      if dtype == 'float32':
          dataframe[column] -= dataframe[column].mean()
          dataframe[column] /= dataframe[column].std()
  return dataframe


# Join train_x and eval_x to normalize on overall means and standard
# deviations. Then separate them again.
all_x = pd.concat([train_x, eval_x], keys=['train', 'eval'])
all_x = standardize(all_x)
train_x, eval_x = all_x.xs('train'), all_x.xs('eval')

```


```python
def input_fn(features, labels, shuffle, num_epochs, batch_size):
  """Generates an input function to be used for model training.

  Args:
    features: numpy array of features used for training or inference
    labels: numpy array of labels for each example
    shuffle: boolean for whether to shuffle the data or not (set True for
      training, False for evaluation)
    num_epochs: number of epochs to provide the data for
    batch_size: batch size for training

  Returns:
    A tf.data.Dataset that can provide data to the Keras model for training or
      evaluation
  """
  if labels is None:
    inputs = features
  else:
    inputs = (features, labels)
  dataset = tf.data.Dataset.from_tensor_slices(inputs)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=len(features))

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset

```


```python
# Pass a numpy array by using DataFrame.values
training_dataset = input_fn(features=train_x.values,
                    labels=train_y,
                    shuffle=True,
                    num_epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE)

num_eval_examples = eval_x.shape[0]

# Pass a numpy array by using DataFrame.values
validation_dataset = input_fn(features=eval_x.values,
                    labels=eval_y,
                    shuffle=False,
                    num_epochs=NUM_EPOCHS,
                    batch_size=num_eval_examples)

```

    2022-05-07 10:00:27.834956: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2022-05-07 10:00:27.835047: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
    2022-05-07 10:00:27.835083: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (10b118e40db8): /proc/driver/nvidia/version does not exist
    2022-05-07 10:00:27.840960: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 AVX512F FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.



```python
def create_keras_model(input_dim, learning_rate):
  """Creates Keras Model for Binary Classification.

  Args:
    input_dim: How many features the input has
    learning_rate: Learning rate for training

  Returns:
    The compiled Keras model (still needs to be trained)
  """
  Dense = tf.keras.layers.Dense
  model = tf.keras.Sequential(
    [
        Dense(100, activation=tf.nn.relu, kernel_initializer='uniform', input_shape=(input_dim,)),
        Dense(75, activation=tf.nn.relu),
        Dense(50, activation=tf.nn.relu),
        Dense(25, activation=tf.nn.relu),
        Dense(1, activation=tf.nn.sigmoid)
    ])

  # Custom Optimizer:
  # https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer
  optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)

  # Compile Keras model
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model

```


```python
num_train_examples, input_dim = train_x.shape
print('Number of features: {}'.format(input_dim))
print('Number of examples: {}'.format(num_train_examples))

keras_model = create_keras_model(input_dim=input_dim, learning_rate=LEARNING_RATE)
```

    Number of features: 11
    Number of examples: 32561


    /opt/conda/lib/python3.9/site-packages/keras/optimizer_v2/optimizer_v2.py:355: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      warnings.warn(



```python
keras_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 100)               1200      
    _________________________________________________________________
    dense_1 (Dense)              (None, 75)                7575      
    _________________________________________________________________
    dense_2 (Dense)              (None, 50)                3800      
    _________________________________________________________________
    dense_3 (Dense)              (None, 25)                1275      
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 26        
    =================================================================
    Total params: 13,876
    Trainable params: 13,876
    Non-trainable params: 0
    _________________________________________________________________



```python
# Setup Learning Rate decay.
lr_decay_cb = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: LEARNING_RATE + 0.02 * (0.5 ** (1 + epoch)),
    verbose=True)
```


```python
history = keras_model.fit(training_dataset,
                          epochs=NUM_EPOCHS,
                          steps_per_epoch=int(num_train_examples/BATCH_SIZE),
                          validation_data=validation_dataset,
                          validation_steps=1,
                          callbacks=[lr_decay_cb],
                          verbose=1)
```

    Epoch 1/20
    
    Epoch 00001: LearningRateScheduler setting learning rate to 0.02.


    2022-05-07 10:02:20.801160: I tensorflow/compiler/mlir/mlir_graph_optimization_pass.cc:185] None of the MLIR Optimization Passes are enabled (registered 2)


    254/254 [==============================] - 1s 3ms/step - loss: 0.5875 - accuracy: 0.7865 - val_loss: 0.3878 - val_accuracy: 0.8085
    Epoch 2/20
    
    Epoch 00002: LearningRateScheduler setting learning rate to 0.015.
    254/254 [==============================] - 1s 3ms/step - loss: 0.3603 - accuracy: 0.8339 - val_loss: 0.3703 - val_accuracy: 0.8308
    Epoch 3/20
    
    Epoch 00003: LearningRateScheduler setting learning rate to 0.0125.
    254/254 [==============================] - 1s 3ms/step - loss: 0.3418 - accuracy: 0.8414 - val_loss: 0.3361 - val_accuracy: 0.8462
    Epoch 4/20
    
    Epoch 00004: LearningRateScheduler setting learning rate to 0.01125.
    254/254 [==============================] - 1s 3ms/step - loss: 0.3349 - accuracy: 0.8451 - val_loss: 0.3383 - val_accuracy: 0.8484
    Epoch 5/20
    
    Epoch 00005: LearningRateScheduler setting learning rate to 0.010625.
    254/254 [==============================] - 1s 3ms/step - loss: 0.3327 - accuracy: 0.8469 - val_loss: 0.3264 - val_accuracy: 0.8517
    Epoch 6/20
    
    Epoch 00006: LearningRateScheduler setting learning rate to 0.0103125.
    254/254 [==============================] - 1s 2ms/step - loss: 0.3308 - accuracy: 0.8486 - val_loss: 0.3255 - val_accuracy: 0.8480
    Epoch 7/20
    
    Epoch 00007: LearningRateScheduler setting learning rate to 0.01015625.
    254/254 [==============================] - 1s 3ms/step - loss: 0.3280 - accuracy: 0.8480 - val_loss: 0.3217 - val_accuracy: 0.8519
    Epoch 8/20
    
    Epoch 00008: LearningRateScheduler setting learning rate to 0.010078125.
    254/254 [==============================] - 1s 2ms/step - loss: 0.3269 - accuracy: 0.8485 - val_loss: 0.3261 - val_accuracy: 0.8522
    Epoch 9/20
    
    Epoch 00009: LearningRateScheduler setting learning rate to 0.0100390625.
    254/254 [==============================] - 1s 2ms/step - loss: 0.3249 - accuracy: 0.8492 - val_loss: 0.3221 - val_accuracy: 0.8520
    Epoch 10/20
    
    Epoch 00010: LearningRateScheduler setting learning rate to 0.01001953125.
    254/254 [==============================] - 1s 2ms/step - loss: 0.3253 - accuracy: 0.8500 - val_loss: 0.3199 - val_accuracy: 0.8511
    Epoch 11/20
    
    Epoch 00011: LearningRateScheduler setting learning rate to 0.010009765625.
    254/254 [==============================] - 1s 2ms/step - loss: 0.3243 - accuracy: 0.8488 - val_loss: 0.3233 - val_accuracy: 0.8470
    Epoch 12/20
    
    Epoch 00012: LearningRateScheduler setting learning rate to 0.010004882812500001.
    254/254 [==============================] - 1s 3ms/step - loss: 0.3236 - accuracy: 0.8485 - val_loss: 0.3293 - val_accuracy: 0.8514
    Epoch 13/20
    
    Epoch 00013: LearningRateScheduler setting learning rate to 0.01000244140625.
    254/254 [==============================] - 1s 3ms/step - loss: 0.3243 - accuracy: 0.8505 - val_loss: 0.3251 - val_accuracy: 0.8468
    Epoch 14/20
    
    Epoch 00014: LearningRateScheduler setting learning rate to 0.010001220703125.
    254/254 [==============================] - 1s 3ms/step - loss: 0.3231 - accuracy: 0.8488 - val_loss: 0.3282 - val_accuracy: 0.8515
    Epoch 15/20
    
    Epoch 00015: LearningRateScheduler setting learning rate to 0.0100006103515625.
    254/254 [==============================] - 1s 3ms/step - loss: 0.3241 - accuracy: 0.8518 - val_loss: 0.3217 - val_accuracy: 0.8511
    Epoch 16/20
    
    Epoch 00016: LearningRateScheduler setting learning rate to 0.01000030517578125.
    254/254 [==============================] - 1s 3ms/step - loss: 0.3222 - accuracy: 0.8505 - val_loss: 0.3304 - val_accuracy: 0.8481
    Epoch 17/20
    
    Epoch 00017: LearningRateScheduler setting learning rate to 0.010000152587890625.
    254/254 [==============================] - 1s 3ms/step - loss: 0.3236 - accuracy: 0.8490 - val_loss: 0.3321 - val_accuracy: 0.8420
    Epoch 18/20
    
    Epoch 00018: LearningRateScheduler setting learning rate to 0.010000076293945313.
    254/254 [==============================] - 1s 3ms/step - loss: 0.3244 - accuracy: 0.8493 - val_loss: 0.3219 - val_accuracy: 0.8523
    Epoch 19/20
    
    Epoch 00019: LearningRateScheduler setting learning rate to 0.010000038146972657.
    254/254 [==============================] - 1s 2ms/step - loss: 0.3220 - accuracy: 0.8509 - val_loss: 0.3221 - val_accuracy: 0.8532
    Epoch 20/20
    
    Epoch 00020: LearningRateScheduler setting learning rate to 0.010000019073486329.
    254/254 [==============================] - 1s 2ms/step - loss: 0.3235 - accuracy: 0.8503 - val_loss: 0.3366 - val_accuracy: 0.8425



```python
tf.keras.models.save_model(keras_model, 'saved_model/keras_export')
```

    INFO:tensorflow:Assets written to: saved_model/keras_export/assets
    Model exported to:  None



```python

```
