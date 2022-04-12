---
title: Training In Script Mode
---
This notebook demonstrates the process of using AWS Sagemaker in script mode to train a simple classification model.

The dataset I will be using is the fashion mnist dataset.  And the script will be run in local model without using a actual Sagemaker notebook instance.

### Import packages


```python
import sagemaker
import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist as mnist

# aws_env is my helper package that contains aws related settings
from aws_env import alias

sess = sagemaker.Session()
role = 'arn:aws:iam::%s:role/service-role/AmazonSageMaker-ExecutionRole-20210613T122808' % alias
```

### Download datasets


```python
(x_train, y_train), (x_val, y_val) = mnist.load_data()
fashion_mnist = tf.keras.datasets.fashion_mnist

os.makedirs("./data", exist_ok = True)

np.savez('./data/training', image=x_train, label=y_train)
np.savez('./data/validation', image=x_val, label=y_val)
```


```python
!pygmentize mnist-trainer.py
```

    import argparse, os
    import numpy as np
    
    import tensorflow
    from tensorflow.keras import backend as K
    from tensorflow.keras.optimizers import Adam
    
    
    class MyCallback(tensorflow.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('acc') > 0.998:
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True
    
    
    def create_model(lr):
        model = tensorflow.keras.models.Sequential([
            tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tensorflow.keras.layers.MaxPooling2D(2, 2),
            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dense(128, activation='relu'),
            tensorflow.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer=Adam(lr=lr),
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        return model
    
    
    if __name__ == '__main__':
    
        parser = argparse.ArgumentParser()
    
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--learning-rate', type=float, default=0.001)
        parser.add_argument('--batch-size', type=int, default=32)
        parser.add_argument('--gpu-count', type=int, default=os.environ['SM_NUM_GPUS'])
        parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
        parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
        parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    
        args, _ = parser.parse_known_args()
    
        epochs = args.epochs
        lr = args.learning_rate
        batch_size = args.batch_size
        gpu_count = args.gpu_count
        model_dir = args.model_dir
        training_dir = args.training
        validation_dir = args.validation
    
        mnist_train_images = np.load(os.path.join(training_dir, 'training.npz'))['image']
        mnist_train_labels = np.load(os.path.join(training_dir, 'training.npz'))['label']
        mnist_test_images  = np.load(os.path.join(validation_dir, 'validation.npz'))['image']
        mnist_test_labels  = np.load(os.path.join(validation_dir, 'validation.npz'))['label']
    
        K.set_image_data_format('channels_last')
        train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 28, 28, 1)
        test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)
    
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')
        train_images /= 255.0
        test_images /= 255.0
    
        train_labels = tensorflow.keras.utils.to_categorical(mnist_train_labels, 10)
        test_labels = tensorflow.keras.utils.to_categorical(mnist_test_labels, 10)
    
        callbacks = MyCallback()
        model = create_model(lr=lr)
        print(model.summary())
    
        model.fit(train_images,
                  train_labels,
                  batch_size=batch_size,
                  validation_data=(test_images, test_labels),
                  epochs=epochs,
                  verbose=1,
                  callbacks=[callbacks])
    
        score = model.evaluate(test_images, test_labels, verbose=0)
        print('Validation loss    :', score[0])
        print('Validation accuracy:', score[1])
    
        # save Keras model for Tensorflow Serving
        sess = K.get_session()
        tensorflow.saved_model.simple_save(
            sess,
            os.path.join(model_dir, 'model/1'),
            inputs={'inputs': model.input},
            outputs={t.name: t for t in model.outputs})
    

### Create the sagemaker estimator in local mode


```python
from sagemaker.tensorflow import TensorFlow

tf_estimator = TensorFlow(entry_point='mnist-trainer.py', 
                          role=role,
                          instance_count=1, 
                          instance_type='local',
                          framework_version="1.12", 
                          py_version='py3',
                          script_mode=True,
                          hyperparameters={'epochs': 1}
                         )
```

    Windows Support for Local Mode is Experimental
    


```python
training_input_path = 'file://./data/training'
validation_input_path = 'file://./data/validation'
tf_estimator.fit({'training': training_input_path, 'validation': validation_input_path})
```

    Creating 6hx3ay9l71-algo-1-zbxqn ... 
    Creating 6hx3ay9l71-algo-1-zbxqn ... done
    Docker Compose is now in the Docker CLI, try `docker compose up`
    
    Attaching to 6hx3ay9l71-algo-1-zbxqn
    [36m6hx3ay9l71-algo-1-zbxqn |[0m 2021-06-13 10:03:36,476 sagemaker-containers INFO     Imported framework sagemaker_tensorflow_container.training
    [36m6hx3ay9l71-algo-1-zbxqn |[0m 2021-06-13 10:03:36,488 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
    [36m6hx3ay9l71-algo-1-zbxqn |[0m 2021-06-13 10:03:37,170 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
    [36m6hx3ay9l71-algo-1-zbxqn |[0m 2021-06-13 10:03:37,188 sagemaker-containers INFO     No GPUs detected (normal if no gpus installed)
    [36m6hx3ay9l71-algo-1-zbxqn |[0m 2021-06-13 10:03:37,197 sagemaker-containers INFO     Invoking user script
    [36m6hx3ay9l71-algo-1-zbxqn |[0m 
    [36m6hx3ay9l71-algo-1-zbxqn |[0m Training Env:
    [36m6hx3ay9l71-algo-1-zbxqn |[0m 
    [36m6hx3ay9l71-algo-1-zbxqn |[0m {
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "additional_framework_parameters": {},
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "channel_input_dirs": {
    [36m6hx3ay9l71-algo-1-zbxqn |[0m         "training": "/opt/ml/input/data/training",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m         "validation": "/opt/ml/input/data/validation"
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     },
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "current_host": "algo-1-zbxqn",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "framework_module": "sagemaker_tensorflow_container.training:main",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "hosts": [
    [36m6hx3ay9l71-algo-1-zbxqn |[0m         "algo-1-zbxqn"
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     ],
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "hyperparameters": {
    [36m6hx3ay9l71-algo-1-zbxqn |[0m         "epochs": 1,
    [36m6hx3ay9l71-algo-1-zbxqn |[0m         "model_dir": "s3://sagemaker-ap-northeast-1-922656660811/sagemaker-tensorflow-scriptmode-2021-06-13-10-03-31-131/model"
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     },
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "input_config_dir": "/opt/ml/input/config",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "input_data_config": {
    [36m6hx3ay9l71-algo-1-zbxqn |[0m         "training": {
    [36m6hx3ay9l71-algo-1-zbxqn |[0m             "TrainingInputMode": "File"
    [36m6hx3ay9l71-algo-1-zbxqn |[0m         },
    [36m6hx3ay9l71-algo-1-zbxqn |[0m         "validation": {
    [36m6hx3ay9l71-algo-1-zbxqn |[0m             "TrainingInputMode": "File"
    [36m6hx3ay9l71-algo-1-zbxqn |[0m         }
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     },
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "input_dir": "/opt/ml/input",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "is_master": true,
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "job_name": "sagemaker-tensorflow-scriptmode-2021-06-13-10-03-31-131",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "log_level": 20,
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "master_hostname": "algo-1-zbxqn",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "model_dir": "/opt/ml/model",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "module_dir": "s3://sagemaker-ap-northeast-1-922656660811/sagemaker-tensorflow-scriptmode-2021-06-13-10-03-31-131/source/sourcedir.tar.gz",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "module_name": "mnist-trainer",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "network_interface_name": "eth0",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "num_cpus": 2,
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "num_gpus": 0,
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "output_data_dir": "/opt/ml/output/data",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "output_dir": "/opt/ml/output",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "output_intermediate_dir": "/opt/ml/output/intermediate",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "resource_config": {
    [36m6hx3ay9l71-algo-1-zbxqn |[0m         "current_host": "algo-1-zbxqn",
    [36m6hx3ay9l71-algo-1-zbxqn |[0m         "hosts": [
    [36m6hx3ay9l71-algo-1-zbxqn |[0m             "algo-1-zbxqn"
    [36m6hx3ay9l71-algo-1-zbxqn |[0m         ]
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     },
    [36m6hx3ay9l71-algo-1-zbxqn |[0m     "user_entry_point": "mnist-trainer.py"
    [36m6hx3ay9l71-algo-1-zbxqn |[0m }
    [36m6hx3ay9l71-algo-1-zbxqn |[0m 
    [36m6hx3ay9l71-algo-1-zbxqn |[0m Environment variables:
    [36m6hx3ay9l71-algo-1-zbxqn |[0m 
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_HOSTS=["algo-1-zbxqn"]
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_NETWORK_INTERFACE_NAME=eth0
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_HPS={"epochs":1,"model_dir":"s3://sagemaker-ap-northeast-1-922656660811/sagemaker-tensorflow-scriptmode-2021-06-13-10-03-31-131/model"}
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_USER_ENTRY_POINT=mnist-trainer.py
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_FRAMEWORK_PARAMS={}
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_RESOURCE_CONFIG={"current_host":"algo-1-zbxqn","hosts":["algo-1-zbxqn"]}
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_INPUT_DATA_CONFIG={"training":{"TrainingInputMode":"File"},"validation":{"TrainingInputMode":"File"}}
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_OUTPUT_DATA_DIR=/opt/ml/output/data
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_CHANNELS=["training","validation"]
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_CURRENT_HOST=algo-1-zbxqn
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_MODULE_NAME=mnist-trainer
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_LOG_LEVEL=20
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_FRAMEWORK_MODULE=sagemaker_tensorflow_container.training:main
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_INPUT_DIR=/opt/ml/input
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_INPUT_CONFIG_DIR=/opt/ml/input/config
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_OUTPUT_DIR=/opt/ml/output
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_NUM_CPUS=2
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_NUM_GPUS=0
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_MODEL_DIR=/opt/ml/model
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_MODULE_DIR=s3://sagemaker-ap-northeast-1-922656660811/sagemaker-tensorflow-scriptmode-2021-06-13-10-03-31-131/source/sourcedir.tar.gz
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{"training":"/opt/ml/input/data/training","validation":"/opt/ml/input/data/validation"},"current_host":"algo-1-zbxqn","framework_module":"sagemaker_tensorflow_container.training:main","hosts":["algo-1-zbxqn"],"hyperparameters":{"epochs":1,"model_dir":"s3://sagemaker-ap-northeast-1-922656660811/sagemaker-tensorflow-scriptmode-2021-06-13-10-03-31-131/model"},"input_config_dir":"/opt/ml/input/config","input_data_config":{"training":{"TrainingInputMode":"File"},"validation":{"TrainingInputMode":"File"}},"input_dir":"/opt/ml/input","is_master":true,"job_name":"sagemaker-tensorflow-scriptmode-2021-06-13-10-03-31-131","log_level":20,"master_hostname":"algo-1-zbxqn","model_dir":"/opt/ml/model","module_dir":"s3://sagemaker-ap-northeast-1-922656660811/sagemaker-tensorflow-scriptmode-2021-06-13-10-03-31-131/source/sourcedir.tar.gz","module_name":"mnist-trainer","network_interface_name":"eth0","num_cpus":2,"num_gpus":0,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_host":"algo-1-zbxqn","hosts":["algo-1-zbxqn"]},"user_entry_point":"mnist-trainer.py"}
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_USER_ARGS=["--epochs","1","--model_dir","s3://sagemaker-ap-northeast-1-922656660811/sagemaker-tensorflow-scriptmode-2021-06-13-10-03-31-131/model"]
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_CHANNEL_TRAINING=/opt/ml/input/data/training
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_CHANNEL_VALIDATION=/opt/ml/input/data/validation
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_HP_EPOCHS=1
    [36m6hx3ay9l71-algo-1-zbxqn |[0m SM_HP_MODEL_DIR=s3://sagemaker-ap-northeast-1-922656660811/sagemaker-tensorflow-scriptmode-2021-06-13-10-03-31-131/model
    [36m6hx3ay9l71-algo-1-zbxqn |[0m PYTHONPATH=/opt/ml/code:/usr/local/bin:/usr/lib/python36.zip:/usr/lib/python3.6:/usr/lib/python3.6/lib-dynload:/usr/local/lib/python3.6/dist-packages:/usr/lib/python3/dist-packages
    [36m6hx3ay9l71-algo-1-zbxqn |[0m 
    [36m6hx3ay9l71-algo-1-zbxqn |[0m Invoking script with the following command:
    [36m6hx3ay9l71-algo-1-zbxqn |[0m 
    [36m6hx3ay9l71-algo-1-zbxqn |[0m /usr/bin/python mnist-trainer.py --epochs 1 --model_dir s3://sagemaker-ap-northeast-1-922656660811/sagemaker-tensorflow-scriptmode-2021-06-13-10-03-31-131/model
    [36m6hx3ay9l71-algo-1-zbxqn |[0m 
    [36m6hx3ay9l71-algo-1-zbxqn |[0m 
    [36m6hx3ay9l71-algo-1-zbxqn |[0m _________________________________________________________________
    [36m6hx3ay9l71-algo-1-zbxqn |[0m Layer (type)                 Output Shape              Param #   
    [36m6hx3ay9l71-algo-1-zbxqn |[0m =================================================================
    [36m6hx3ay9l71-algo-1-zbxqn |[0m conv2d (Conv2D)              (None, 26, 26, 32)        320       
    [36m6hx3ay9l71-algo-1-zbxqn |[0m _________________________________________________________________
    [36m6hx3ay9l71-algo-1-zbxqn |[0m max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
    [36m6hx3ay9l71-algo-1-zbxqn |[0m _________________________________________________________________
    [36m6hx3ay9l71-algo-1-zbxqn |[0m flatten (Flatten)            (None, 5408)              0         
    [36m6hx3ay9l71-algo-1-zbxqn |[0m _________________________________________________________________
    [36m6hx3ay9l71-algo-1-zbxqn |[0m dense (Dense)                (None, 128)               692352    
    [36m6hx3ay9l71-algo-1-zbxqn |[0m _________________________________________________________________
    [36m6hx3ay9l71-algo-1-zbxqn |[0m dense_1 (Dense)              (None, 10)                1290      
    [36m6hx3ay9l71-algo-1-zbxqn |[0m =================================================================
    [36m6hx3ay9l71-algo-1-zbxqn |[0m Total params: 693,962
    [36m6hx3ay9l71-algo-1-zbxqn |[0m Trainable params: 693,962
    [36m6hx3ay9l71-algo-1-zbxqn |[0m Non-trainable params: 0
    [36m6hx3ay9l71-algo-1-zbxqn |[0m _________________________________________________________________
    [36m6hx3ay9l71-algo-1-zbxqn |[0m None
    [36m6hx3ay9l71-algo-1-zbxqn |[0m Train on 60000 samples, validate on 10000 samples
    [36m6hx3ay9l71-algo-1-zbxqn |[0m Epoch 1/1
    60000/60000 [==============================] - 20s 337us/step - loss: 0.3911 - acc: 0.8616 - val_loss: 0.3164 - val_acc: 0.8854
    [36m6hx3ay9l71-algo-1-zbxqn |[0m Validation loss    : 0.3164025844335556
    [36m6hx3ay9l71-algo-1-zbxqn |[0m Validation accuracy: 0.8854
    [36m6hx3ay9l71-algo-1-zbxqn |[0m WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/saved_model/simple_save.py:85: calling SavedModelBuilder.add_meta_graph_and_variables (from tensorflow.python.saved_model.builder_impl) with legacy_init_op is deprecated and will be removed in a future version.
    [36m6hx3ay9l71-algo-1-zbxqn |[0m Instructions for updating:
    [36m6hx3ay9l71-algo-1-zbxqn |[0m Pass your op to the equivalent parameter main_op instead.
    [36m6hx3ay9l71-algo-1-zbxqn |[0m 2021-06-13 10:03:59,724 sagemaker-containers INFO     Reporting training SUCCESS
    [36m6hx3ay9l71-algo-1-zbxqn exited with code 0
    [0mAborting on container exit...
    ===== Job Complete =====
    

### Model will be saved in S3
