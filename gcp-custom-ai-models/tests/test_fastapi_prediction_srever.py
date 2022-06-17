from dotenv import load_dotenv
import os
import sys
import json
import pickle

"""
load_dotenv()
BASE_DIR = os.getenv('BASE_DIR')
assert BASE_DIR, 'error base dir'
sys.path.append(BASE_DIR)
"""

import tensorflow as tf
from fastapi.testclient import TestClient
from model_prediction import CustomModelPrediction
from fastapi_prediction_server import app

client = TestClient(app)


def test_app_hello_world():
    response = client.get("/hello_world")
    assert response.status_code == 200
    assert response.json() == {"msg": "Hello World"}


def test_app_hello_world_post():
    data = dict()
    data['name'] = 'test user'
    data['description'] = 'test desc'
    response = client.post("/hello_world_post", json=data)
    assert response.status_code == 200
    res_data = json.loads(response.text)
    assert res_data['name'] == 'test user'
    assert res_data['description'] == 'test desc'


def test_prediction_server_standalone():
    base_dir = os.getenv('BASE_DIR')
    model_dir = os.getenv('MODEL_DIR')
    assert base_dir
    assert model_dir
    local_model = tf.keras.models.load_model(model_dir)
    local_model.trainable = False

    with open('{}/processor_state.pkl'.format(base_dir), 'rb') as f:
        p = pickle.load(f)

    predict_server = CustomModelPrediction(local_model, p)
    techcrunch = [
        'Uber shuts down self-driving trucks unit',
        'Grover raises €37M Series A to offer latest tech products as a subscription',
        'Tech companies can now bid on the Pentagon’s $10B cloud contract'
    ]
    nytimes = [
        '‘Lopping,’ ‘Tips’ and the ‘Z-List’: Bias Lawsuit Explores Harvard’s Admissions',
        'A $3B Plan to Turn Hoover Dam into a Giant Battery',
        'A MeToo Reckoning in China’s Workplace Amid Wave of Accusations'
    ]
    github = [
        'Show HN: Moon – 3kb JavaScript UI compiler',
        'Show HN: Hello, a CLI tool for managing social media',
        'Firefox Nightly added support for time-travel debugging'
    ]
    instances = (techcrunch + nytimes + github)

    response = predict_server.predict(instances)
    print(response)


def test_prediction_server_webserver():
    techcrunch = [
        'Uber shuts down self-driving trucks unit',
        'Grover raises €37M Series A to offer latest tech products as a subscription',
        'Tech companies can now bid on the Pentagon’s $10B cloud contract'
    ]
    nytimes = [
        '‘Lopping,’ ‘Tips’ and the ‘Z-List’: Bias Lawsuit Explores Harvard’s Admissions',
        'A $3B Plan to Turn Hoover Dam into a Giant Battery',
        'A MeToo Reckoning in China’s Workplace Amid Wave of Accusations'
    ]
    github = [
        'Show HN: Moon – 3kb JavaScript UI compiler',
        'Show HN: Hello, a CLI tool for managing social media',
        'Firefox Nightly added support for time-travel debugging'
    ]
    instances = (techcrunch + nytimes + github)

    data = dict()
    data['instances'] = json.dumps(instances)
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    res_data = json.loads(response.text)
    assert res_data.get('predictions')
    predictions = res_data['predictions']
    assert len(predictions) == 9
    # [techcrunch, techcrunch, techcrunch, nytimes, nytimes, nytimes, github, github, techcrunch]
    print(predictions)
