from fastapi.testclient import TestClient
import json
import requests
from prediction_server import app

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


def test_app_predict():
    input_data = list()
    with open('prediction_input_us_census.json', 'r') as f:
        lines = f.readlines()
        for line in lines:
            input_data.append(json.loads(line.strip()))
    data = dict()
    data['instances'] = json.dumps(input_data)
    response = client.post("/predict/", json=data)
    assert response.status_code == 200
    res_data = json.loads(response.text)
    assert res_data.get('predictions')
    predictions = res_data['predictions']
    assert len(predictions) == 20
    print(predictions)


"""
def test_use_requests():
    url = 'http://127.0.0.1:8080/predict/'
    data = dict()
    data['instances'] = ''
    res = requests.post(url, json=data)
    assert res.status_code == 200
"""
