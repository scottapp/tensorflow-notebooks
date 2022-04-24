# FastAPI Tensorflow Prediction Server


### Run some simple test after downloading
```bash
pytest
```

### Start server
```bash
uvicorn prediction_server:app --host 0.0.0.0 --port 8080
```

### Run query from cmd
```bash
curl -H "Content-Type: application/json" -X POST -d "{\"instances\":\"[[1.1929514778959478, 3.0, 1.525541514460902, 2.0, 9.0, 0.0, 4.0, -0.14479173735784842, -0.21713186390175285, 0.934371839987696, 38.0]]\"}" http://127.0.0.1:8080/predict/
```

Result should be
```bash
{"predictions":[0.743172287940979]}
```
