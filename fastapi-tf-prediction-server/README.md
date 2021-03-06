# Tensorflow Prediction Server With FastAPI

- A tensorflow prediction server implemented with FastAPI framework
- Model is from [this google tutorial](https://cloud.google.com/ai-platform/docs/getting-started-keras)
- The notebook is [here](https://github.com/scottapp/tensorflow-notebooks/blob/main/fastapi-tf-prediction-server/income_prediction_us_census_data.md)
- Uses the United States Census Income Dataset
- Can be easily deployed to GCP Cloud Run


### Run some simple test after cloning
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
