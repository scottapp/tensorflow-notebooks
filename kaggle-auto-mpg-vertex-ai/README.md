# Kaggle Auto MPG Model Deployed to Google Vertex AI

This repo demonstrates the use of custom container to train an AI model and deploy the model to GCP Vertex AI after training

First you will need to create a training and deployment pipeline project in the GCP Vertex AI console.

Then you can build, run and push the docker image in the repo to GCP for training and then deploy to an end point.

After an end point is deployed, you can test the model by running run_predictions.py. 

### Set environment variable
```bash
set PROJECT_ID=GOOGLE_GCP_PROJECT_ID
set BUCKET_NAME=gs://GOOGLE_GCP_BUCKET
set IMAGE_URI=gcr.io/%PROJECT_ID%/mpg:v1
```
or use Artifact URI
```
set IMAGE_URI=LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY/IMAGE
```

### Create storage bucket
```bash
gsutil mb -l us-central1 %BUCKET_NAME%
```

### Build docker image
```bash
docker build ./ -t %IMAGE_URI%
```

### Run image to make sure it runs
```bash
docker run %IMAGE_URI%
```

### Push to GCP
```bash
docker push %IMAGE_URI%
```
 