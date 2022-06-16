# Set base image (host OS)
FROM ubuntu:20.04

# By default, listen on port 8080
EXPOSE 8080/tcp

# Set the working directory in the container
WORKDIR /app

# Install any dependencies
RUN apt-get update -y
RUN apt-get install -y python3-pip
RUN pip3 install --upgrade pip
#RUN apt-get install -y python3-dev libpq-dev

# Copy the dependencies file to the working directory
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY .env_docker .env
COPY saved_model ./saved_model
COPY processor_state.pkl ./processor_state.pkl
COPY src ./src
COPY tests ./tests

ENV PYTHONPATH "${PYTHONPATH}:/app/src"
RUN pytest

CMD ["uvicorn", "src.fastapi_prediction_server:app", "--host", "0.0.0.0", "--port", "8080"]
