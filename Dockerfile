FROM jupyter/tensorflow-notebook:tensorflow-2.6.2

RUN pip3 install findspark pyspark==3.3.0

USER root
RUN apt-get update
RUN apt-get install openjdk-11-jdk-headless -qq > /dev/null

# docker build -t jupyter .
# docker run -it -p 8888:8888 -v %cd%:/home/jovyan/workspace jupyter
