FROM jupyter/tensorflow-notebook:tensorflow-2.6.2

RUN pip3 install findspark pyspark==3.3.0

USER root
RUN apt-get update
RUN apt-get install openjdk-11-jdk-headless -qq > /dev/null
RUN apt-get install -y curl

RUN curl https://dlcdn.apache.org/spark/spark-3.3.0/spark-3.3.0-bin-hadoop3.tgz -o spark.tgz && \
    tar -xf spark.tgz && \
    mv spark-3.3.0-bin-hadoop3 /usr/bin/ && \
    mkdir /usr/bin/spark-3.3.0-bin-hadoop3/logs && \
    rm spark.tgz

RUN mkdir -p /tmp/logs/ && chmod a+w /tmp/logs/ && mkdir /app && chmod a+rwx /app && mkdir /data && chmod a+rwx /data
ENV JAVA_HOME=/usr
ENV SPARK_HOME=/usr/bin/spark-3.3.0-bin-hadoop3
ENV SPARK_NO_DAEMONIZE=true
ENV PATH=$SPARK_HOME:$PATH:/bin:$JAVA_HOME/bin:$JAVA_HOME/jre/bin
ENV SPARK_MASTER_HOST spark-master
ENV SPARK_MASTER_PORT 7077
ENV PYSPARK_PYTHON=/usr/bin/python
ENV PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
ENV APP=/app

#ENV SHARED_WORKSPACE=/home/jovyan/shared_workspace
#RUN mkdir -p ${SHARED_WORKSPACE}
#RUN chown jovyan /home/jovyan/shared_workspace

# docker build -t jupyter .
# docker run -it -p 8888:8888 -v %cd%:/home/jovyan/workspace jupyter
