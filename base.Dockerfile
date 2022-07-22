FROM debian:bullseye

RUN apt-get clean && apt-get update -y
RUN apt-get install -y python3 python3-pip curl wget unzip procps
RUN apt-get install -y openjdk-11-jdk

RUN wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public | apt-key add -
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN rm -rf /var/lib/apt/lists/*

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
ENV SHARED_WORKSPACE=/opt/workspace
RUN mkdir -p ${SHARED_WORKSPACE}
VOLUME ${SHARED_WORKSPACE}


# docker build -f base.Dockerfile -t mk-spark-base .
