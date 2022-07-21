FROM mk-spark-base

# Python packages
RUN pip3 install wget requests datawrangler
# RUN wget https://apache.mirrors.tworzy.net/incubator/livy/0.7.0-incubating/apache-livy-0.7.0-incubating-bin.zip -O livy.zip && unzip livy.zip -d /usr/bin/


EXPOSE 8080 7077 8998 8888
WORKDIR ${APP}
ENTRYPOINT ["sh", "/usr/bin/spark-3.3.0-bin-hadoop3/sbin/start-master.sh"]

# docker build -f master.Dockerfile -t mk-spark-master .
