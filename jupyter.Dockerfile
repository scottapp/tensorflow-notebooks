FROM mk-spark-base

RUN pip3 install wget requests pandas numpy datawrangler findspark pyspark==3.3.0
RUN pip3 install jupyterlab

EXPOSE 8888

WORKDIR ${SHARED_WORKSPACE}

CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token=

# docker build -f jupyter.Dockerfile -t mk-jupyter .
