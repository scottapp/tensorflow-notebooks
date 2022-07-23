from pyspark.sql import SparkSession
from pyspark.sql.functions import col, date_format


def init_spark():
    mem_size = '512m'
    sql = SparkSession.builder \
        .appName("ny-mta-bus-trip-app") \
        .config("spark.jars", "/home/jovyan/workspace/spark-mta-bus/data/postgresql-42.2.22.jar") \
        .config("spark.executor.memory", mem_size) \
        .config("spark.driver.memory", mem_size) \
        .getOrCreate()
    sc = sql.sparkContext
    return sql, sc


def main():
    # The original file is very big so you might run into error when executing this spark job
    # Here I use just a small fraction of the data to test the process works with memory size of just 512m
    # The result will be named output_result.csv in the current directory

    # file = "/home/jovyan/workspace/spark-mta-bus/data/MTA_Bus_Time_2014-08-01.csv"
    file = "/home/jovyan/workspace/spark-mta-bus/test_data.csv"
    sql, sc = init_spark()

    df = sql.read.load(file, format="csv", inferSchema="true", sep="\t", header="true") \
        .withColumn("report_hour", date_format(col("time_received"), "yyyy-MM-dd HH:00:00")) \
        .withColumn("report_date", date_format(col("time_received"), "yyyy-MM-dd"))

    # Filter invalid coordinates

    # output to local csv
    df.where("latitude <= 90 AND latitude >= -90 AND longitude <= 180 AND longitude >= -180") \
        .where("latitude != 0.000000 OR longitude !=  0.000000 ") \
        .toPandas().to_csv('output_result.csv')

    # output to database
    url = "jdbc:postgresql://demo-database:5432/mta_data"
    properties = {
        "user": "postgres",
        "password": "postgres1234",
        "driver": "org.postgresql.Driver"
    }

    """        
    df.where("latitude <= 90 AND latitude >= -90 AND longitude <= 180 AND longitude >= -180") \
        .where("latitude != 0.000000 OR longitude !=  0.000000 ") \
        .write \
        .jdbc(url=url, table="mta_reports", mode='append', properties=properties) \
        .save()
    """


if __name__ == '__main__':
    # spark-submit --master spark://spark-master:7077 --jars data/postgresql-42.2.22.jar main.py
    main()
