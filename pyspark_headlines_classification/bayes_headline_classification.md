# NaiveBayes Headlines Classification Using Spark

**This is a simple walkthrough of classification of UCI ML News Aggregator dataset using Spark**


```python
import os
import sys
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import mean, split, col, regexp_extract, when, lit, udf, regexp_replace,isnull
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import QuantileDiscretizer

from pyspark.ml.feature import CountVectorizer, StringIndexer, RegexTokenizer, StopWordsRemover
from pyspark.sql.types import StringType, IntegerType
from pyspark.ml.classification import NaiveBayes
```


```python
spark = SparkSession.builder.appName("pyspark-notebook").master("spark://spark-master:7077").config("spark.executor.memory", "512m").getOrCreate()
```


```python
spark
```





    <div>
        <p><b>SparkSession - in-memory</b></p>

<div>
    <p><b>SparkContext</b></p>

    <p><a href="http://ec78a262844f:4040">Spark UI</a></p>

    <dl>
      <dt>Version</dt>
        <dd><code>v3.3.0</code></dd>
      <dt>Master</dt>
        <dd><code>spark://spark-master:7077</code></dd>
      <dt>AppName</dt>
        <dd><code>pyspark-notebook</code></dd>
    </dl>
</div>

    </div>





```python
data = spark.read.csv('data/uci-news-aggregator.csv', header='True', inferSchema='True')
```

                                                                                    


```python
data.show()
```

    +---+--------------------+--------------------+--------------------+--------+--------------------+--------------------+-------------+
    | ID|               TITLE|                 URL|           PUBLISHER|CATEGORY|               STORY|            HOSTNAME|    TIMESTAMP|
    +---+--------------------+--------------------+--------------------+--------+--------------------+--------------------+-------------+
    |  1|Fed official says...|http://www.latime...|   Los Angeles Times|       b|ddUyU0VZz0BRneMio...|     www.latimes.com|1394470370698|
    |  2|Fed's Charles Plo...|http://www.livemi...|            Livemint|       b|ddUyU0VZz0BRneMio...|    www.livemint.com|1394470371207|
    |  3|US open: Stocks f...|http://www.ifamag...|        IFA Magazine|       b|ddUyU0VZz0BRneMio...| www.ifamagazine.com|1394470371550|
    |  4|Fed risks falling...|http://www.ifamag...|        IFA Magazine|       b|ddUyU0VZz0BRneMio...| www.ifamagazine.com|1394470371793|
    |  5|Fed's Plosser: Na...|http://www.moneyn...|           Moneynews|       b|ddUyU0VZz0BRneMio...|   www.moneynews.com|1394470372027|
    |  6|Plosser: Fed May ...|http://www.nasdaq...|              NASDAQ|       b|ddUyU0VZz0BRneMio...|      www.nasdaq.com|1394470372212|
    |  7|Fed's Plosser: Ta...|http://www.market...|         MarketWatch|       b|ddUyU0VZz0BRneMio...| www.marketwatch.com|1394470372405|
    |  8|Fed's Plosser exp...|http://www.fxstre...|        FXstreet.com|       b|ddUyU0VZz0BRneMio...|    www.fxstreet.com|1394470372615|
    |  9|US jobs growth la...|http://economicti...|      Economic Times|       b|ddUyU0VZz0BRneMio...|economictimes.ind...|1394470372792|
    | 10|ECB unlikely to e...|http://www.iii.co...|Interactive Investor|       b|dPhGU51DcrolUIMxb...|       www.iii.co.uk|1394470501265|
    | 11|ECB unlikely to e...|http://in.reuters...|       Reuters India|       b|dPhGU51DcrolUIMxb...|      in.reuters.com|1394470501410|
    | 12|EU's half-baked b...|http://blogs.reut...| Reuters UK \(blog\)|       b|dPhGU51DcrolUIMxb...|   blogs.reuters.com|1394470501587|
    | 13|Europe reaches cr...|http://in.reuters...|             Reuters|       b|dPhGU51DcrolUIMxb...|      in.reuters.com|1394470501755|
    | 14|ECB FOCUS-Stronge...|http://in.reuters...|             Reuters|       b|dPhGU51DcrolUIMxb...|      in.reuters.com|1394470501948|
    | 15|EU aims for deal ...|http://main.omano...| Oman Daily Observer|       b|dPhGU51DcrolUIMxb...|main.omanobserver.om|1394470502141|
    | 16|Forex - Pound dro...|http://www.nasdaq...|              NASDAQ|       b|dPhGU51DcrolUIMxb...|      www.nasdaq.com|1394470502316|
    | 17|Noyer Says Strong...|http://www.sfgate...|San Francisco Chr...|       b|dPhGU51DcrolUIMxb...|      www.sfgate.com|1394470502543|
    | 18|EU Week Ahead Mar...|http://blogs.wsj....|Wall Street Journ...|       b|dPhGU51DcrolUIMxb...|       blogs.wsj.com|1394470502744|
    | 19|ECB member Noyer ...|http://www.ifamag...|        IFA Magazine|       b|dPhGU51DcrolUIMxb...| www.ifamagazine.com|1394470502946|
    | 20|Euro Anxieties Wa...|http://www.busine...|        Businessweek|       b|dPhGU51DcrolUIMxb...|www.businessweek.com|1394470503148|
    +---+--------------------+--------------------+--------------------+--------+--------------------+--------------------+-------------+
    only showing top 20 rows
    



```python
data.count()
```

                                                                                    




    422937




```python
title_category = data.select("TITLE","CATEGORY")
```


```python
title_category.show()
```

    +--------------------+--------+
    |               TITLE|CATEGORY|
    +--------------------+--------+
    |Fed official says...|       b|
    |Fed's Charles Plo...|       b|
    |US open: Stocks f...|       b|
    |Fed risks falling...|       b|
    |Fed's Plosser: Na...|       b|
    |Plosser: Fed May ...|       b|
    |Fed's Plosser: Ta...|       b|
    |Fed's Plosser exp...|       b|
    |US jobs growth la...|       b|
    |ECB unlikely to e...|       b|
    |ECB unlikely to e...|       b|
    |EU's half-baked b...|       b|
    |Europe reaches cr...|       b|
    |ECB FOCUS-Stronge...|       b|
    |EU aims for deal ...|       b|
    |Forex - Pound dro...|       b|
    |Noyer Says Strong...|       b|
    |EU Week Ahead Mar...|       b|
    |ECB member Noyer ...|       b|
    |Euro Anxieties Wa...|       b|
    +--------------------+--------+
    only showing top 20 rows
    



```python
def null_value_count(df):
  null_columns_counts = []
  numRows = df.count()
  for k in df.columns:
    nullRows = df.where(col(k).isNull()).count()
    if(nullRows > 0):
      temp = k,nullRows
      null_columns_counts.append(temp)
  return(null_columns_counts)
```


```python
null_columns_count_list = null_value_count(title_category)
```

                                                                                    


```python
null_columns_count_list
```




    [('TITLE', 389), ('CATEGORY', 516)]




```python
df = spark.createDataFrame(null_columns_count_list, ['Column_With_Null_Value', 'Null_Values_Count'])
```


```python
df.show()
```

    [Stage 16:>                                                         (0 + 1) / 1]

    +----------------------+-----------------+
    |Column_With_Null_Value|Null_Values_Count|
    +----------------------+-----------------+
    |                 TITLE|              389|
    |              CATEGORY|              516|
    +----------------------+-----------------+
    


                                                                                    


```python
title_category = title_category.dropna()
```


```python
title_category.count()
```

                                                                                    




    422421




```python
title_category.show(truncate=False)
```

    +---------------------------------------------------------------------------+--------+
    |TITLE                                                                      |CATEGORY|
    +---------------------------------------------------------------------------+--------+
    |Fed official says weak data caused by weather, should not slow taper       |b       |
    |Fed's Charles Plosser sees high bar for change in pace of tapering         |b       |
    |US open: Stocks fall after Fed official hints at accelerated tapering      |b       |
    |Fed risks falling 'behind the curve', Charles Plosser says                 |b       |
    |Fed's Plosser: Nasty Weather Has Curbed Job Growth                         |b       |
    |Plosser: Fed May Have to Accelerate Tapering Pace                          |b       |
    |Fed's Plosser: Taper pace may be too slow                                  |b       |
    |Fed's Plosser expects US unemployment to fall to 6.2% by the end of 2014   |b       |
    |US jobs growth last month hit by weather:Fed President Charles Plosser     |b       |
    |ECB unlikely to end sterilisation of SMP purchases - traders               |b       |
    |ECB unlikely to end sterilization of SMP purchases: traders                |b       |
    |EU's half-baked bank union could work                                      |b       |
    |Europe reaches crunch point on banking union                               |b       |
    |ECB FOCUS-Stronger euro drowns out ECB's message to keep rates low for  ...|b       |
    |EU aims for deal on tackling failing banks                                 |b       |
    |Forex - Pound drops to one-month lows against euro                         |b       |
    |Noyer Says Strong Euro Creates Unwarranted Economic Pressure               |b       |
    |EU Week Ahead March 10-14: Bank Resolution, Transparency, Ukraine          |b       |
    |ECB member Noyer is 'very open to all kinds of tools'                      |b       |
    |Euro Anxieties Wane as Bunds Top Treasuries, Spain Debt Rallies            |b       |
    +---------------------------------------------------------------------------+--------+
    only showing top 20 rows
    



```python
title_category.select("Category").distinct().count()
```

                                                                                    




    265




```python
title_category.groupBy("Category").count().orderBy(col("count").desc()).show(truncate=False)
```

    [Stage 28:=============================>                            (1 + 1) / 2]

    +--------------------+------+
    |Category            |count |
    +--------------------+------+
    |e                   |152127|
    |b                   |115935|
    |t                   |108237|
    |m                   |45616 |
    |Us Magazine         |31    |
    |GossipCop           |20    |
    |Contactmusic.com    |20    |
    |Complex.com         |12    |
    |CBS News            |12    |
    |The Hollywood Gossip|11    |
    |HipHopDX            |11    |
    |We Got This Covered |10    |
    |HeadlinePlanet.com  |10    |
    |Gamepur             |8     |
    |TooFab.com          |7     |
    |Wetpaint            |7     |
    |WorstPreviews.com   |7     |
    |Consequence of Sound|7     |
    |The Escapist        |6     |
    |Reality TV World    |5     |
    +--------------------+------+
    only showing top 20 rows
    


                                                                                    


```python
title_category.groupBy("TITLE").count().orderBy(col("count").desc()).show(truncate=False)
```

    [Stage 33:=============================>                            (1 + 1) / 2]

    +----------------------------------------------------------------------------------+-----+
    |TITLE                                                                             |count|
    +----------------------------------------------------------------------------------+-----+
    |The article requested cannot be found! Please refresh your browser or go back  ...|145  |
    |Business Highlights                                                               |59   |
    |Posted by Parvez Jabri                                                            |59   |
    |Posted by Imaduddin                                                               |53   |
    |Posted by Shoaib-ur-Rehman Siddiqui                                               |52   |
    |(click the phrases to see a list)                                                 |51   |
    |Business Wire                                                                     |41   |
    |PR Newswire                                                                       |38   |
    |Posted by Muhammad Iqbal                                                          |35   |
    |Change text size for the story                                                    |34   |
    |Get the Most Popular Beauty World News Stories in a Weekly Newsletter             |34   |
    |International markets roundup                                                     |33   |
    |Business briefs                                                                   |33   |
    |India Morning Call-Global Markets                                                 |27   |
    |10 Things to Know for Today                                                       |22   |
    |Breaking news                                                                     |21   |
    |Perez Recommends                                                                  |19   |
    |From ColumbusAlive.com                                                            |18   |
    |Texas Weekly Gas Price Update and Outlook                                         |18   |
    |The Daily Dish                                                                    |17   |
    +----------------------------------------------------------------------------------+-----+
    only showing top 20 rows
    


                                                                                    


```python
title_category = title_category.withColumn("only_str", regexp_replace(col('TITLE'), '\d+', ''))
```


```python
title_category.select("TITLE","only_str").show(truncate=False)
```

    +---------------------------------------------------------------------------+---------------------------------------------------------------------------+
    |TITLE                                                                      |only_str                                                                   |
    +---------------------------------------------------------------------------+---------------------------------------------------------------------------+
    |Fed official says weak data caused by weather, should not slow taper       |Fed official says weak data caused by weather, should not slow taper       |
    |Fed's Charles Plosser sees high bar for change in pace of tapering         |Fed's Charles Plosser sees high bar for change in pace of tapering         |
    |US open: Stocks fall after Fed official hints at accelerated tapering      |US open: Stocks fall after Fed official hints at accelerated tapering      |
    |Fed risks falling 'behind the curve', Charles Plosser says                 |Fed risks falling 'behind the curve', Charles Plosser says                 |
    |Fed's Plosser: Nasty Weather Has Curbed Job Growth                         |Fed's Plosser: Nasty Weather Has Curbed Job Growth                         |
    |Plosser: Fed May Have to Accelerate Tapering Pace                          |Plosser: Fed May Have to Accelerate Tapering Pace                          |
    |Fed's Plosser: Taper pace may be too slow                                  |Fed's Plosser: Taper pace may be too slow                                  |
    |Fed's Plosser expects US unemployment to fall to 6.2% by the end of 2014   |Fed's Plosser expects US unemployment to fall to .% by the end of          |
    |US jobs growth last month hit by weather:Fed President Charles Plosser     |US jobs growth last month hit by weather:Fed President Charles Plosser     |
    |ECB unlikely to end sterilisation of SMP purchases - traders               |ECB unlikely to end sterilisation of SMP purchases - traders               |
    |ECB unlikely to end sterilization of SMP purchases: traders                |ECB unlikely to end sterilization of SMP purchases: traders                |
    |EU's half-baked bank union could work                                      |EU's half-baked bank union could work                                      |
    |Europe reaches crunch point on banking union                               |Europe reaches crunch point on banking union                               |
    |ECB FOCUS-Stronger euro drowns out ECB's message to keep rates low for  ...|ECB FOCUS-Stronger euro drowns out ECB's message to keep rates low for  ...|
    |EU aims for deal on tackling failing banks                                 |EU aims for deal on tackling failing banks                                 |
    |Forex - Pound drops to one-month lows against euro                         |Forex - Pound drops to one-month lows against euro                         |
    |Noyer Says Strong Euro Creates Unwarranted Economic Pressure               |Noyer Says Strong Euro Creates Unwarranted Economic Pressure               |
    |EU Week Ahead March 10-14: Bank Resolution, Transparency, Ukraine          |EU Week Ahead March -: Bank Resolution, Transparency, Ukraine              |
    |ECB member Noyer is 'very open to all kinds of tools'                      |ECB member Noyer is 'very open to all kinds of tools'                      |
    |Euro Anxieties Wane as Bunds Top Treasuries, Spain Debt Rallies            |Euro Anxieties Wane as Bunds Top Treasuries, Spain Debt Rallies            |
    +---------------------------------------------------------------------------+---------------------------------------------------------------------------+
    only showing top 20 rows
    



```python
regex_tokenizer = RegexTokenizer(inputCol="only_str", outputCol="words", pattern="\\W")
raw_words = regex_tokenizer.transform(title_category)
```


```python
raw_words.show()
```

    +--------------------+--------+--------------------+--------------------+
    |               TITLE|CATEGORY|            only_str|               words|
    +--------------------+--------+--------------------+--------------------+
    |Fed official says...|       b|Fed official says...|[fed, official, s...|
    |Fed's Charles Plo...|       b|Fed's Charles Plo...|[fed, s, charles,...|
    |US open: Stocks f...|       b|US open: Stocks f...|[us, open, stocks...|
    |Fed risks falling...|       b|Fed risks falling...|[fed, risks, fall...|
    |Fed's Plosser: Na...|       b|Fed's Plosser: Na...|[fed, s, plosser,...|
    |Plosser: Fed May ...|       b|Plosser: Fed May ...|[plosser, fed, ma...|
    |Fed's Plosser: Ta...|       b|Fed's Plosser: Ta...|[fed, s, plosser,...|
    |Fed's Plosser exp...|       b|Fed's Plosser exp...|[fed, s, plosser,...|
    |US jobs growth la...|       b|US jobs growth la...|[us, jobs, growth...|
    |ECB unlikely to e...|       b|ECB unlikely to e...|[ecb, unlikely, t...|
    |ECB unlikely to e...|       b|ECB unlikely to e...|[ecb, unlikely, t...|
    |EU's half-baked b...|       b|EU's half-baked b...|[eu, s, half, bak...|
    |Europe reaches cr...|       b|Europe reaches cr...|[europe, reaches,...|
    |ECB FOCUS-Stronge...|       b|ECB FOCUS-Stronge...|[ecb, focus, stro...|
    |EU aims for deal ...|       b|EU aims for deal ...|[eu, aims, for, d...|
    |Forex - Pound dro...|       b|Forex - Pound dro...|[forex, pound, dr...|
    |Noyer Says Strong...|       b|Noyer Says Strong...|[noyer, says, str...|
    |EU Week Ahead Mar...|       b|EU Week Ahead Mar...|[eu, week, ahead,...|
    |ECB member Noyer ...|       b|ECB member Noyer ...|[ecb, member, noy...|
    |Euro Anxieties Wa...|       b|Euro Anxieties Wa...|[euro, anxieties,...|
    +--------------------+--------+--------------------+--------------------+
    only showing top 20 rows
    



```python
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
words_df = remover.transform(raw_words)
```


```python
words_df.select("words","filtered").show()
```

    +--------------------+--------------------+
    |               words|            filtered|
    +--------------------+--------------------+
    |[fed, official, s...|[fed, official, s...|
    |[fed, s, charles,...|[fed, charles, pl...|
    |[us, open, stocks...|[us, open, stocks...|
    |[fed, risks, fall...|[fed, risks, fall...|
    |[fed, s, plosser,...|[fed, plosser, na...|
    |[plosser, fed, ma...|[plosser, fed, ma...|
    |[fed, s, plosser,...|[fed, plosser, ta...|
    |[fed, s, plosser,...|[fed, plosser, ex...|
    |[us, jobs, growth...|[us, jobs, growth...|
    |[ecb, unlikely, t...|[ecb, unlikely, e...|
    |[ecb, unlikely, t...|[ecb, unlikely, e...|
    |[eu, s, half, bak...|[eu, half, baked,...|
    |[europe, reaches,...|[europe, reaches,...|
    |[ecb, focus, stro...|[ecb, focus, stro...|
    |[eu, aims, for, d...|[eu, aims, deal, ...|
    |[forex, pound, dr...|[forex, pound, dr...|
    |[noyer, says, str...|[noyer, says, str...|
    |[eu, week, ahead,...|[eu, week, ahead,...|
    |[ecb, member, noy...|[ecb, member, noy...|
    |[euro, anxieties,...|[euro, anxieties,...|
    +--------------------+--------------------+
    only showing top 20 rows
    



```python
indexer = StringIndexer(inputCol="CATEGORY", outputCol="categoryIndex")
feature_data = indexer.fit(words_df).transform(words_df)
```

                                                                                    


```python
feature_data.select("CATEGORY","categoryIndex").show()
```

    +--------+-------------+
    |CATEGORY|categoryIndex|
    +--------+-------------+
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    |       b|          1.0|
    +--------+-------------+
    only showing top 20 rows
    



```python
cv = CountVectorizer(inputCol="filtered", outputCol="features")
model = cv.fit(feature_data)
countVectorizer_feateures = model.transform(feature_data)
```

                                                                                    


```python
countVectorizer_feateures.show()
```

                                                                                    

    +--------------------+--------+--------------------+--------------------+--------------------+-------------+--------------------+
    |               TITLE|CATEGORY|            only_str|               words|            filtered|categoryIndex|            features|
    +--------------------+--------+--------------------+--------------------+--------------------+-------------+--------------------+
    |Fed official says...|       b|Fed official says...|[fed, official, s...|[fed, official, s...|          1.0|(49043,[5,42,112,...|
    |Fed's Charles Plo...|       b|Fed's Charles Plo...|[fed, s, charles,...|[fed, charles, pl...|          1.0|(49043,[58,84,112...|
    |US open: Stocks f...|       b|US open: Stocks f...|[us, open, stocks...|[us, open, stocks...|          1.0|(49043,[1,27,112,...|
    |Fed risks falling...|       b|Fed risks falling...|[fed, risks, fall...|[fed, risks, fall...|          1.0|(49043,[5,112,578...|
    |Fed's Plosser: Na...|       b|Fed's Plosser: Na...|[fed, s, plosser,...|[fed, plosser, na...|          1.0|(49043,[112,121,5...|
    |Plosser: Fed May ...|       b|Plosser: Fed May ...|[plosser, fed, ma...|[plosser, fed, ma...|          1.0|(49043,[8,112,223...|
    |Fed's Plosser: Ta...|       b|Fed's Plosser: Ta...|[fed, s, plosser,...|[fed, plosser, ta...|          1.0|(49043,[8,112,123...|
    |Fed's Plosser exp...|       b|Fed's Plosser exp...|[fed, s, plosser,...|[fed, plosser, ex...|          1.0|(49043,[1,112,135...|
    |US jobs growth la...|       b|US jobs growth la...|[us, jobs, growth...|[us, jobs, growth...|          1.0|(49043,[1,112,121...|
    |ECB unlikely to e...|       b|ECB unlikely to e...|[ecb, unlikely, t...|[ecb, unlikely, e...|          1.0|(49043,[135,200,2...|
    |ECB unlikely to e...|       b|ECB unlikely to e...|[ecb, unlikely, t...|[ecb, unlikely, e...|          1.0|(49043,[135,200,2...|
    |EU's half-baked b...|       b|EU's half-baked b...|[eu, s, half, bak...|[eu, half, baked,...|          1.0|(49043,[78,535,60...|
    |Europe reaches cr...|       b|Europe reaches cr...|[europe, reaches,...|[europe, reaches,...|          1.0|(49043,[405,940,1...|
    |ECB FOCUS-Stronge...|       b|ECB FOCUS-Stronge...|[ecb, focus, stro...|[ecb, focus, stro...|          1.0|(49043,[133,171,1...|
    |EU aims for deal ...|       b|EU aims for deal ...|[eu, aims, for, d...|[eu, aims, deal, ...|          1.0|(49043,[29,593,64...|
    |Forex - Pound dro...|       b|Forex - Pound dro...|[forex, pound, dr...|[forex, pound, dr...|          1.0|(49043,[10,171,17...|
    |Noyer Says Strong...|       b|Noyer Says Strong...|[noyer, says, str...|[noyer, says, str...|          1.0|(49043,[5,171,438...|
    |EU Week Ahead Mar...|       b|EU Week Ahead Mar...|[eu, week, ahead,...|[eu, week, ahead,...|          1.0|(49043,[56,78,168...|
    |ECB member Noyer ...|       b|ECB member Noyer ...|[ecb, member, noy...|[ecb, member, noy...|          1.0|(49043,[200,240,3...|
    |Euro Anxieties Wa...|       b|Euro Anxieties Wa...|[euro, anxieties,...|[euro, anxieties,...|          1.0|(49043,[47,171,10...|
    +--------------------+--------+--------------------+--------------------+--------------------+-------------+--------------------+
    only showing top 20 rows
    



```python
(trainingData, testData) = countVectorizer_feateures.randomSplit([0.8, 0.2],seed = 11)
```


```python
nb = NaiveBayes(modelType="multinomial",labelCol="categoryIndex", featuresCol="features")
nbModel = nb.fit(trainingData)
nb_predictions = nbModel.transform(testData)
```

                                                                                    


```python
nb_predictions.select("prediction", "categoryIndex", "features").show(5)
```

    22/07/22 09:04:25 WARN DAGScheduler: Broadcasting large task binary with size 86.6 MiB


    [Stage 49:>                                                         (0 + 1) / 1]

    +----------+-------------+--------------------+
    |prediction|categoryIndex|            features|
    +----------+-------------+--------------------+
    |       1.0|          0.0|(49043,[15,22,26,...|
    |       0.0|         20.0|(49043,[74,113,39...|
    |       0.0|          0.0|(49043,[21,50,51,...|
    |       0.0|          0.0|(49043,[21,50,62,...|
    |       0.0|          0.0|(49043,[6,21,22,5...|
    +----------+-------------+--------------------+
    only showing top 5 rows
    


                                                                                    


```python
evaluator = MulticlassClassificationEvaluator(labelCol="categoryIndex", predictionCol="prediction", metricName="accuracy")
nb_accuracy = evaluator.evaluate(nb_predictions)
print("Accuracy of NaiveBayes is = %g"% (nb_accuracy))
print("Test Error of NaiveBayes = %g " % (1.0 - nb_accuracy))
```

    22/07/22 09:04:49 WARN DAGScheduler: Broadcasting large task binary with size 86.7 MiB


    [Stage 50:=============================>                            (1 + 1) / 2]

    Accuracy of NaiveBayes is = 0.926257
    Test Error of NaiveBayes = 0.0737432 


                                                                                    


```python

```

* [original notebook](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/5722190290795989/2546946806099472/8175309257345795/latest.html)
