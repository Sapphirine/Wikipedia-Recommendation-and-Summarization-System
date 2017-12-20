# load data
import os
import glob
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, DoubleType, LongType
from pyspark.sql import Row

def loadTopicData(main_path, topic, spark, show = False):

    # schema = StructType([
    #     StructField("index", LongType(), True),
    #     StructField("text", StringType(), True),
    #     StructField("topicDistribution", ArrayType(DoubleType()), True),
    #     StructField("wiki_index", StringType(), True)
    # ])

    topic_path = main_path + "/topic=" + str(topic)

    main_path = glob.glob(topic_path + '/*.json')[0]

    # print(main_path)

    df = spark.read.json(main_path)

    df2 = df.withColumn("topicDistribution", df.topicDistribution.getField("values"))

    if (show):

        df2.show()

        df2.printSchema()

    return df2

def loadInput(text, spark, sc):

    line = [(0, text, "wikipeida_input")]

    rdd = sc.parallelize(line)

    rdd_map = rdd.map(lambda x: Row(index=x[0], text=x[1], wiki_index = x[2]))

    df = spark.createDataFrame(rdd_map)

    return df

