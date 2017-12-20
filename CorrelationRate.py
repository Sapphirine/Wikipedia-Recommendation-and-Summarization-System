import os
import findspark
os.environ["SPARK_HOME"] = "/Users/LongNguyen/spark-2.2.0-bin-hadoop2.7"
findspark.init()

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession

import DataLoad as DT
import ModelLoad as MD
import CorrelationRate as CR

conf = SparkConf().setAppName('pubmed_open_access').setMaster('local[32]')
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

spark = SparkSession \
    .builder \
    .appName("Recommendation System") \
    .getOrCreate()

from scipy.spatial.distance import euclidean as euclidean_
from pyspark.sql.functions import udf

def calSimi(input_dis, df):

    euclidean = lambda x: udf(lambda y: euclidean_(x, y), "double")

    new_df = df.withColumn("dis", euclidean(input_dis)("topicDistribution"))

    return new_df

def recomm(text, data_path, pip_path, lda_path, recomm_num = 5):

    input_text = DT.loadInput(text, spark, sc)

    pred_df, pred_dis, pred_index = MD.Model().ldaPredict(input_text, pip_path=pip_path, lda_path=lda_path)

    data_withTopic = DT.loadTopicData(data_path, topic=pred_index, spark=spark)

    data_withDis = CR.calSimi(pred_dis, data_withTopic)

    data_sort = data_withDis.sort("dis")

    text_list = list()
    source = data_sort.select("text").rdd.take(recomm_num)

    for i in range(recomm_num):

        text_list.append(source[i]["text"])

    return data_sort, text_list