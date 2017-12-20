import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.udf
import org.apache.spark.ml.linalg.Vector



object main {

  def main(args: Array[String]): Unit ={

    val spark = SparkSession.builder
      .appName("main")
      .config("spark.master", "local")
      .getOrCreate()

    import spark.implicits._

    spark.sparkContext.setLogLevel("ERROR")

    val K = 40

    val save_loc = "output_K40"
    val pip_path = save_loc + "/df_pip"
    val lda_path = save_loc + "/lda"
    val json_path = save_loc + "/result.json"

    val data = new Data_Processing()
      .data_process("/Users/hongbowang/Personal/Data/wiki data/wiki.csv", spark = spark)

//    val df_test = spark.read
//      .format("csv")
//      .option("header", "true")
//      .option("inferSchema", "true")
//      .option("delimiter", "|")
//      .load("/Users/hongbowang/Personal/Data/wiki.csv")

    val pipeline = new Pipeline_Builder()
    val df_nlp = pipeline.pip_builder(data_train=data, save = true, path = pip_path) // training data cleaning
    val lda_model = pipeline.lda_pip(data = df_nlp, k=K, save = true, path = lda_path) // training lda model
    val df_predict =  pipeline.lda_predict(data_test=df_nlp, lda=lda_model)

    val df_select_certain = df_predict.select("index", "wiki_index", "topicDistribution", "text")
//    val df_select_certain = df_predict.select("topicDistribution")

    val func = udf( (x: Vector) => x.toDense.values.toSeq.indices.maxBy(x.toDense.values) )
    val df = df_select_certain.withColumn("topic", func(($"topicDistribution")))

    df.show(20)
    df.coalesce(1).write.partitionBy("topic").format("org.apache.spark.sql.json").save(json_path)

    print(" --INFO:|System|: done")
  }
}
