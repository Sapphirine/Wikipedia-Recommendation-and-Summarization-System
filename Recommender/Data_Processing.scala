import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, concat, monotonically_increasing_id}

class Data_Processing {

  // Load raw Data
  def data_raw_csv(path: String,
                   header: String = "false",
                   delimiter: String = ",",
                   inferSchema: String = "true",
                   spark: SparkSession): DataFrame ={

    val df = spark.read
      .format("csv")
      .option("header", header)
      .option("inferSchema", inferSchema)
      .option("delimiter", delimiter)
      .load(path)
//      .sample(true, 0.001)

    println(" --INFO:|Data| Raw data load")
    df
  }

  // Process data
  def data_clean(df: DataFrame, spark: SparkSession): DataFrame ={

    val df_null = df.na.fill(" ")

    var text = df_null
      .select( concat(df.columns.takeRight(df.columns.length - 1).map(c => col(c)): _*) )
      .withColumn("id", monotonically_increasing_id())
    val wiki_index = df_null
      .select("_c0")
      .withColumn("id", monotonically_increasing_id())

    val column_name = text.columns.head

    spark.conf.set("spark.sql.crossJoin.enabled", "true")

    val df_cleaned = wiki_index.join(text, Seq("id"),joinType="outer").toDF("index", "wiki_index", "text")

    println(" --INFO:|Data| Data pre-clean: NA, concat")
    df_cleaned
  }

  // load data and clean

  def data_process(path: String,
                   header: String = "false",
                   delimiter: String = ",",
                   inferSchema: String = "true",
                   spark: SparkSession): DataFrame = {

    val raw = data_raw_csv(path, header , delimiter, inferSchema, spark)

    val data = data_clean(raw, spark)

    data
  }
}
