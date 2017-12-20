import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.ml.feature.{RegexTokenizer}
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.clustering.{LDA, LDAModel}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}

class Pipeline_Builder{

  def feature_trans_pip(colnames: String): Array[PipelineStage] ={

    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
    val hashingTF = new HashingTF().setInputCol("filtered").setOutputCol("tf_hash").setNumFeatures(20)
    val idf = new IDF().setInputCol("tf_hash").setOutputCol("rawFeatures")

    Array(tokenizer,remover,hashingTF,idf)
  }

  def pip_builder(data_train: DataFrame,
                  colnames: String = "text",
                  save: Boolean=false,
                  path:String = null): DataFrame ={

    val stages = feature_trans_pip(colnames)
    println( " --INFO:|Pipe| Stage")

    val pip_model = new Pipeline().setStages(stages).fit(data_train)

    if (save){
      if (path.isEmpty){return null}
      pip_model.save(path)
      println(" --INFO:|Pipe| Data_Pip Model Saved")
    }

    val data_pip = pip_model.transform(data_train)
    println( " --INFO:|Pipe| Pipeline_model")

    data_pip
  }

  def lda_pip(data: DataFrame,
              k: Int = 20,
              iter: Int = 10,
              colnames: String = "rawFeatures",
              save: Boolean=false,
              path: String = null): LDAModel = {

    val lda = new LDA().setK(k)
      .setMaxIter(iter)
      .setFeaturesCol(colnames)
      .setSeed(1)
      .fit(data)

    if (save) {
      lda.save(path)
      println(" --INFO:|Pipe| LDA Model Saved")
    }

    lda
  }

  def lda_predict(data_test: DataFrame,
                 lda: LDAModel): DataFrame ={

    val cluster_result = lda.transform(data_test)

    cluster_result
  }
}
