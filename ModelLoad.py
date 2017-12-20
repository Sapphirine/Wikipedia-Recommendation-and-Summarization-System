import numpy as np
from pyspark.ml.clustering import LocalLDAModel
from pyspark.ml import PipelineModel

class Model:

    def __init__(self):

        self.pipeline = None

        self.lda = None

    ## Load Model
    def loadPipModel(self, path):

        pipeline = PipelineModel.load(path)

        self.pipeline = pipeline

    def loadLDA(self, path):

        model = LocalLDAModel.load(path)

        self.lda = model

    def loadAll(self, pip_path, lda_path):

        self.loadPipModel(pip_path)

        self.loadLDA(lda_path)

        return self.pipeline, self.lda

    # Use Model to predict
    def pipTrans(self, text):

        data = self.pipeline.transform(text)

        return data

    def ldaTrans(self, text):

        topic_df = self.lda.transform(text)

        return topic_df

    def ldaPredict(self, text, pip_path, lda_path):

        self.loadAll(pip_path, lda_path)

        clean_df = self.pipTrans(text)

        topic = self.ldaTrans(clean_df)

        dist = np.array(topic.select("topicDistribution").rdd.collect()[0]["topicDistribution"])

        index = np.argmax(dist)

        return topic, dist, index





