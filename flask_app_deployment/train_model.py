from pyspark.sql import SparkSession
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql.functions import *
from pyspark.ml.pipeline import Transformer,Estimator
from pyspark.ml.feature import StringIndexer,VectorAssembler,QuantileDiscretizer
from pyspark.ml.classification import LogisticRegression
from  pyspark.ml.param.shared import *
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark import keyword_only
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator

import preprocess_file

st = SparkSession \
        .builder \
        .appName('Titanic') \
        .getOrCreate()

train_df = st.read.csv('titanic_train.csv',inferSchema=True,header=True)

def training_stage():
    my_model = preprocess_file.preprocess_transform()
    df = my_model.transform(train_df)
    feature = VectorAssembler(inputCols=['Pclass','Age','Fare','Alone','Sex_index','Embarked_index','Initial_index'],outputCol="features")
    #   lr = LogisticRegression(labelCol='Survived',featuresCol='features')
    rf = RandomForestClassifier(labelCol="Survived", featuresCol="features", numTrees=10)
    #   gb = GBTClassifier(labelCol="Survived", featuresCol="features", maxIter=10)

    '''
    pipeline stages initilization , fit and transform.
    '''
    pipeline = Pipeline(stages=[feature,rf])
    #   model = pipeline.fit(df)

    paramGrid = ParamGridBuilder().addGrid(rf.numTrees, [100,300]).build()

    evaluator = MulticlassClassificationEvaluator(labelCol="Survived", predictionCol="prediction", metricName="accuracy")

    crossval = CrossValidator(estimator=pipeline,
                            estimatorParamMaps=paramGrid,
                            evaluator=evaluator,
                            numFolds=3)  # use 3+ folds in practice

    # Run cross-validation, and choose the best set of parameters.
    cvModel = crossval.fit(df)
    cvModel.bestModel.write().overwrite().save('titanic_saved_model')