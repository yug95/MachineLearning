from flask import Flask
from sklearn.externals import joblib
from flask import Flask, jsonify, request
import pandas as pd
import pyspark
from pyspark.ml import Pipeline,PipelineModel
import preprocess_file
import train_model
from pyspark.sql import SparkSession
from pyspark.sql import Row,SQLContext
from collections import OrderedDict

app = Flask(__name__)

st = SparkSession \
        .builder \
        .appName('Titanic') \
        .getOrCreate()

sqlContext = SQLContext(st.sparkContext)

@app.route('/')
def train_model_fun():
    train_model.training_stage()
    return 'model_train is done'

@app.route('/predict',methods=['POST'])
def predict_func():
    test_data = request.get_json()
    
    # test_df = pd.DataFrame.from_dict(,orient='index')
    # test_df = test_df.T
    new_df = sqlContext.read.json(st.sparkContext.parallelize(test_data))
    model = PipelineModel.load('titanic_saved_model')
    process_data = preprocess_file.preprocess_transform().transform(new_df)
    predicted_data = model.transform(process_data)
    #print("predicted_data",predicted_data.show())
    print("my prediction",predicted_data.select("PassengerId","prediction").show())
    output = {'PassengerId':predicted_data.select("PassengerId").collect()[0][0],'Survived':predicted_data.select("prediction").collect()[0][0]}
    print(output)
    return jsonify(output)