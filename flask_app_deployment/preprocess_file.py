import pyspark
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



class preprocess_transform(Transformer,HasOutputCols,DefaultParamsReadable, DefaultParamsWritable,):
  
    value = Param(Params._dummy(),"value","value to fill")
  
    @keyword_only
    def __init__(self, outputCols=None):
        super(preprocess_transform, self).__init__()
        kwargs = self._input_kwargs
        self._set(**kwargs)

    @keyword_only
    def setParams(self, outputCols=None):
        """
        setParams(self, outputCols=None, value=0.0)
        Sets params for this SetValueTransformer.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)
      
    def setValue(self, value):
        """
        Sets the value of :py:attr:`value`.
        """
        return self._set(value=value)

    def getValue(self):
        """
        Gets the value of :py:attr:`value` or its default value.
        """
        return self.getOrDefault(self.value)
  
    def _transform(self,df):
      print("********************************  in Transform method ...************************************")
#       self = self.getCenterObject()
      
      
      """
      Generate feature column in dataframe based on specific logic

      @param - 
               df - dataframe for operation.

      @return - 
               df - dataframe with generated feature.
      """
      
#       def feature_generation(self,df):
#         print(self.df.show(2))
#         self.df = self.df.withColumn("Family_Size",col('SibSp')+col('Parch'))
#         self.df = self.df.withColumn('Alone',lit(0))
#         self.df = self.df.withColumn("Alone",when(self.df["Family_Size"] ==0, 1).otherwise(self.df["Alone"]))
#         return self.df
      
      
      def feature_generation(self,df):
        df = df.withColumn("Initial",regexp_extract(col("Name"),"([A-Za-z]+)\.",1))
        df = df.replace(['Mlle','Mme', 'Ms', 'Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],
                        ['Miss','Miss','Miss','Mr','Mr',  'Mrs',  'Mrs',  'Other',  'Other','Other','Mr','Mr','Mr'])
        df = df.withColumn("Family_Size",col('SibSp')+col('Parch'))
        df = df.withColumn('Alone',lit(0))
        df = df.withColumn("Alone",when(df["Family_Size"] ==0, 1).otherwise(df["Alone"]))
        return df


      """
      Impute Age based on Age mean of specific gender. ex for male mean is 46 update all null male row with 46, similarly for others

      @param - 
              df - dataframe for operation

      @return -
             df - with imputed value

      """
  
      def Age_impute(self,df):
        Age_mean = df.groupBy("Initial").avg('Age')
        Age_mean = Age_mean.withColumnRenamed('avg(Age)','mean_age')
        Initials_list = Age_mean.select("Initial").rdd.flatMap(lambda x: x).collect()
        Mean_list = Age_mean.select("mean_age").rdd.flatMap(lambda x: x).collect()
        for i,j in zip(Initials_list,Mean_list):
            df = df.withColumn("Age",when((df["Initial"] == i) & (df["Age"].isNull()), j).otherwise(df["Age"]))

        return df
        
        
      """
      Impute Embark based on mode of embark column
      @param - 
              df - dataframe for operation

      @return -
             df - with imputed value

      """
      def Embark_impute(self,df):
        mode_value = df.groupBy('Embarked').count().sort(col('count').desc()).collect()[0][0]
        df = df.fillna({'Embarked':mode_value})
        return df
      
      
      """
      Impute Fare based on the class which he/she had sat ex: class 3rd has mean fare 9 and null fare belong to 3rd class so fill 9
      @param - 
              df - dataframe for operation

      @return -
             df - with imputed value

      """
      def Fare_impute(self,df):
        Select_pclass = df.filter(col('Fare').isNull()).select('Pclass')
        if Select_pclass.count() > 0:
          Pclass = Select_pclass.rdd.flatMap(lambda x: x).collect()
          for i in Pclass:
            mean_pclass_fare = df.groupBy('Pclass').mean().select('Pclass','avg(Fare)').filter(col('Pclass')== i).collect()[0][1]
            df = df.withColumn("Fare",when((col('Fare').isNull()) & (col('Pclass') == i),mean_pclass_fare).otherwise(col('Fare')))
        return df
      
      
      '''
      combining all column imputation together..

      @param - 
            df - a dataframe for operation.

      @return - 
            df - dataframe with imputed value.

      '''
      def all_impute_together(df):
        df = Age_impute(self,df)
        df = Embark_impute(self,df)
        df = Fare_impute(self,df)
        return df
      
      
      '''
      converting string to numeric values.

      @param - 
               df - dataframe contained all columns.
               col_list - list of column need to be 

      @return - 
              df - transformed dataframe.
      '''
      def stringToNumeric_conv(df,col_list):
        indexer = [StringIndexer(inputCol=column,outputCol=column+"_index").fit(df) for column in col_list]
        string_change_pipeline = Pipeline(stages=indexer)
        df = string_change_pipeline.fit(df).transform(df)
        return df

      
      """
      Drop column from dataframe
      @param -
             df - dataframe 
             col_name - name of column which need to be dropped.
      @return -
             df - a dataframe except dropped column
      """
      def drop_column(df,col_list):
        for i in col_list:
            df = df.drop(col(i))
        return df
      
      
      col_list = ["Sex","Embarked","Initial"]
      dataset = feature_generation(self,df)
      df_impute = all_impute_together(dataset)
      df_numeric = stringToNumeric_conv(df_impute,col_list)
      df_final = drop_column(df_numeric,['Cabin','Name','Ticket','Family_Size','SibSp','Parch','Sex','Embarked','Initial'])
      return df_final
