from pyspark import SparkConf, SparkContext
from pyspark.sql.context import SQLContext, HiveContext
import pyspark.sql.types as typ
import pyspark.ml.feature as ft
import pyspark.ml.classification as cl
from pyspark.ml import Pipeline
import pyspark.ml.evaluation as ev
from pyspark.ml import PipelineModel
import pyspark.ml.tuning as tune

if __name__ == "__main__":

    # create spark context
    conf = SparkConf().setMaster("local[*]").setAppName("Test6_1")

    sc = SparkContext(conf = conf)

    sqlContext = SQLContext(sc)

    hiveContext = HiveContext(sc)

    # create DataFrame
    labels = [
        ('INFANT_ALIVE_AT_REPORT', typ.IntegerType()),
        ('BIRTH_PLACE', typ.StringType()),
        ('MOTHER_AGE_YEARS', typ.IntegerType()),
        ('FATHER_COMBINED_AGE', typ.IntegerType()),
        ('CIG_BEFORE', typ.IntegerType()),
        ('CIG_1_TRI', typ.IntegerType()),
        ('CIG_2_TRI', typ.IntegerType()),
        ('CIG_3_TRI', typ.IntegerType()),
        ('MOTHER_HEIGHT_IN', typ.IntegerType()),
        ('MOTHER_PRE_WEIGHT', typ.IntegerType()),
        ('MOTHER_DELIVERY_WEIGHT', typ.IntegerType()),
        ('MOTHER_WEIGHT_GAIN', typ.IntegerType()),
        ('DIABETES_PRE', typ.IntegerType()),
        ('DIABETES_GEST', typ.IntegerType()),
        ('HYP_TENS_PRE', typ.IntegerType()),
        ('HYP_TENS_GEST', typ.IntegerType()),
        ('PREV_BIRTH_PRETERM', typ.IntegerType())
    ]

    schema = typ.StructType([typ.StructField(e[0], e[1], False) for e in labels])

    births = spark.read.csv('births_transformed.csv.gz', header=True, schema=schema)

    # transform
    births = births.withColumns('BIRTH_PLACE_INT', births['BIRTH_PLACE'].cast(typ.IntegerType()))

    encoder = ft.OneHotEncoder(inputCol='BIRTH_PLACE_INT', outputCol='BIRTH_PLACE_VEC')

    featuresCreate = ft.VectorAssembler(inputCols=[col[0] for col in labels[2:]], outputCol='features')

    # model
    logistic = cl.LogisticRegression(
        maxIter=10,
        regParam=0.01,
        labelCol='INFANT_ALIVE_AT_REPORT'
    )

    # pipeline
    pipeline = Pipeline(stages=[
        encoder, featuresCreate, logistic
    ])

    # fit model
    births_train, births_test = births.randomSplit([0.7, 0.3], seed=666)

    model = pipeline.fit(births_train)
    test_model = model.transform(births_test)

    test_model.take(1)

    # evaluator
    evaluator = ev.BinaryClassificationEvaluator(
        rawPredictionCol='probability',
        labelCol='INFANT_ALIVE_AT_REPORT'
    )

    print(evaluator.evaluate(test_model, {evaluator.metricName: 'areaUnderROC'}))
    print(evaluator.evaluate(test_model, {evaluator.metricName: 'areaUnderPR'}))

    # save model
    pipelinePath = './infant_oneHotEncoder_Logistic_Pipeline'
    pipeline.write().overwrite().save(pipelinePath)

    loadedPipeline = Pipeline.load(pipelinePath)
    loadedPipeline.fit(births_train).transform(births_test).take(1)

    modelPath = './infant_oneHotEncoder_Logistic_Model'
    model.write().overwrite().save(modelPath)

    loadedModel = PipelineModel.load(modelPath)
    loadedModel.transform(births_test)

    # cv
    ## cvmodel
    logistic = cl.LogisticRegression(
        labelCol='INFANT_ALIVE_AT_REPORT'
    )

    grid = tune.ParamGridBuilder().addGrid(logistic.maxIter, [2, 10, 50]).addGrid(logistic.regParam, [0.01, 0.05, 0.3]).build()

    evaluator = ev.BinaryClassificationEvaluator(
        rawPredictionCol='probability',
        evaluator=evaluator
    )

    cv = tune.CrossValidator(
        estimator=logistic,
        estimatorParamMaps=grid,
        evaluator=evaluator
    )

    pipeline = Pipeline(stages=[encoder, featuresCreate])
    data_transformer = pipeline.fit(births_train)

    cvModel = cv.fit(data_transformer.transform(births_train))

    data_train = data_transformer.transform(births_test)

    results = cvModel.transform(births_test)

    print(evaluator.evaluate(results, {evaluator.metricName: 'areaUnderROC'}))
    print(evaluator.evaluate(results, {evaluator.metricName: 'areaUnderPR'}))





