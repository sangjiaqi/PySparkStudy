from pyspark import SparkConf, SparkContext
from pyspark.sql.context import SQLContext, HiveContext
import pyspark.sql.types as typ

if __name__ == "__main__":

    conf = SparkConf().setMaster("local[*]").setAppName("Test7_1")

    sc = SparkContext(conf = conf)

    sqlContext = SQLContext(sc)

    hiveContext = HiveContext(sc)

    labels = [
        ('INFANT_ALIVE_AT_REPORT', typ.StringType()),
        ('BIRTH_YEAR', typ.IntegerType()),
        ('BIRTH_MONTH', typ.IntegerType()),
        ('BIRTH_PLACE', typ.StringType()),
        ('MOTHER_AGE_YEARS', typ.IntegerType()),
        ('MOTHER_RACE_6CODE', typ.StringType()),
        ('MOTHER_EDUCATION', typ.StringType()),
        ('FATHER_COMBINED_AGE', typ.IntegerType()),
        ('FATHER_EDUCATION', typ.StringType()),
        ('MONTH_PRECARE_RECODE', typ.StringType()),
        ('CIG_BEFORE', typ.IntegerType()),
        ('CIG_1_TRI', typ.IntegerType()),
        ('CIG_2_TRI', typ.IntegerType()),
        ('CIG_3_TRI', typ.IntegerType()),
        ('MOTHER_HEIGHT_IN', typ.IntegerType()),
        ('MOTHER_BMI_RECODE', typ.IntegerType()),
        ('MOTHER_PRE_WEIGHT', typ.IntegerType()),
        ('MOTHER_DELIVERY_WEIGHT', typ.IntegerType()),
        ('MOTHER_WEIGHT_GAIN', typ.IntegerType()),
        ('DIABETES_PRE', typ.StringType()),
        ('DIABETES_GEST', typ.StringType()),
        ('HYP_TENS_PRE', typ.StringType()),
        ('HYP_TENS_GEST', typ.StringType()),
        ('PREV_BIRTH_PRETERM', typ.StringType()),
        ('NO_RISK', typ.StringType()),
        ('NO_INFECTIONS_REPORTED', typ.StringType()),
        ('LABOR_IND', typ.StringType()),
        ('LABOR_AUGM', typ.StringType()),
        ('STEROIDS', typ.StringType()),
        ('ANTIBIOTICS', typ.StringType()),
        ('ANESTHESIA', typ.StringType()),
        ('DELIV_METHOD_RECODE_COMB', typ.StringType()),
        ('ATTENDANT_BIRTH', typ.StringType()),
        ('APGAR_5', typ.IntegerType()),
        ('APGAR_5_RECODE', typ.StringType()),
        ('APGAR_10', typ.IntegerType()),
        ('APGAR_10_RECODE', typ.StringType()),
        ('INFANT_SEX', typ.StringType()),
        ('OBSTETRIC_GESTATION_WEEKS', typ.IntegerType()),
        ('INFANT_WEIGHT_GRAMS', typ.IntegerType()),
        ('INFANT_ASSIST_VENTI', typ.StringType()),
        ('INFANT_ASSIST_VENTI_6HRS', typ.StringType()),
        ('INFANT_NICU_ADMISSION', typ.StringType()),
        ('INFANT_SURFACANT', typ.StringType()),
        ('INFANT_ANTIBIOTICS', typ.StringType()),
        ('INFANT_SEIZURES', typ.StringType()),
        ('INFANT_NO_ABNORMALITIES', typ.StringType()),
        ('INFANT_ANCEPHALY', typ.StringType()),
        ('INFANT_MENINGOMYELOCELE', typ.StringType()),
        ('INFANT_LIMB_REDUCTION', typ.StringType()),
        ('INFANT_DOWN_SYNDROME', typ.StringType()),
        ('INFANT_SUSPECTED_CHROMOSOMAL_DISORDER', typ.StringType()),
        ('INFANT_NO_CONGENITAL_ANOMALIES_CHECKED', typ.StringType()),
        ('INFANT_BREASTFED', typ.StringType())
    ]

    schema = typ.StructType([
                                typ.StructField(e[0], e[1], False) for e in labels
                                ])

    births = sqlContext.read.csv('births_train.csv.gz',
                            header=True,
                            schema=schema)

    recode_dictionary = {
        'YNU': {
            'Y': 1,
            'N': 0,
            'U': 0
        }
    }

    selected_features = [
        'INFANT_ALIVE_AT_REPORT',
        'BIRTH_PLACE',
        'MOTHER_AGE_YEARS',
        'FATHER_COMBINED_AGE',
        'CIG_BEFORE',
        'CIG_1_TRI',
        'CIG_2_TRI',
        'CIG_3_TRI',
        'MOTHER_HEIGHT_IN',
        'MOTHER_PRE_WEIGHT',
        'MOTHER_DELIVERY_WEIGHT',
        'MOTHER_WEIGHT_GAIN',
        'DIABETES_PRE',
        'DIABETES_GEST',
        'HYP_TENS_PRE',
        'HYP_TENS_GEST',
        'PREV_BIRTH_PRETERM'
    ]

    births_trimmed = births.select(selected_features)

    import pyspark.sql.functions as func

    def recode(col, key):
        return recode_dictionary[key][col]

    def correct_cig(feat):
        return func \
            .when(func.col(feat) != 99, func.col(feat)) \
            .otherwise(0)

    rec_integer = func.udf(recode, typ.IntegerType())

    births_transformed = births_trimmed \
        .withColumn('CIG_BEFORE', correct_cig('CIG_BEFORE')) \
        .withColumn('CIG_1_TRI', correct_cig('CIG_1_TRI')) \
        .withColumn('CIG_2_TRI', correct_cig('CIG_2_TRI')) \
        .withColumn('CIG_3_TRI', correct_cig('CIG_3_TRI'))

    cols = [(col.name, col.dataType) for col in births_trimmed.schema]

    YNU_cols = []

    for i, s in enumerate(cols):
        if s[1] == typ.StringType():
            dis = births.select(s[0]) \
                .distinct() \
                .rdd \
                .map(lambda row: row[0]) \
                .collect()

            if 'Y' in dis:
                YNU_cols.append(s[0])

    births.select([
        'INFANT_NICU_ADMISSION',
        rec_integer(
            'INFANT_NICU_ADMISSION', func.lit('YNU')
        ) \
            .alias('INFANT_NICU_ADMISSION_RECODE')]
    ).take(5)

    exprs_YNU = [
        rec_integer(x, func.lit('YNU')).alias(x)
        if x in YNU_cols
        else x
        for x in births_transformed.columns
        ]

    births_transformed = births_transformed.select(exprs_YNU)

    births_transformed.select(YNU_cols[-5:]).show(5)

    # show the data describe
    import pyspark.mllib.stat as st
    import numpy as np

    numeric_cols = ['MOTHER_AGE_YEARS','FATHER_COMBINED_AGE',
                    'CIG_BEFORE','CIG_1_TRI','CIG_2_TRI','CIG_3_TRI',
                    'MOTHER_HEIGHT_IN','MOTHER_PRE_WEIGHT',
                    'MOTHER_DELIVERY_WEIGHT','MOTHER_WEIGHT_GAIN'
                    ]

    numeric_rdd = births_transformed \
        .select(numeric_cols) \
        .rdd \
        .map(lambda row: [e for e in row])

    mllib_stats = st.Statistics.colStats(numeric_rdd)

    for col, m, v in zip(numeric_cols,
                         mllib_stats.mean(),
                         mllib_stats.variance()):
        print('{0}: \t{1:.2f} \t {2:.2f}'.format(col, m, np.sqrt(v)))

    categorical_cols = [e for e in births_transformed.columns
                        if e not in numeric_cols]

    categorical_rdd = births_transformed \
        .select(categorical_cols) \
        .rdd \
        .map(lambda row: [e for e in row])

    for i, col in enumerate(categorical_cols):
        agg = categorical_rdd \
            .groupBy(lambda row: row[i]) \
            .map(lambda row: (row[0], len(row[1])))

        print(col, sorted(agg.collect(),
                          key=lambda el: el[1],
                          reverse=True))

    # show the corr
    corrs = st.Statistics.corr(numeric_rdd)

    for i, el in enumerate(corrs > 0.5):
        correlated = [
            (numeric_cols[j], corrs[i][j])
            for j, e in enumerate(el)
            if e == 1.0 and j != i]

        if len(correlated) > 0:
            for e in correlated:
                print('{0}-to-{1}: {2:.2f}' \
                      .format(numeric_cols[i], e[0], e[1]))

    features_to_keep = [
        'INFANT_ALIVE_AT_REPORT',
        'BIRTH_PLACE',
        'MOTHER_AGE_YEARS',
        'FATHER_COMBINED_AGE',
        'CIG_1_TRI',
        'MOTHER_HEIGHT_IN',
        'MOTHER_PRE_WEIGHT',
        'DIABETES_PRE',
        'DIABETES_GEST',
        'HYP_TENS_PRE',
        'HYP_TENS_GEST',
        'PREV_BIRTH_PRETERM'
    ]

    births_transformed = births_transformed.select([e for e in features_to_keep])

    # Statistical testing
    import pyspark.mllib.linalg as ln

    for cat in categorical_cols[1:]:
        agg = births_transformed \
            .groupby('INFANT_ALIVE_AT_REPORT') \
            .pivot(cat) \
            .count()

        agg_rdd = agg \
            .rdd \
            .map(lambda row: (row[1:])) \
            .flatMap(lambda row:
                     [0 if e == None else e for e in row]) \
            .collect()

        row_length = len(agg.collect()[0]) - 1
        agg = ln.Matrices.dense(row_length, 2, agg_rdd)

        test = st.Statistics.chiSqTest(agg)
        print(cat, round(test.pValue, 4))

    # create the model
    import pyspark.mllib.feature as ft
    import pyspark.mllib.regression as reg

    hashing = ft.HashingTF(7)

    births_hashed = births_transformed \
        .rdd \
        .map(lambda row: [
        list(hashing.transform(row[1]).toArray())
        if col == 'BIRTH_PLACE'
        else row[i]
        for i, col
        in enumerate(features_to_keep)]) \
        .map(lambda row: [[e] if type(e) == int else e
                          for e in row]) \
        .map(lambda row: [item for sublist in row
                          for item in sublist]) \
        .map(lambda row: reg.LabeledPoint(
        row[0],
        ln.Vectors.dense(row[1:]))
             )

    births_train, births_test = births_hashed.randomSplit([0.6, 0.4])

    from pyspark.mllib.classification \
        import LogisticRegressionWithLBFGS

    LR_Model = LogisticRegressionWithLBFGS \
        .train(births_train, iterations=10)

    LR_results = (
        births_test.map(lambda row: row.label) \
            .zip(LR_Model \
                 .predict(births_test \
                          .map(lambda row: row.features)))
    ).map(lambda row: (row[0], row[1] * 1.0))

    import pyspark.mllib.evaluation as ev
    LR_evaluation = ev.BinaryClassificationMetrics(LR_results)

    print('Area under PR: {0:.2f}' \
          .format(LR_evaluation.areaUnderPR))
    print('Area under ROC: {0:.2f}' \
          .format(LR_evaluation.areaUnderROC))
    LR_evaluation.unpersist()

    selector = ft.ChiSqSelector(4).fit(births_train)

    topFeatures_train = (
        births_train.map(lambda row: row.label) \
            .zip(selector \
                 .transform(births_train \
                            .map(lambda row: row.features)))
    ).map(lambda row: reg.LabeledPoint(row[0], row[1]))

    topFeatures_test = (
        births_test.map(lambda row: row.label) \
            .zip(selector \
                 .transform(births_test \
                            .map(lambda row: row.features)))
    ).map(lambda row: reg.LabeledPoint(row[0], row[1]))

    from pyspark.mllib.tree import RandomForest

    RF_model = RandomForest \
        .trainClassifier(data=topFeatures_train,
                         numClasses=2,
                         categoricalFeaturesInfo={},
                         numTrees=6,
                         featureSubsetStrategy='all',
                         seed=666)

    RF_results = (
        topFeatures_test.map(lambda row: row.label) \
            .zip(RF_model \
                 .predict(topFeatures_test \
                          .map(lambda row: row.features)))
    )

    RF_evaluation = ev.BinaryClassificationMetrics(RF_results)

    print('Area under PR: {0:.2f}' \
          .format(RF_evaluation.areaUnderPR))
    print('Area under ROC: {0:.2f}' \
          .format(RF_evaluation.areaUnderROC))
    RF_evaluation.unpersist()

    LR_Model_2 = LogisticRegressionWithLBFGS \
        .train(topFeatures_train, iterations=10)

    LR_results_2 = (
        topFeatures_test.map(lambda row: row.label) \
            .zip(LR_Model_2 \
                 .predict(topFeatures_test \
                          .map(lambda row: row.features)))
    ).map(lambda row: (row[0], row[1] * 1.0))

    LR_evaluation_2 = ev.BinaryClassificationMetrics(LR_results_2)

    print('Area under PR: {0:.2f}' \
          .format(LR_evaluation_2.areaUnderPR))
    print('Area under ROC: {0:.2f}' \
          .format(LR_evaluation_2.areaUnderROC))
    LR_evaluation_2.unpersist()