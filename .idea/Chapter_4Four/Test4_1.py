from pyspark import SparkConf, SparkContext
from pyspark.sql.context import SQLContext, HiveContext
import pyspark.sql.functions as fn
import pyspark.sql.types as typ

if __name__ == "__main__":

    # initialize the spark
    conf = SparkConf().setMaster("local[*]").setAppName("Test4_1")

    sc = SparkContext(conf = conf)

    sqlContext = SQLContext(sc)

    # create DataFrame duplicates
    df = sqlContext.createDataFrame([
        (1, 144.5, 5.9, 33, 'M'),
        (2, 167.2, 5.4, 45, 'M'),
        (3, 124.1, 5.2, 23, 'F'),
        (4, 144.5, 5.9, 33, 'M'),
        (5, 124.1, 5.2, 54, 'F'),
        (3, 124.1, 5.2, 23, 'F'),
        (5, 129.2, 5.3, 42, 'M')
    ], ['id', 'weight', 'height', 'age', 'gender'])

    print('Count of rows: {0}'.format(df.count()))
    print('Count of distinct rows: {0}'.format(df.distinct().count()))

    # duplicated
    df_drop = df.dropDuplicates()

    print('Count of ids: {0}'.format(df_drop.count()))
    print('Count of distinct ids: {0}'.format(
        df.select([
            c for c in df_drop.columns if c != 'id'
        ]).distinct().count()
    ))

    df_drop2 = df_drop.dropDuplicates(subset = [
        c for c in df.columns if c != 'id'
    ])

    df_drop2.agg(
        fn.count('id').alias('count'),
        fn.countDistinct('id').alias('distinct')
    ).show()

    df_drop2.withColumn('new_id', fn.monotonically_increasing_id()).show()

    # create data na
    df_miss = sqlContext.createDataFrame([
        (1, 143.5, 5.6, 28, 'M', 100000),
        (2, 167.2, 5.4, 45, 'M', None),
        (3, None, 5.2, None, None, None),
        (4, 144.5, 5.9, 33, 'M', None),
        (5, 124.1, 5.2, 54, 'F', None),
        (3, 124.1, 5.2, None, 'F', None),
        (5, 129.2, 5.3, 42, 'M', 76000)
    ], ['id', 'weight', 'height', 'age', 'gender', 'income'])

    df_miss.rdd.map(
        lambda row: (row['id'], sum([c == None for c in row]))
    ).collect()

    df_miss.where('id == 3').show()

    df_miss.agg(*[
        (1 - (fn.count(c) / fn.count('*'))) for c in df_miss.columns
    ]).show()

    df_miss_no_income = df_miss.select([
        c for c in df_miss.columns if c != 'income'
    ])

    df_miss_no_income.dropna(thread=3).show()

    means = df_miss_no_income.agg(
        *[fn.mean(c).alias(c)
          for c in df_miss_no_income.columns if c != 'gender']
    ).toPandas().to_dict('records')[0]

    means['gender'] = 'missing'

    df_miss_no_income.fillna(means).show()

    # create data special
    df_outliers = sqlContext.createDataFrame([
        (1, 143.5, 5.3, 28),
        (2, 154.2, 5.5, 45),
        (3, 342.1, 5.1, 99),
        (4, 144.5, 5.5, 33),
        (5, 133.1, 5.4, 54),
        (3, 124.1, 5.1, 21),
        (5, 129.2, 5.3, 42)
    ], ['id', 'weight', 'height', 'age'])

    cols = ['weight', 'height', 'age']
    bounds = {}

    for col in cols:
        quantiles = df_outliers.approxQuantile(
            col, [0.25, 0.75], 0.05
        )

        IQR = quantiles[1] - quantiles[0]

        bounds[col] = [
            quantiles[0] - 1.5 * IQR,
            quantiles[1] + 1.5 * IQR
        ]

    outliers = df_outliers.select(*['di'] + [
        (
            (df_outliers[c] < bounds[c][0]) |
            (df_outliers[c] > bounds[c][1])
        ).alias(c + '_o') for c in cols
    ])

    outliers.show()
    df_outliers = df_outliers.join(outliers, on = 'id')
    df_outliers.filter('weight_o').select('id', 'weight').show()
    df_outliers.filter('age_o').select('id', 'age').show()

    fraud = sc.textFile('ccFraud.csv.gz')
    header = fraud.first()

    fraud = fraud.filter(lambda row: row != header).map(lambda row: [int(elem) for elem in row.split(',')])

    # fields = [
    #     *[
    #         typ.StructFieldh(h[1:-1], typ.IntegerType(), True)
    #         for h in header.split(',')
    #     ]
    # ]
    # schema = typ.StringType(fields)
    #
    # fraud_df = sqlContext.createDataFrame(fraud, schema)
    #
    # fraud_df.groupby('gender').count().show()
    #
    # numerical = ['balance', 'numTrans', 'numIntlTrans']
    # desc = fraud_df.describe(numberical)
    # desc.show()
    #
    # fraud_df.agg({'balance': 'skewness'}).show()