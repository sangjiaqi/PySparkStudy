from pyspark import SparkConf, SparkContext
from pyspark.sql.context import SQLContext
from pyspark.sql.types import *

if __name__ == "__main__":

    # initialize the sparkconf
    conf = SparkConf().setMaster("local[*]").setAppName("Test3_1")

    sc = SparkContext(conf = conf)

    sqlContext = SQLContext(sc)

    # create json data
    stringJSONRDD = sc.parallelize((
        """{"id": "123", "name": "Katie", "age": 19, "eyeColor: "brown"}""",
        """{"id": "234", "name": "Michael", "age": 22, "eyeColor: "green"}""",
        """{"id": "345", "name": "Simone", "age": 23, "eyeColor": "blue"}"""
            ))

    # create dataFrame
    swimmersJSON = sqlContext.read.json(stringJSONRDD)

    # create register table
    swimmersJSON.registerTempTable("swimmersJSON")

    swimmersJSON.show

    swimmersJSON.sql("select * from swimmersJSON").collect()

    swimmersJSON.printSchema

    # schema reflection
    stringCSVRDD = sc.parallelize([
        (123, 'Katie', 19, 'brown'),
        (234, 'Michael', 22, 'green'),
        (345, 'Simone', 23, 'blue')
    ])

    schema = StructType([
        StructField("id", LongType, True),
        StructField("name", StringType, True),
        StructField("age", LongType, True),
        StructField("eyeColor", StringType, True)
    ])

    swimmers = sqlContext.createDataFrame(stringCSVRDD, schema)

    swimmers.registerTempTable("swimmers")

    swimmers.printSchema

    # API
    swimmers.count

    swimmers.select("id", "age").filter("age = 22").show

    swimmers.select(swimmers.id, swimmers.age).filter(swimmers.age == 22).show

    swimmers.select("name", "eyeColor").filter("eyeColor like 'b%'").show

    swimmers.sql("select count(1) from swimmers").show

    swimmers.sql("select id, age from swimmers where age = 22").show

    swimmers.sql("select name, eyeColor from swimmers where eyeColor like 'b%'").show

    # reality data
    flightPerfFilePath = "C:\\Users\\sangjiaqi\\Desktop\\软件\\Scala\\learningPySpark-master\\Chapter03\\flight-data\\departuredelays.csv"
    airportsFilePath = "C:\\Users\\sangjiaqi\\Desktop\\软件\\Scala\\learningPySpark-master\\Chapter03\\flight-data\\airport-codes-na.txt"

    airports = sqlContext.read.csv(airportsFilePath, header = 'true', inferSchema = 'true', sep = '\t')
    airports.createOrReplaceTempView("airports")

    flightPerf = sqlContext.read.csv(flightPerfFilePath, header = 'true')
    flightPerf.createOrReplaceTempView("FlightPerformance")

    flightPerf.cache

    sqlContext.sql("""
    select a.City,
    f.origin,
    sum(f.delay) as Delays
    from FlightPerformace f
    join airports a
    on a.IATA = f.orifin
    where a.State = 'WA'
    group by a.City, f.origin
    order by sum(f.delay) desc
    """).show
