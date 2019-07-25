from pyspark import SparkConf, SparkContext

if __name__ == "__main__":

    conf = SparkConf().setMaster("local[*]").setAppName("My App")

    sc = SparkContext(conf = conf)

    lines = sc.parallelize(["pandas", "cat", "i like pandas"])
    word = lines.filter(lambda s: "pandas" in s)
    print(word.count())