/**
  * Word Count
  *
  * Arguments: <Input File> <Output File>
  */

import org.apache.spark.{SparkConf, SparkContext}

object WordCount {

  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setAppName("MovieLensAvgRating")
    conf.setMaster("local[*]")
    conf.set("spark.driver.allowMultipleContexts", "true")

    val sc = new SparkContext(conf)


    // Load the data
    val readData = sc.textFile(args(0))

    // Generate list of words
    val wordsList = readData.flatMap(line =>
    line.split("\\W+"))
      .map(each => each.toLowerCase)

    // Create tuples (Word, 1)
    val tuples = wordsList.map(word =>
      (word, 1))

    val wordsCount = tuples.reduceByKey((w1, w2) => (w1+w2)).sortByKey(true)

    wordsCount.map{case(word, count) =>
      var line = word.toString + "," + count.toString
      line
    }.coalesce(1).saveAsTextFile(args(1))


    sc.stop()


  }

}
