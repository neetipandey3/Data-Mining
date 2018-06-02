import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.types._
import org.apache.spark.sql._

/**
  * Task1: use ratings.csv to find movie average rating by movieId
  *
  * Usage:
  *
  * Step1: Download jar file
  *
  * Step2: Execute:-
  *                 spark-submit --class Task1 movie_avg_rating.jar <-ratings.csv path> <-ratings.csv path> <-output path>
  *
  * Used MovieLens Dataset:
  *                 ml-latest-small and ml-20m
  *
  */

object Task1 {

  val spark = SparkSession
    .builder()
    .master("local[1]")
    .appName("MovieLensAvgRatingSession")
    .getOrCreate()

  import spark.implicits._
  case class Ratings(movieId: String, avg_rating: String)

  def main(args: Array[String]): Unit = {

    if (args.length != 2){
      println("Usage: bin/spark-submit --class Task1" + "target/scala/*Task1*.jar <input path> <output path>")
      sys.exit(1)
    }

    val conf = new SparkConf()
    conf.setAppName("MovieLensAvgRating")
    conf.setMaster("local[*]")
    conf.set("spark.driver.allowMultipleContexts", "true")

    val sc = new SparkContext(conf)

    val ratings_path = args(0)
    //val op_path = args(1)
    val op_filename = "result_task1.csv"

    val data_hdr = sc.textFile(ratings_path+"/ratings.csv")
    val header = data_hdr.first()
    val data = data_hdr.filter(_(0) != header(0))

    val movie_ratings = data.map { rows =>
      val cols = rows.toString.split(",")

      (cols(1).toInt, cols(2).toFloat)
    }

    val ratings_sumcount = movie_ratings.aggregateByKey((0.0, 0.0))(
      (x, y) => (x._1 + y, x._2 + 1),
      (rdd1, rdd2) => (rdd1._1 + rdd2._1, rdd1._2 + rdd2._2)
    )

    val ratings_avg = ratings_sumcount.map( x => (x._1.toInt, x._2._1/x._2._2))


    val results = ratings_avg.sortByKey(ascending = true)

    //val dfSchema = Seq("movieId", "avg_rating")
    //results.toDF(dfSchema: _*).coalesce(1).write.format("csv").mode("overwrite")
      //.option("header", "true")
      //.csv(args(1)+"/result_task1.csv")
    //  val df = spark.createDataFrame(results).toDF("movieId", "avg_rating")

    //df.show(10)


    //df.coalesce(1).write
    //df.coalesce(1).write
      //.option("header", "true")
      //.format("com.databricks.spark.csv")
      //.mode("overwrite")
      //.save(args(1))



    // Writing the ouptput to one single text file instead of splitting
    // Hence, using coalesce(1)

    results.map{case(movieId, avg_rating) =>
      var line = movieId.toString + "," + avg_rating.toString
      line
    }.coalesce(1).saveAsTextFile(args(1));

    println("Done with file creation")


    spark.stop()

    sc.stop


  } }

