import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Task2: use ratings.csv and tags.csv to find movie average rating by tags
  *
  * Usage:
  *
  * Step1: Download jar file
  *
  * Step2: Execute:
  *                 spark-submit --class Task2 movie_avg_rating.jar <-ratings.csv path> <-ratings.csv path> <-tags.csv path> <-output path>
  *
  * Used MovieLens Dataset
  *
  */

object Task2 {

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("MovieLensAvgRating")
    .getOrCreate()

  def main(args: Array[String]): Unit = {

    if (args.length != 3){
      println("Usage: bin/spark-submit --class Task1" + "target/scala/*Task1*.jar <input path> <output path>")
      sys.exit(1)
    }


    val conf = new SparkConf()
    conf.setAppName("MovieLensAvgRating")
    conf.setMaster("local[*]")
    conf.set("spark.driver.allowMultipleContexts", "true")

    val sc = new SparkContext(conf)

    val ratings_path = args(0)
    val tags_path = args(1)
    val op_filename = "/result_task1.csv"


    // Remove headers from the csv
    val ratings_with_hdr = sc.textFile(ratings_path+"/ratings.csv")
    val tags_with_hdr = sc.textFile(tags_path+"tags.csv")
    val ratings_hdr = ratings_with_hdr.first()
    val ratings = ratings_with_hdr.filter(_(0) != ratings_hdr(0))
    val tags_hdr = tags_with_hdr.first()
    val tags = tags_with_hdr.filter(_(0) != tags_hdr(0))

    val movie_ratings = ratings.map { rows =>
      val cols = rows.toString.split(",")

      (cols(1).toInt, cols(2).toFloat)
    }

    val tags_movie = tags.map { rows =>
      val cols = rows.toString.split(",")

      (cols(1).toInt, cols(2))
    }

    val tag_join_ratings = tags_movie.join(movie_ratings)

    val tag_ratings = tag_join_ratings.map(row => (row._2._1.toString, row._2._2))

    val ratings_sumcount = tag_ratings.aggregateByKey((0.0, 0.0))(
      (x, y) => (x._1 + y, x._2 + 1),
      (rdd1, rdd2) => (rdd1._1 + rdd2._1, rdd1._2 + rdd2._2)
    )

    val ratings_avg = ratings_sumcount.map( x => (x._1, x._2._1/x._2._2))

    val results = ratings_avg.sortByKey(ascending = false)

    // Writing the ouptput to one single text file instead of splitting
    // Hence, using coalesce(1)
    results.map{case(a, b) =>
      var line = a.toString + "," + b.toString
      line
    }.coalesce(1).saveAsTextFile(args(2));

    println("Done with file creation")


    spark.stop()
    sc.stop()


  } }

