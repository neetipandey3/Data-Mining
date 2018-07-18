import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable
import scala.collection.mutable.Map
import scala.collection.mutable.ListBuffer
import java.io._
import scala.collection.mutable.Stack
import scala.collection.mutable.Queue
import scala.collection.mutable.HashMap



/**
  * Task: use ratings.csv to find betweenness and communities
  *
  * Usage:
  *
  * Step1: Download jar file. Download ratings.csv from MovieLens ml-latest-small dataset
  *
  * Step2: Execute:-
  *                 spark-submit --class betweenness.jar <-ratings.csv path>
  *
  * Used MovieLens Dataset:
  *                 ml-latest-small
  *
  */

object Betweenness {

  val spark = SparkSession
    .builder()
    .master("local[8]")
    .appName("MovieLensCommunityDetection")
    .getOrCreate()


  def main(args: Array[String]): Unit = {

    if (args.length != 1){
      println("Usage: bin/spark-submit --class Betweenness" + "target/scala/*Betweenness*.jar <input path>")
      sys.exit(1)
    }

    val conf = new SparkConf()
    conf.setAppName("MovieLensBetweenness")
    conf.setMaster("local[8]")
    conf.set("spark.driver.allowMultipleContexts", "true")

    val sc = new SparkContext(conf)

    val ratings_path = args(0)
    

    val data_hdr = sc.textFile(ratings_path+"/ratings.csv")
    val header = data_hdr.first()
    val data = data_hdr.filter(_(0) != header(0))

    val dataset = data.map {rows =>
      val cols = rows.split(",")

      (cols(0).toLong, cols(1).toLong)
    }.distinct().groupByKey().sortByKey().map(line => (line._1, line._2.toSet)).toLocalIterator.toList



    var u1 = 0
    var u2 = 1
    val edge_cutoff = 9
    val nodes = collection.mutable.Set.empty[Long]
    val edges = collection.mutable.Set.empty[(Long, Long)]

    while (u1 < dataset.size) {
      u2 = u1 + 1
      while (u2 < dataset.size) {
        val intxn = dataset.apply(u1)._2 intersect dataset.apply(u2)._2
        if (intxn.size >= edge_cutoff) {
          edges.add(u1 + 1, u2 + 1)
          edges.add(u2 + 1, u1 + 1)
          nodes.add(u2 + 1)
          nodes.add(u1 + 1)

        }

        u2 += 1
      }

      u1 += 1
    }


    val edges_rdd = sc.parallelize(edges.toList).collect().toList

    val edges_for_graph = sc.parallelize(edges.toList)

    //val graph = Graph.fromEdgeTuples(edges_for_graph, 1)

    val edges_dict : Map[Long, ListBuffer[Long]] = Map()
    for (edge <- edges_rdd) {
      if (edges_dict.contains(edge._1)) {
        edges_dict.apply(edge._1) += edge._2
      }
      else {
        val listBuffer = ListBuffer.empty[Long]
        listBuffer += edge._2
        edges_dict += (edge._1 -> listBuffer)
      }
    }

    val betweeness = get_betweenness(nodes.toList, edges.toList)

    val sorted_edge = betweeness.map(edge => {
      if (edge._1._2 < edge._1._1) {
        (edge._1.swap, edge._2)
      } else {
        (edge._1, edge._2)
      }
    }).toList.sorted
    val writer = new PrintWriter(new File("Neeti_Pandey_Betweenness.txt"))
    for (res <- sorted_edge) {
      val user_1 = res._1._1
      val user_2 = res._1._2
      val betwn = res._2
      //println("Output(" + user_1 +"," + user_2 + "," + betwn + ")" + '\n')
      writer.append("("+user_1 +"," + user_2 + "," + betwn + ")\n")
    }

    writer.flush
    writer.close()



    spark.stop()
    sc.stop
  }



  def get_neighbours(node_list: List[Long], edge_list: List[(Long, Long)]): HashMap[Long, List[Long]] =
  {
    val neighbours = new HashMap[Long, List[Long]]()

    node_list.foreach { case(v) =>
    {val neighb = (edge_list.filter{case(edge) => ((edge._1 == v) || (edge._2 == v))})
      .map{case(edge) => {if (edge._1 == v) edge._2 else edge._1 }}
      neighbours.+=((v, neighb.distinct))
    }
    }

    neighbours
  }


  def traverse_and_calc(root: Long, betweenness: mutable.Map[(Long, Long), Double], neighbours: HashMap[Long, List[Long]]):  mutable.Map[(Long, Long), Double] =
  {
    //println("traverse")
    val s = Stack[Long]()
    val q = Queue[Long]()

    val dist = mutable.Map[Long, Double]()
    val visited = mutable.Map[Long, Boolean]()
    val credits = mutable.Map[Long, Double]()
    val predecessors = mutable.Map[Long, ListBuffer[Long]]()

    //val betweenness = new ListBuffer[(Long, Double)]()
    //var betweenness = new HashMap[(Long, Long), Double]()
    q.enqueue(root)
    dist.put(root, 0.0)
    visited.put(root, true)


    for (node <- neighbours(root))
    {
      dist += (node -> 1.0)
      visited += (node -> true)
      q.enqueue(node)
      predecessors += (node -> ListBuffer(root))
    }

    var i = 1
    while (i < q.size) {
      //println("Queue iteration")
      val node = q(i)
      s.push(node)
      val distance = dist.apply(node)
      visited += (node -> true)
      for (child <- neighbours(node)) {
        if (!visited.contains(child)) {
          if (!dist.contains(child)) {
            q.enqueue(child)
            dist += (child -> (distance + 1.0))
            predecessors += (child -> ListBuffer(node))
          }
          else {
            if (dist(child) > distance) {
              predecessors.update(child, (predecessors.apply(child) += node))
              dist.update(child, (distance + 1.0))
            }
          }
        }
      }
      i += 1
    }

    while (!(s.isEmpty)) {
      //println("stck iteration")
      val node = s.pop()
      if (!credits.contains(node)){
        credits += (node -> 1.0)
      }
      if (predecessors(node).size != 0){
        val weight = 1.0 / predecessors(node).size
        for (parent <- predecessors(node)) {
          if (!credits.contains(parent)){
            credits += (parent -> 1.0)
          }
          val credit = (weight * credits.apply(node))
          //println("weight: " + weight+ " credit:" + credits(node))
          //println("betwn: " + credit)
          credits.update(parent,  (credits.apply(parent) + credit))

          if (parent < node) {
            val edge = (parent, node)
            if (betweenness.contains(edge)) {
              betweenness.update(edge,  (betweenness.apply(edge) + credit))
            }
            else {
              betweenness += (edge -> credit)
            }
          }
        }
      }
    }

    betweenness
  }


  def get_betweenness(nodes_list: List[Long], edges_list: List[(Long, Long)]): Map[(Long, Long), BigDecimal] = {
    //println("\nget_betweeenness")
    var betweenness = mutable.Map[(Long, Long), Double]()
    var betweeness_res: Map[(Long, Long), BigDecimal] = Map()
    val neighbours : HashMap[Long, List[Long]] = get_neighbours(nodes_list, edges_list)
    val nodes = nodes_list.sorted
    for (root_node <- nodes) {

      betweenness = traverse_and_calc(root_node, betweenness, neighbours)

      }

    for (elem <- betweenness) {
      betweeness_res += (elem._1 -> BigDecimal(elem._2 / 2))
    }

    betweeness_res
  }
}

