import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

/**
  * Runs KMeans (Spark MLlib) on the MNist Dataset and prints the centroids for each cluster
  */

object KMeansScala {

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "C:\\hadoop")
    val sc = new SparkContext(new SparkConf().setAppName("KMeansScala").setMaster("local"))
    // Load data
    val data = sc.textFile("mnist_test.csv")
    val parsedData = data.map(s => Vectors.dense(s.split(',').map(_.toDouble))).cache()

    val numClusters = 10
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)

    val WSSSE = clusters.computeCost(parsedData)
    println(s"Within Set Sum of Squared Errors = $WSSSE")

    for(c <- clusters.clusterCenters){
      println(c)
    }
  }

}
