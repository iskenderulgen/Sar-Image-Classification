package Test_Environment;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.tools.cmd.gen.AnyVals;

import java.util.List;

public class Union {
    private static final String Main_Path = "C:\\Users\\ULGEN\\Documents\\Idea_Workspace\\Sar_Cluster\\set\\deneme\\";
    private static final String Sar_image_set1 = Main_Path + "toplam1.csv";
    private static final String Sar_image_set2 = Main_Path + "toplam2.csv";
    private static final String Sar_image_set3 = Main_Path + "toplam3.csv";
    private static final String Sar_image_set4 = Main_Path + "toplam4.csv";
    private static final String Sar_image_set5 = Main_Path + "toplam5.csv";
    private static final String Sar_image_set6 = Main_Path + "toplam6.csv";
    private static final String Sar_image_set7 = Main_Path + "toplam7.csv";
    private static final String Sar_image_set8 = Main_Path + "toplam8.csv";
    private static final String Sar_image_set9 = Main_Path + "toplam9.csv";


    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("Test_Environment.Union");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        jsc.setLogLevel("ERROR");


    /*    JavaRDD<String> set = jsc.textFile("C:\\Users\\ULGEN\\Documents\\Idea_Workspace\\Sar_Cluster\\set\\toplam1_union2.csv");
        System.out.println(set.count());
        List<String> liste =  set.takeSample(false,1000000);
        JavaRDD<String> rdd2 = jsc.parallelize(liste);
        rdd2.coalesce(1).saveAsTextFile("C:\\Users\\ULGEN\\Documents\\Idea_Workspace\\Sar_Cluster\\set\\deneme2");*/


        JavaRDD<String> set1 = jsc.textFile(Sar_image_set1);
        JavaRDD<String> set2 = jsc.textFile(Sar_image_set2);
        JavaRDD<String> set3 = jsc.textFile(Sar_image_set3);
        JavaRDD<String> set4 = jsc.textFile(Sar_image_set4);
        JavaRDD<String> set5 = jsc.textFile(Sar_image_set5);
        JavaRDD<String> set6 = jsc.textFile(Sar_image_set6);
        JavaRDD<String> set7 = jsc.textFile(Sar_image_set7);
        JavaRDD<String> set8 = jsc.textFile(Sar_image_set8);
        JavaRDD<String> set9 = jsc.textFile(Sar_image_set9);


        JavaRDD<String> total_set = set1.
                union(set2).
                union(set3).
                union(set4).
                union(set5).
                union(set6).
                union(set7).
                union(set8).
                union(set9);


        long toplam = set1.count()+
                set2.count()+
                set3.count()+
                set5.count()+
                set6.count()+
                set7.count()+
                set8.count()+
                set9.count();

        System.out.println(set1.count());
        System.out.println(set2.count());
        System.out.println(set3.count());
        System.out.println(set4.count());
        System.out.println(set5.count());
        System.out.println(set6.count());
        System.out.println(set7.count());
        System.out.println(set8.count());
        System.out.println(set9.count());
        System.out.println(total_set.count());
        System.out.println(toplam);


        /*total_set.foreach((line)->{
            System.out.println(line);
        });*/
        //total_set.coalesce(1).saveAsTextFile("C:\\Users\\ULGEN\\Documents\\Idea_Workspace\\Sar_Cluster\\set\\deneme2");
    }
}

        /*SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example")
                .config(conf)
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

        Dataset<Row> set1 =  spark.read().csv(Sar_image_set1);
        Dataset<Row> set2 =  spark.read().csv(Sar_image_set2);
        Dataset<Row> set3 =  spark.read().csv(Sar_image_set3);
        Dataset<Row> set4 =  spark.read().csv(Sar_image_set4);
        Dataset<Row> set5 =  spark.read().csv(Sar_image_set5);
        Dataset<Row> set6 =  spark.read().csv(Sar_image_set6);
        Dataset<Row> set7 =  spark.read().csv(Sar_image_set7);
        Dataset<Row> set8 =  spark.read().csv(Sar_image_set8);
        Dataset<Row> set9 =  spark.read().csv(Sar_image_set9);

        Dataset<Row> total_set = set1.Test_Environment.Union(set2).Test_Environment.Union(set3).Test_Environment.Union(set4).Test_Environment.Union(set5).Test_Environment.Union(set6).Test_Environment.Union(set7).Test_Environment.Union(set8).Test_Environment.Union(set9);
        Dataset<Row> total_set2 = total_set.distinct();
        Dataset<Row> total_set3 = set1.unionAll(set2).unionAll(set3).unionAll(set4).unionAll(set5).unionAll(set6).unionAll(set7).unionAll(set8).unionAll(set9);*/