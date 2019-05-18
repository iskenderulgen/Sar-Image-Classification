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
    private static final String Main_Path = "C:\\Users\\ULGEN\\Documents\\Idea_Workspace\\Sar_Classify\\Sar_Datasets\\Denoised_Big_Set\\";
    private static final String Sar_image_set1 = Main_Path + "toplam1.csv";
    private static final String Sar_image_set2 = Main_Path + "toplam2.csv";
    private static final String Sar_image_set3 = Main_Path + "toplam3.csv";
    private static final String Sar_image_set4 = Main_Path + "toplam4.csv";
    private static final String Sar_image_set5 = Main_Path + "toplam5.csv";
    private static final String Sar_image_set6 = Main_Path + "toplam6.csv";
    private static final String Sar_image_set7 = Main_Path + "toplam7.csv";
    private static final String Sar_image_set8 = Main_Path + "toplam8.csv";
    private static final String Sar_image_set9 = Main_Path + "toplam9.csv";
    private static final String Sar_image_set10 = Main_Path + "toplam10.csv";


    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("Test_Environment.Union");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        jsc.setLogLevel("ERROR");


        JavaRDD<String> set1 = jsc.textFile(Sar_image_set1);
        JavaRDD<String> set2 = jsc.textFile(Sar_image_set2);
        JavaRDD<String> set3 = jsc.textFile(Sar_image_set3);
        JavaRDD<String> set4 = jsc.textFile(Sar_image_set4);
        JavaRDD<String> set5 = jsc.textFile(Sar_image_set5);
        JavaRDD<String> set6 = jsc.textFile(Sar_image_set6);
        JavaRDD<String> set7 = jsc.textFile(Sar_image_set7);
        JavaRDD<String> set8 = jsc.textFile(Sar_image_set8);
        JavaRDD<String> set9 = jsc.textFile(Sar_image_set9);
        JavaRDD<String> set10 = jsc.textFile(Sar_image_set10);


        JavaRDD<String> total_set = set1.
                union(set2).
                union(set3).
                union(set4).
                union(set5).
                union(set6).
                union(set7).
                union(set8).
                union(set9).
                union(set10);

        long set1_count = set1.count();
        long set2_count = set2.count();
        long set3_count = set3.count();
        long set4_count = set4.count();
        long set5_count = set5.count();
        long set6_count = set6.count();
        long set7_count = set7.count();
        long set8_count = set8.count();
        long set9_count = set9.count();
        long set10_count = set10.count();


        System.out.println(set1_count);
        System.out.println(set2_count);
        System.out.println(set3_count);
        System.out.println(set4_count);
        System.out.println(set5_count);
        System.out.println(set6_count);
        System.out.println(set7_count);
        System.out.println(set8_count);
        System.out.println(set9_count);
        System.out.println(set10_count);

        long toplam = (set1_count+set2_count+set3_count+set4_count+set5_count+
                set6_count+set7_count+set8_count+set9_count+set10_count);

        long total_set_count = total_set.count();
        if ( toplam == total_set_count){
            System.out.println(total_set_count);
        }
        else
            System.out.println("not equal");

        total_set.coalesce(1).saveAsTextFile(Main_Path+"\\total_set.csv");
        System.out.println("done");
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