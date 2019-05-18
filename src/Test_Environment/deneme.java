package Test_Environment;

import Sar_Image_Analysis.Pre_Process;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;

public class deneme {

    private static final String Main_Path = "C:\\Users\\ULGEN\\Documents\\Idea_Workspace\\Sar_Classify\\Sar_Datasets\\";
    private static final String Results_Path = "C:\\Users\\ULGEN\\Desktop\\Total_Results\\";

    private static final String alongside_path = Main_Path + "alongside\\training\\alongside.csv";
    private static final String building_path = Main_Path + "building\\training\\building.csv";
    private static final String road_path = Main_Path + "road\\training\\road.csv";
    private static final String vegetation_path = Main_Path + "vegetation\\training\\vegetation.csv";
    private static final String water_path = Main_Path + "water\\training\\water.csv";


    public static void main(String[] args) throws FileNotFoundException {

        SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("Sar_Analysis");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example")
                .config(conf)
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

        PrintStream outputs = new PrintStream(new FileOutputStream(new File(Results_Path+"SVM_Trial_results.txt")));
        System.setOut(outputs);

        //JavaRDD<LabeledPoint> alongside_pos = Pre_Process.labeling_data(alongside_path, 1.0, jsc);
        JavaRDD<LabeledPoint> alongside_neg = Pre_Process.labeling_data(alongside_path, 0.0, jsc);

        //JavaRDD<LabeledPoint> building_pos = Pre_Process.labeling_data(building_path, 1.0, jsc);
        JavaRDD<LabeledPoint> building_neg = Pre_Process.labeling_data(building_path, 0.0, jsc);

        //JavaRDD<LabeledPoint> road_pos = Pre_Process.labeling_data(road_path, 1.0, jsc);
        JavaRDD<LabeledPoint> road_neg = Pre_Process.labeling_data(road_path, 0.0, jsc);

        //JavaRDD<LabeledPoint> vegetation_pos = Pre_Process.labeling_data(vegetation_path, 1.0, jsc);
        JavaRDD<LabeledPoint> vegetation_neg = Pre_Process.labeling_data(vegetation_path, 0.0, jsc);

        JavaRDD<LabeledPoint> water_pos = Pre_Process.labeling_data(water_path, 1.0, jsc);
        //JavaRDD<LabeledPoint> water_neg = Pre_Process.labeling_data(water_path, 0.0, jsc);

        JavaRDD<LabeledPoint> water_labeled_SVM = alongside_neg.union(building_neg).union(road_neg).union(vegetation_neg).union(water_pos);

        JavaRDD<LabeledPoint>[] splits = water_labeled_SVM.randomSplit(new double[]{0.7, 0.3}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];


        for (int iteration = 5; iteration < 1000; iteration = iteration + 5) {
            for (double step_size = 0.005; step_size < 10.0; step_size = step_size + 0.005) {
                for (double reg_param = 0.005; reg_param < 10.0; reg_param = reg_param + 0.005) {

                    SVMModel model = SVMWithSGD.train(training.rdd(), iteration, step_size, reg_param);
                    JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test
                            .map(p -> new Tuple2<>(model.predict(p.features()), p.label()));
                    BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
                    double SVM_classic_accuracy = 100 * metrics.areaUnderROC();
                    if (SVM_classic_accuracy < 90 && SVM_classic_accuracy > 84) {
                        System.out.println("SVM water model iteration =\t"+iteration+"\tstep_size =\t"+step_size+"\treg_param =\t"+reg_param+"\tAccuracy based on %70-%30 splitting is = \t" + (100 * metrics.areaUnderROC()) + "\n");
                    }
                }
                System.out.println("step size ="+step_size);
            }
            System.out.println("iteration  ="+iteration);
        }
    }
}
