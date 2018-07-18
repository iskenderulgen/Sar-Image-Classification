package Sar_Image_Analysis.ML_Models;

import Sar_Image_Analysis.Charting;
import Sar_Image_Analysis.Cross_Validation_Sar;
import Sar_Image_Analysis.Pre_Process;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import scala.Tuple2;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class DTree {

    private static double Dtree_classic_accuracy=0;


    public static DecisionTreeModel Dtree_Model (String alongside_path, String building_path, String road_path,
                                                 String vegetation_path, String water_path, JavaSparkContext jsc,String results_path) throws IOException {


        long Labeling_Clock_Starts = System.currentTimeMillis();
        System.out.println("\nLabeling Phase Starts");
        JavaRDD<LabeledPoint> alongside_labeled = Pre_Process.labelingdata(alongside_path, 1.0, jsc);
        JavaRDD<LabeledPoint> building_labeled = Pre_Process.labelingdata(building_path, 2.0, jsc);
        JavaRDD<LabeledPoint> road_labeled = Pre_Process.labelingdata(road_path, 3.0, jsc);
        JavaRDD<LabeledPoint> vegetation_labeled = Pre_Process.labelingdata(vegetation_path, 4.0, jsc);
        JavaRDD<LabeledPoint> water_labeled = Pre_Process.labelingdata(water_path, 5.0, jsc);
        System.out.println("Labeling Phase Ends\n");
        long Labeling_Clock_Ends = System.currentTimeMillis();
        long labeling_time = Labeling_Clock_Ends - Labeling_Clock_Starts;
        System.out.println("Total time elapsed for labeling section is =  " + labeling_time + "  Milisec\t" + (labeling_time / 1000) + "  Seconds\t" + labeling_time / 60000 + "  Minutes");

        JavaRDD<LabeledPoint> collected_labels = alongside_labeled
                .union(building_labeled)
                .union(road_labeled)
                .union(vegetation_labeled)
                .union(water_labeled);

        long train_start = System.currentTimeMillis();
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        DecisionTreeModel Dtree_model = DecisionTree.trainClassifier(collected_labels,6,
                categoricalFeaturesInfo,"gini",5,32);
        long train_end = System.currentTimeMillis();
        long train_time = train_end - train_start;
        System.out.println("Total time for train phase = "+train_time+"  Milisec\t" + (train_time / 1000) + "  Seconds\t" + train_time / 60000 + "  Minutes");

        Dtree_Accuracy(collected_labels);
        Cross_Validation_Sar.Dtree_CV_Accuracy(alongside_labeled,building_labeled,road_labeled,vegetation_labeled,water_labeled,jsc);
        Charting.DTree_Model_Chart(Dtree_classic_accuracy,Cross_Validation_Sar.cross_validation_result,results_path);
        return Dtree_model;
    }

    private static void Dtree_Accuracy(JavaRDD<LabeledPoint> datalabel){
        JavaRDD<LabeledPoint>[] splits = datalabel.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        DecisionTreeModel Dtree_accuracy_model = DecisionTree.trainClassifier(trainingData,6,categoricalFeaturesInfo,"gini",5,32);
        JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(p -> new Tuple2<>(Dtree_accuracy_model.predict(p.features()), p.label()));
        double accuracy = predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) testData.count();
        Dtree_classic_accuracy = 100 * accuracy;
        System.out.println("Decision Tree Accuracy based on %70-%30 splitting = \t" + 100 *accuracy);
    }
}
