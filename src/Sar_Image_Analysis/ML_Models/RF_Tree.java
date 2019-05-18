package Sar_Image_Analysis.ML_Models;

import Sar_Image_Analysis.Charting;
import Sar_Image_Analysis.Cross_Validation;
import Sar_Image_Analysis.Pre_Process;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import scala.Tuple2;

import java.io.IOException;
import java.util.Map;

public class RF_Tree {

    public static RandomForestModel RF_Tree_Modelling(String[] Training_Path, JavaSparkContext jsc,
                                                      Map<Integer, Integer> categoricalFeaturesInfo, int NumClass,
                                                      int NumTrees, String featureSubset, String impurity, int MaxDepth,
                                                      int MaxBins, String results_path) throws IOException {


        long Labeling_Clock_Starts = System.currentTimeMillis();
        System.out.println("\nLabeling Phase Starts");
        JavaRDD<LabeledPoint> alongside_labeled = Pre_Process.labeling_data(Training_Path[0], 1.0, jsc);
        JavaRDD<LabeledPoint> building_labeled = Pre_Process.labeling_data(Training_Path[1], 2.0, jsc);
        JavaRDD<LabeledPoint> road_labeled = Pre_Process.labeling_data(Training_Path[2], 3.0, jsc);
        JavaRDD<LabeledPoint> vegetation_labeled = Pre_Process.labeling_data(Training_Path[3], 4.0, jsc);
        JavaRDD<LabeledPoint> water_labeled = Pre_Process.labeling_data(Training_Path[4], 5.0, jsc);
        System.out.println("Labeling Phase Ends\n");
        long Labeling_Clock_Ends = System.currentTimeMillis();
        long labeling_time = Labeling_Clock_Ends - Labeling_Clock_Starts;
        System.out.println("Total time elapsed for labeling section is =  " + labeling_time + "  Milisec\t" +
                (labeling_time / 1000) + "  Seconds\t" + labeling_time / 60000 + "  Minutes");

        JavaRDD<LabeledPoint> collected_labels = alongside_labeled
                .union(building_labeled)
                .union(road_labeled)
                .union(vegetation_labeled)
                .union(water_labeled);
        long train_start = System.currentTimeMillis();

        RandomForestModel RF_tree_model = RandomForest.trainClassifier(collected_labels, NumClass,
                categoricalFeaturesInfo, NumTrees, featureSubset, impurity, MaxDepth, MaxBins, 12345);
        long train_end = System.currentTimeMillis();
        long train_time = train_end - train_start;
        System.out.println("Total time for train phase = " + train_time + "  Milisec\t" + (train_time / 1000) +
                "  Seconds\t" + train_time / 60000 + "  Minutes");

        double RFTree_Accuracy = RF_tree_Accuracy(collected_labels, categoricalFeaturesInfo, NumClass, NumTrees,
                featureSubset, impurity, MaxDepth, MaxBins);

        JavaRDD<LabeledPoint>[] RF_tree_split_along = Pre_Process.splitting_10(alongside_labeled, jsc);
        JavaRDD<LabeledPoint>[] RF_tree_split_build = Pre_Process.splitting_10(building_labeled, jsc);
        JavaRDD<LabeledPoint>[] RF_tree_split_road = Pre_Process.splitting_10(road_labeled, jsc);
        JavaRDD<LabeledPoint>[] RF_tree_split_vegetation = Pre_Process.splitting_10(vegetation_labeled, jsc);
        JavaRDD<LabeledPoint>[] RF_tree_split_water = Pre_Process.splitting_10(water_labeled, jsc);
        /*
        double RFTree_CV_Accuracy = Cross_Validation.Evaluation_Section(RF_tree_split_along, RF_tree_split_build,
                RF_tree_split_road, RF_tree_split_vegetation, RF_tree_split_water,
                "Random Forest Tree Cross Validation","Random_Forest_Tree", jsc);

        Charting.RF_Tree_Model_Accuracy_Chart(RFTree_Accuracy, RFTree_CV_Accuracy, results_path);
        */


        return RF_tree_model;

    }

    private static double RF_tree_Accuracy(JavaRDD<LabeledPoint> labeled_raw,
                                           Map<Integer, Integer> categoricalFeaturesInfo, int NumClass, int NumTrees,
                                           String featureSubset, String impurity, int MaxDepth, int MaxBins) {
        JavaRDD<LabeledPoint>[] splits = labeled_raw.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        RandomForestModel RF_tree_accuracy_model = RandomForest.trainClassifier(trainingData, NumClass,
                categoricalFeaturesInfo, NumTrees, featureSubset, impurity, MaxDepth, MaxBins, 12345);
        JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(p ->
                new Tuple2<>(RF_tree_accuracy_model.predict(p.features()), p.label()));
        double RFTree_Accuracy = 100 * (predictionAndLabel.filter(pl ->
                pl._1().equals(pl._2())).count() / (double) testData.count());

        System.out.println("Random Forest Tree Accuracy based on %70-%30 splitting = \t" + RFTree_Accuracy + "\n\n");
        return RFTree_Accuracy;
    }
}
