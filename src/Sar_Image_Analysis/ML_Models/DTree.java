package Sar_Image_Analysis.ML_Models;

import Sar_Image_Analysis.Charting;
import Sar_Image_Analysis.Cross_Validation;
import Sar_Image_Analysis.Pre_Process;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import scala.Tuple2;

import java.io.IOException;
import java.util.Map;

public class DTree {

    public static DecisionTreeModel DTree_Model(String[] Training_Path, JavaSparkContext jsc,
                                                Map<Integer, Integer> categoricalFeaturesInfo, int NumClass,
                                                String impurity, int MaxDepth, int MaxBins, String results_path)
            throws IOException {



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

        DecisionTreeModel DTree_model = DecisionTree.trainClassifier(collected_labels, NumClass,
                categoricalFeaturesInfo, impurity, MaxDepth, MaxBins);
        long train_end = System.currentTimeMillis();
        long train_time = train_end - train_start;
        System.out.println("Total time for train phase = " + train_time + "  Milisec\t" + (train_time / 1000) +
                "  Seconds\t" + train_time / 60000 + "  Minutes");

        double DTree_Accuracy = DTree_Accuracy(collected_labels, categoricalFeaturesInfo, NumClass,
                impurity, MaxDepth, MaxBins);

        JavaRDD<LabeledPoint>[] DTree_split_along = Pre_Process.splitting_10(alongside_labeled, jsc);
        JavaRDD<LabeledPoint>[] DTree_split_build = Pre_Process.splitting_10(building_labeled, jsc);
        JavaRDD<LabeledPoint>[] DTree_split_road = Pre_Process.splitting_10(road_labeled, jsc);
        JavaRDD<LabeledPoint>[] DTree_split_vegetation = Pre_Process.splitting_10(vegetation_labeled, jsc);
        JavaRDD<LabeledPoint>[] DTree_split_water = Pre_Process.splitting_10(water_labeled, jsc);
        /*
        double DTree_CV_Accuracy = Cross_Validation.Evaluation_Section(DTree_split_along, DTree_split_build,
                DTree_split_road, DTree_split_vegetation, DTree_split_water,"Decision Tree Cross Validation",
                "Decision_Tree",jsc);

        Charting.DTree_Model_Accuracy_Chart(DTree_Accuracy, DTree_CV_Accuracy, results_path);
        */


        return DTree_model;
    }

    private static double DTree_Accuracy(JavaRDD<LabeledPoint> datalabel, Map<Integer, Integer> categoricalFeaturesInfo,
                                         int NumClass, String impurity, int MaxDepth, int MaxBins) {
        JavaRDD<LabeledPoint>[] splits = datalabel.randomSplit(new double[]{0.7, 0.3});
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> testData = splits[1];

        DecisionTreeModel DTree_model = DecisionTree.trainClassifier(trainingData, NumClass,
                categoricalFeaturesInfo, impurity, MaxDepth, MaxBins);

        JavaPairRDD<Double, Double> predictionAndLabel = testData.mapToPair(p ->
                new Tuple2<>(DTree_model.predict(p.features()), p.label()));

        double DTree_Accuracy = 100 * (predictionAndLabel.filter(pl ->
                pl._1().equals(pl._2())).count() / (double) testData.count());

        System.out.println("Decision Tree Accuracy based on %70-%30 splitting = \t" + DTree_Accuracy + "\n\n");

        return DTree_Accuracy;
    }
}
