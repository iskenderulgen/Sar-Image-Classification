package Sar_Image_Analysis;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.util.LongAccumulator;
import scala.Tuple2;

import java.util.*;


public class Cross_Validation {
    private static int[] numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // fn stands for fold_number

    public static double Evaluation_Section(JavaRDD<LabeledPoint>[] split_along, JavaRDD<LabeledPoint>[] split_build,
                                             JavaRDD<LabeledPoint>[] split_road, JavaRDD<LabeledPoint>[] split_vegetation,
                                             JavaRDD<LabeledPoint>[] split_water, String identifier,
                                             String Cross_identifier, JavaSparkContext jsc) {

        long CV_Clock_Starts = System.currentTimeMillis();
        int[] fn = numbers.clone(); // fn stands for fold_number
        LongAccumulator Cross_Validation_Total = jsc.sc().longAccumulator();

        System.out.println("//////////////////////////  " + identifier + " STARTS  //////////////////////////////\n\n");
        for (int i = 0; i < 10; i++) {

            JavaRDD<LabeledPoint> test_un = split_along[fn[0]].union(split_build[fn[0]]).union(split_road[fn[0]]).
                    union(split_vegetation[fn[0]]).union(split_water[fn[0]]);

            JavaRDD<LabeledPoint> train_un = split_along[fn[1]].union(split_along[fn[2]]).union(split_along[fn[3]]).
                    union(split_along[fn[4]]).union(split_along[fn[5]]).union(split_along[fn[6]]).
                    union(split_along[fn[7]]).union(split_along[fn[8]]).union(split_along[fn[9]])
                    .union(split_build[fn[1]]).union(split_build[fn[2]]).union(split_build[fn[3]]).
                            union(split_build[fn[4]]).union(split_build[fn[5]]).union(split_build[fn[6]]).
                            union(split_build[fn[7]]).union(split_build[fn[8]]).union(split_build[fn[9]])
                    .union(split_road[fn[1]]).union(split_road[fn[2]]).union(split_road[fn[3]]).
                            union(split_road[fn[4]]).union(split_road[fn[5]]).union(split_road[fn[6]]).
                            union(split_road[fn[7]]).union(split_road[fn[8]]).union(split_road[fn[9]])
                    .union(split_vegetation[fn[1]]).union(split_vegetation[fn[2]]).union(split_vegetation[fn[3]]).
                            union(split_vegetation[fn[4]]).union(split_vegetation[fn[5]]).
                            union(split_vegetation[fn[6]]).union(split_vegetation[fn[7]]).
                            union(split_vegetation[fn[8]]).union(split_vegetation[fn[9]])
                    .union(split_water[fn[1]]).union(split_water[fn[2]]).union(split_water[fn[3]]).
                            union(split_water[fn[4]]).union(split_water[fn[5]]).union(split_water[fn[6]]).
                            union(split_water[fn[7]]).union(split_water[fn[8]]).union(split_water[fn[9]]);


            switch (Cross_identifier) {
                case "Naive_Bayes": {
                    NaiveBayesModel model = NaiveBayes.train(train_un.rdd(), 1.0);
                    JavaPairRDD<Object, Object> predictionAndLabels = test_un.mapToPair(p ->
                            new Tuple2<>(model.predict(p.features()), p.label()));
                    MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
                    double accuracy = 100 * metrics.accuracy();
                    System.out.println("K = " + (i + 1) + ". " + identifier + " Fold Analysis Result is = " + accuracy);
                    Cross_Validation_Total.add((long) accuracy);
                    break;
                }
                case "Decision_Tree": {
                    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
                    DecisionTreeModel model = DecisionTree.trainClassifier(train_un, 6, categoricalFeaturesInfo,
                            "gini", 5, 32);
                    JavaPairRDD<Double, Double> predictionAndLabel = test_un.mapToPair(p ->
                            new Tuple2<>(model.predict(p.features()), p.label()));
                    double accuracy = 100 * (predictionAndLabel.filter(pl ->
                            pl._1().equals(pl._2())).count() / (double) test_un.count());
                    System.out.println("K = " + (i + 1) + ". " + identifier + " Fold Analysis Result is = " + accuracy);
                    Cross_Validation_Total.add((long) accuracy);
                    break;
                }
                case "Random_Forest_Tree": {
                    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
                    RandomForestModel RF_tree_accuracy_model = RandomForest.trainClassifier(train_un, 6,
                            categoricalFeaturesInfo, 10, "auto", "gini", 10,
                            32, 12345);
                    JavaPairRDD<Double, Double> predictionAndLabel = test_un.mapToPair(p ->
                            new Tuple2<>(RF_tree_accuracy_model.predict(p.features()), p.label()));
                    double accuracy = 100 * (predictionAndLabel.filter(pl ->
                            pl._1().equals(pl._2())).count() / (double) test_un.count());
                    System.out.println("K = " + (i + 1) + ". " + identifier + " Fold Analysis Result is = " + accuracy);
                    Cross_Validation_Total.add((long) (accuracy));
                    break;
                }
            }

            if (i == 9) swap(fn, 0, 9);
            else swap(fn, 0, i + 1);
        }


        double cross_validation_result = Cross_Validation_Total.value() / 10.0;
        long CV_Clock_Ends = System.currentTimeMillis();
        long analysis_time = CV_Clock_Ends - CV_Clock_Starts;

        System.out.println("K = 10 Fold " + identifier + " Total Analysis Result is = " + cross_validation_result);
        System.out.println("Total time elapsed for " + identifier + " Cross Validation section is =  " + analysis_time +
                "  Milisec\t" + (analysis_time / 1000) + "  Seconds\t" + analysis_time / 60000 + "  Minutes\n\n");
        System.out.println("//////////////////////////  " + identifier + " ENDS  ////////////////////////////////\n\n");
        return cross_validation_result;
    }



    private static void swap(int[] a, int i, int j) {
        Object temp = a[i];
        a[i] = a[j];
        a[j] = (int) temp;
    }
}

/*
  case "SVM": {
                    SVMModel model = SVMWithSGD.train(train_un.rdd(), 1).setThreshold(0.5);
                    model.clearThreshold();
                    JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test_un
                            .map(p -> new Tuple2<>(model.predict(p.features()), p.label()));
                    BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
                    double accuracy = 100 * metrics.areaUnderROC();
                    System.out.println("K = " + (i + 1) + ". " + identifier + " Fold Analysis Result is = " + accuracy);
                    Cross_Validation_Total.add((long) accuracy);
                    break;
                }
 */