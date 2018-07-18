package Sar_Image_Analysis;

import Sar_Image_Analysis.ML_Models.SVM_Model_Train;
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
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.util.LongAccumulator;
import scala.Tuple2;

import java.util.*;


public class Cross_Validation_Sar {
    private static int[] numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // fn stands for fold_number
    public static double cross_validation_result =0;

    public static void NB_CV_Accuaricy(JavaRDD<LabeledPoint> alongside_labeled, JavaRDD<LabeledPoint> building_labeled,
                                       JavaRDD<LabeledPoint> road_labeled, JavaRDD<LabeledPoint> vegetation_labeled,
                                       JavaRDD<LabeledPoint> water_labeled, JavaSparkContext jsc) {

        JavaRDD<LabeledPoint> nb_sonuc_along[] = splitting_10(alongside_labeled, jsc);
        JavaRDD<LabeledPoint> nb_sonuc_build[] = splitting_10(building_labeled, jsc);
        JavaRDD<LabeledPoint> nb_sonuc_road[] = splitting_10(road_labeled, jsc);
        JavaRDD<LabeledPoint> nb_sonuc_vege[] = splitting_10(vegetation_labeled, jsc);
        JavaRDD<LabeledPoint> nb_sonuc_water[] = splitting_10(water_labeled, jsc);

        Evaluation_Section(nb_sonuc_along, nb_sonuc_build, nb_sonuc_road, nb_sonuc_vege, nb_sonuc_water,"Naive Bayes Cross Validation", "Naive_Bayes",jsc);

    }

    public static void SVM_CV_Accuracy(JavaRDD<LabeledPoint> alongside_pos, JavaRDD<LabeledPoint> alongside_neg,
                                       JavaRDD<LabeledPoint> building_pos, JavaRDD<LabeledPoint> building_neg,
                                       JavaRDD<LabeledPoint> road_pos, JavaRDD<LabeledPoint> road_neg,
                                       JavaRDD<LabeledPoint> vegetation_pos, JavaRDD<LabeledPoint> vegetation_neg,
                                       JavaRDD<LabeledPoint> water_pos, JavaRDD<LabeledPoint> water_neg,
                                       JavaSparkContext jsc) {


        JavaRDD<LabeledPoint> svm_along_pos[] = splitting_10(alongside_pos, jsc);
        JavaRDD<LabeledPoint> svm_along_neg[] = splitting_10(alongside_neg, jsc);
        JavaRDD<LabeledPoint> svm_build_pos[] = splitting_10(building_pos, jsc);
        JavaRDD<LabeledPoint> svm_build_neg[] = splitting_10(building_neg, jsc);
        JavaRDD<LabeledPoint> svm_road_pos[] = splitting_10(road_pos, jsc);
        JavaRDD<LabeledPoint> svm_road_neg[] = splitting_10(road_neg, jsc);
        JavaRDD<LabeledPoint> svm_vege_pos[] = splitting_10(vegetation_pos, jsc);
        JavaRDD<LabeledPoint> svm_vege_neg[] = splitting_10(vegetation_neg, jsc);
        JavaRDD<LabeledPoint> svm_water_pos[] = splitting_10(water_pos, jsc);
        JavaRDD<LabeledPoint> svm_water_neg[] = splitting_10(water_neg, jsc);


        Evaluation_Section(svm_along_pos, svm_build_neg, svm_road_neg, svm_vege_neg, svm_water_neg,  "Alongside Cross Validation", "SVM",jsc);
        SVM_Model_Train.SVM_CV_Array[0] = cross_validation_result;
        Evaluation_Section(svm_along_neg, svm_build_pos, svm_road_neg, svm_vege_neg, svm_water_neg,  "Building Cross Validation", "SVM",jsc);
        SVM_Model_Train.SVM_CV_Array[1] = cross_validation_result;
        Evaluation_Section(svm_along_neg, svm_build_neg, svm_road_pos, svm_vege_neg, svm_water_neg,  "Road Cross Validation", "SVM",jsc);
        SVM_Model_Train.SVM_CV_Array[2] = cross_validation_result;
        Evaluation_Section(svm_along_neg, svm_build_neg, svm_road_neg, svm_vege_pos, svm_water_neg,  "Vegetation Cross Validation", "SVM",jsc);
        SVM_Model_Train.SVM_CV_Array[3] = cross_validation_result;
        Evaluation_Section(svm_along_neg, svm_build_neg, svm_road_neg, svm_vege_neg, svm_water_pos,  "Water Cross Validation", "SVM",jsc);
        SVM_Model_Train.SVM_CV_Array[4] = cross_validation_result;
    }

    public static void Dtree_CV_Accuracy(JavaRDD<LabeledPoint> alongside_labeled, JavaRDD<LabeledPoint> building_labeled,
                                         JavaRDD<LabeledPoint> road_labeled, JavaRDD<LabeledPoint> vegetation_labeled,
                                         JavaRDD<LabeledPoint> water_labeled, JavaSparkContext jsc){

        JavaRDD<LabeledPoint> dtree_sonuc_along[] = splitting_10(alongside_labeled, jsc);
        JavaRDD<LabeledPoint> dtree_sonuc_build[] = splitting_10(building_labeled, jsc);
        JavaRDD<LabeledPoint> dtree_sonuc_road[] = splitting_10(road_labeled, jsc);
        JavaRDD<LabeledPoint> dtree_sonuc_vege[] = splitting_10(vegetation_labeled, jsc);
        JavaRDD<LabeledPoint> dtree_sonuc_water[] = splitting_10(water_labeled, jsc);

        Evaluation_Section(dtree_sonuc_along, dtree_sonuc_build, dtree_sonuc_road, dtree_sonuc_vege, dtree_sonuc_water,"Decision Tree Cross Validation", "Decision_Tree",jsc);

    }


    private static void Evaluation_Section(JavaRDD<LabeledPoint>[] sonuc_along, JavaRDD<LabeledPoint>[] sonuc_build,
                                           JavaRDD<LabeledPoint>[] sonuc_road, JavaRDD<LabeledPoint>[] sonuc_vege,
                                           JavaRDD<LabeledPoint>[] sonuc_water,  String identifier,
                                           String Cross_identifier,JavaSparkContext jsc) {

        long CV_Clock_Starts = System.currentTimeMillis();
        int[] fn = numbers.clone(); // fn stands for fold_number
        LongAccumulator Cross_Validation_Total =  jsc.sc().longAccumulator();

        System.out.println("/////////////////////////////////  " + identifier + "  ////////////////////////////////////////\n\n");
        for (int i = 0; i < 10; i++) {

            JavaRDD<LabeledPoint> test_un = sonuc_along[fn[0]].union(sonuc_build[fn[0]]).union(sonuc_road[fn[0]]).union(sonuc_vege[fn[0]]).union(sonuc_water[fn[0]]);

            JavaRDD<LabeledPoint> train_un = sonuc_along[fn[1]].union(sonuc_along[fn[2]]).union(sonuc_along[fn[3]]).union(sonuc_along[fn[4]]).union(sonuc_along[fn[5]]).union(sonuc_along[fn[6]]).union(sonuc_along[fn[7]]).union(sonuc_along[fn[8]]).union(sonuc_along[fn[9]])
                    .union(sonuc_build[fn[1]]).union(sonuc_build[fn[2]]).union(sonuc_build[fn[3]]).union(sonuc_build[fn[4]]).union(sonuc_build[fn[5]]).union(sonuc_build[fn[6]]).union(sonuc_build[fn[7]]).union(sonuc_build[fn[8]]).union(sonuc_build[fn[9]])
                    .union(sonuc_road[fn[1]]).union(sonuc_road[fn[2]]).union(sonuc_road[fn[3]]).union(sonuc_road[fn[4]]).union(sonuc_road[fn[5]]).union(sonuc_road[fn[6]]).union(sonuc_road[fn[7]]).union(sonuc_road[fn[8]]).union(sonuc_road[fn[9]])
                    .union(sonuc_vege[fn[1]]).union(sonuc_vege[fn[2]]).union(sonuc_vege[fn[3]]).union(sonuc_vege[fn[4]]).union(sonuc_vege[fn[5]]).union(sonuc_vege[fn[6]]).union(sonuc_vege[fn[7]]).union(sonuc_vege[fn[8]]).union(sonuc_vege[fn[9]])
                    .union(sonuc_water[fn[1]]).union(sonuc_water[fn[2]]).union(sonuc_water[fn[3]]).union(sonuc_water[fn[4]]).union(sonuc_water[fn[5]]).union(sonuc_water[fn[6]]).union(sonuc_water[fn[7]]).union(sonuc_water[fn[8]]).union(sonuc_water[fn[9]]);


            switch (Cross_identifier) {
                case "Naive_Bayes": {
                    NaiveBayesModel model = NaiveBayes.train(train_un.rdd(), 1.0);
                    JavaPairRDD<Object, Object> predictionAndLabels = test_un.mapToPair(p ->
                            new Tuple2<>(model.predict(p.features()), p.label()));
                    MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
                    System.out.println("K = " + (i + 1) + ". " + identifier + " Fold Analysis Result is = " + 100 * metrics.accuracy());
                    Cross_Validation_Total.add((long) (100 * metrics.accuracy()));
                    break;
                }
                case "SVM": {
                    SVMModel model = SVMWithSGD.train(train_un.rdd(), 1).setThreshold(0.5);
                    model.clearThreshold();
                    JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test_un
                            .map(p -> new Tuple2<>(model.predict(p.features()), p.label()));
                    BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
                    System.out.println("K = " + (i + 1) + ". " + identifier + " Fold Analysis Result is = " + 100 * metrics.areaUnderROC());
                    Cross_Validation_Total.add((long) (100 * metrics.areaUnderROC()));
                    break;
                }
                case "Decision_Tree": {
                    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
                    DecisionTreeModel model = DecisionTree.trainClassifier(train_un, 6, categoricalFeaturesInfo, "gini", 5, 32);
                    JavaPairRDD<Double, Double> predictionAndLabel = test_un.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
                    double accuracy = predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) test_un.count();
                    System.out.println("K = " + (i + 1) + ". " + identifier + " Fold Analysis Result is = " + 100 * accuracy);
                    Cross_Validation_Total.add((long) (100 * accuracy));
                    break;
                }
            }

        if (i == 9) swap(fn, 0, 9);
        else swap(fn, 0, i + 1);
        }


        cross_validation_result = Cross_Validation_Total.value()/10;
        System.out.println("K = 10 Fold " + identifier + " Total Analysis Result is = " + (Cross_Validation_Total.value()/10));
        long CV_Clock_Ends = System.currentTimeMillis();
        long analysis_time = CV_Clock_Ends - CV_Clock_Starts;
        System.out.println("Total taime elapsed for "+identifier+" Cross Validation section is =  "+analysis_time+"  Milisec\t"+(analysis_time/1000)+"  Seconds\t"+analysis_time/60000+"  Minutes\n\n");
        System.out.println("/////////////////////////////////  " + identifier + "  ////////////////////////////////////////\n\n");
        }

private static JavaRDD<LabeledPoint>[] splitting_10(JavaRDD<LabeledPoint> data, JavaSparkContext jsc) {
        long sayi = data.count();
        int i = (int) sayi / 10;
        JavaRDD<LabeledPoint> data1 = jsc.parallelize(data.take(i));
        JavaRDD<LabeledPoint> data2 = jsc.parallelize(data.take(2 * i)).subtract(jsc.parallelize(data.take(i)));
        JavaRDD<LabeledPoint> data3 = jsc.parallelize(data.take(3 * i)).subtract(jsc.parallelize(data.take(2 * i)));
        JavaRDD<LabeledPoint> data4 = jsc.parallelize(data.take(4 * i)).subtract(jsc.parallelize(data.take(3 * i)));
        JavaRDD<LabeledPoint> data5 = jsc.parallelize(data.take(5 * i)).subtract(jsc.parallelize(data.take(4 * i)));
        JavaRDD<LabeledPoint> data6 = jsc.parallelize(data.take(6 * i)).subtract(jsc.parallelize(data.take(5 * i)));
        JavaRDD<LabeledPoint> data7 = jsc.parallelize(data.take(7 * i)).subtract(jsc.parallelize(data.take(6 * i)));
        JavaRDD<LabeledPoint> data8 = jsc.parallelize(data.take(8 * i)).subtract(jsc.parallelize(data.take(7 * i)));
        JavaRDD<LabeledPoint> data9 = jsc.parallelize(data.take(9 * i)).subtract(jsc.parallelize(data.take(8 * i)));
        JavaRDD<LabeledPoint> data10 = data.subtract(jsc.parallelize(data.take(9 * i)));

        JavaRDD<LabeledPoint> sonuc[] = new JavaRDD[]{data1, data2, data3, data4, data5, data6, data7, data8, data9, data10};
        return sonuc;
        }

private static void swap(int[] a, int i, int j) {
        Object temp = a[i];
        a[i] = a[j];
        a[j] = (int) temp;
        }
}