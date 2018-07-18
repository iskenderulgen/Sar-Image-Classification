package Sar_Image_Analysis.ML_Models;

import Sar_Image_Analysis.Charting;
import Sar_Image_Analysis.Cross_Validation_Sar;
import Sar_Image_Analysis.Main_Sar_Analysis;
import Sar_Image_Analysis.Pre_Process;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.io.IOException;

public class NB_Model_Train {

    private static double NB_classic_accuracy=0;

    public static NaiveBayesModel NB_Modelling(String alongside_path, String building_path, String road_path,
                                               String vegetation_path, String water_path, JavaSparkContext jsc,
                                               double lambda,String results_path) throws IOException {

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

        //This section unionize each labeled set to train them
        JavaRDD<LabeledPoint> collected_labels = alongside_labeled
                .union(building_labeled)
                .union(road_labeled)
                .union(vegetation_labeled)
                .union(water_labeled);

        long train_start = System.currentTimeMillis();
        NaiveBayesModel NB_Model = NaiveBayes.train(collected_labels.rdd(), lambda);
        long train_end = System.currentTimeMillis();
        long train_time = train_end - train_start;
        System.out.println("Total time for train phase = "+train_time+"  Milisec\t" + (train_time / 1000) + "  Seconds\t" + train_time / 60000 + "  Minutes");

        // This Part calculates the accuracy of the naive bayes model in two ways one is classical split 70-30 train test way
        // second is more robust cross validation way
        NB_Accuaricy(collected_labels, lambda);
        Cross_Validation_Sar.NB_CV_Accuaricy(alongside_labeled,building_labeled,road_labeled,vegetation_labeled,water_labeled,jsc);
        Charting.NB_Model_Chart(NB_classic_accuracy,Cross_Validation_Sar.cross_validation_result,results_path);
        return NB_Model;
    }

    private static void NB_Accuaricy(JavaRDD<LabeledPoint> datalabel, double lambda) {
        JavaRDD<LabeledPoint>[] tmp = datalabel.randomSplit(new double[]{0.7, 0.3}, 11L);
        JavaRDD<LabeledPoint> training = tmp[0]; // training set
        JavaRDD<LabeledPoint> test = tmp[1]; // test set

        NaiveBayesModel model = NaiveBayes.train(training.rdd(), lambda);
        JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        NB_classic_accuracy = 100 * metrics.accuracy();
        System.out.println("Naive Bayes Accuracy based on %70-%30 splitting = \t" + 100 * metrics.accuracy());
    }
}
