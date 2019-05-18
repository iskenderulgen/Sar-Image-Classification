package Sar_Image_Analysis.ML_Models;

import Sar_Image_Analysis.Charting;
import Sar_Image_Analysis.Cross_Validation;
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

    public static NaiveBayesModel NB_Modelling(String[] Training_Path, JavaSparkContext jsc,
                                               double lambda, String results_path) throws IOException {


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
        System.out.println("Total time elapsed for labeling section is =  " + labeling_time +
                "  Milisec\t" + (labeling_time / 1000) + "  Seconds\t" + labeling_time / 60000 + "  Minutes");

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
        System.out.println("Total time for train phase = " + train_time + "  Milisec\t" +
                (train_time / 1000) + "  Seconds\t" + train_time / 60000 + "  Minutes");

        // This Part calculates the accuracy of the naive bayes model in two ways one is classical split 70-30 train test way
        // second is more robust cross validation way
        double NB_classic_accuracy = NB_Accuaricy(collected_labels, lambda);

        JavaRDD<LabeledPoint>[] nb_split_along = Pre_Process.splitting_10(alongside_labeled, jsc);
        JavaRDD<LabeledPoint>[] nb_split_build = Pre_Process.splitting_10(building_labeled, jsc);
        JavaRDD<LabeledPoint>[] nb_split_road = Pre_Process.splitting_10(road_labeled, jsc);
        JavaRDD<LabeledPoint>[] nb_split_vegetation = Pre_Process.splitting_10(vegetation_labeled, jsc);
        JavaRDD<LabeledPoint>[] nb_split_water = Pre_Process.splitting_10(water_labeled, jsc);
        /*
        double NB_Cross_Validation_Result = Cross_Validation.Evaluation_Section(nb_split_along, nb_split_build,
                nb_split_road, nb_split_vegetation, nb_split_water,"Naive Bayes Cross Validation",
                "Naive_Bayes",jsc);

        Charting.NB_Model_Accuracy_Chart(NB_classic_accuracy, NB_Cross_Validation_Result, results_path);
        */


        return NB_Model;
    }

    private static double NB_Accuaricy(JavaRDD<LabeledPoint> datalabeled, double lambda) {
        JavaRDD<LabeledPoint>[] tmp = datalabeled.randomSplit(new double[]{0.7, 0.3}, 11L);
        JavaRDD<LabeledPoint> training = tmp[0]; // training set
        JavaRDD<LabeledPoint> test = tmp[1]; // test set

        NaiveBayesModel model = NaiveBayes.train(training.rdd(), lambda);
        JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        double NB_Accuracy = 100 * metrics.accuracy();
        System.out.println("Naive Bayes Accuracy based on %70-%30 splitting = \t" + NB_Accuracy + "\n\n");
        return NB_Accuracy;
    }
}
