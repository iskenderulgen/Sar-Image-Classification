package Test_Environment;

import Sar_Image_Analysis.Charting;
import Sar_Image_Analysis.Cross_Validation;
import Sar_Image_Analysis.Pre_Process;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

import java.io.IOException;


public class SVM_Model_Train {
    // One vs Rest Approach used in support vector classifier

    public static SVMModel[] SVM_Model(String alongside_path, String building_path, String road_path,
                                       String vegetation_path, String water_path, JavaSparkContext jsc, String path) throws IOException {


        long Labeling_Clock_Starts = System.currentTimeMillis();
        System.out.println("\nSVM Labeling Phase Starts");
        JavaRDD<LabeledPoint> alongside_pos = Pre_Process.labeling_data(alongside_path, 1.0, jsc);
        JavaRDD<LabeledPoint> alongside_neg = Pre_Process.labeling_data(alongside_path, 0.0, jsc);

        JavaRDD<LabeledPoint> building_pos = Pre_Process.labeling_data(building_path, 1.0, jsc);
        JavaRDD<LabeledPoint> building_neg = Pre_Process.labeling_data(building_path, 0.0, jsc);

        JavaRDD<LabeledPoint> road_pos = Pre_Process.labeling_data(road_path, 1.0, jsc);
        JavaRDD<LabeledPoint> road_neg = Pre_Process.labeling_data(road_path, 0.0, jsc);

        JavaRDD<LabeledPoint> vegetation_pos = Pre_Process.labeling_data(vegetation_path, 1.0, jsc);
        JavaRDD<LabeledPoint> vegetation_neg = Pre_Process.labeling_data(vegetation_path, 0.0, jsc);

        JavaRDD<LabeledPoint> water_pos = Pre_Process.labeling_data(water_path, 1.0, jsc);
        JavaRDD<LabeledPoint> water_neg = Pre_Process.labeling_data(water_path, 0.0, jsc);

        JavaRDD<LabeledPoint> alongside_labeled_SVM = alongside_pos.union(building_neg).union(road_neg).union(vegetation_neg).union(water_neg);
        JavaRDD<LabeledPoint> building_labeled_SVM = alongside_neg.union(building_pos).union(road_neg).union(vegetation_neg).union(water_neg);
        JavaRDD<LabeledPoint> road_labeled_SVM = alongside_neg.union(building_neg).union(road_pos).union(vegetation_neg).union(water_neg);
        JavaRDD<LabeledPoint> vegetation_labeled_SVM = alongside_neg.union(building_neg).union(road_neg).union(vegetation_pos).union(water_neg);
        JavaRDD<LabeledPoint> water_labeled_SVM = alongside_neg.union(building_neg).union(road_neg).union(vegetation_neg).union(water_pos);
        System.out.println("SVM Labeling Phase Ends\n");
        long Labeling_Clock_Ends = System.currentTimeMillis();
        long labeling_time = Labeling_Clock_Ends - Labeling_Clock_Starts;
        System.out.println("Total time elapsed for SVM labeling section is =  " + labeling_time + "  Milisec\t" + (labeling_time / 1000) + "  Seconds\t" + labeling_time / 60000 + "  Minutes");


        long train_start = System.currentTimeMillis();
        SVMModel SVM_Alongside_Model = SVMWithSGD.train(alongside_labeled_SVM.rdd(), 200, 0.1, 0.01).setThreshold(0.5);
        SVMModel SVM_building_Model = SVMWithSGD.train(building_labeled_SVM.rdd(), 200, 0.1, 0.01).setThreshold(0.5);
        SVMModel SVM_road_Model = SVMWithSGD.train(road_labeled_SVM.rdd(), 200, 0.1, 0.01).setThreshold(0.5);
        SVMModel SVM_vegetation_Model = SVMWithSGD.train(vegetation_labeled_SVM.rdd(), 200, 0.1, 0.01).setThreshold(0.5);
        SVMModel SVM_water_Model = SVMWithSGD.train(water_labeled_SVM.rdd(), 200, 0.1, 0.01).setThreshold(0.5);
        long train_end = System.currentTimeMillis();
        long train_time = train_end - train_start;
        System.out.println("Total time for train phase = " + train_time + "  Milisec\t" + (train_time / 1000) + "  Seconds\t" + train_time / 60000 + "  Minutes");


        // This Part calculates the accuracy of the SVM model in two ways one is classical split 70-30 train test way
        // second is more robust cross validation way

        double SVM_Alongside_Accuracy = SVM_Accuracy(alongside_labeled_SVM, "Alongside");
        double SVM_Building_Accuracy = SVM_Accuracy(building_labeled_SVM, "Building");
        double SVM_Road_Accuracy = SVM_Accuracy(road_labeled_SVM, "Road");
        double SVM_Vegetation_Accuracy = SVM_Accuracy(vegetation_labeled_SVM, "Vegetation");
        double SVM_Water_Accuracy = SVM_Accuracy(water_labeled_SVM, "Water");

        double[] SVM_Accuracy = {SVM_Alongside_Accuracy, SVM_Building_Accuracy, SVM_Road_Accuracy, SVM_Vegetation_Accuracy, SVM_Water_Accuracy};
        // double[] SVM_CV_Accuracy = Cross_Validation.SVM_CV_Accuracy(alongside_pos, alongside_neg, building_pos, building_neg, road_pos, road_neg, vegetation_pos, vegetation_neg, water_pos, water_neg, jsc);
        // Charting.SVM_Model_Accuracy_Chart(SVM_Accuracy, SVM_CV_Accuracy, path);

        SVMModel SVM_Models[] = {SVM_Alongside_Model, SVM_building_Model, SVM_road_Model, SVM_vegetation_Model, SVM_water_Model};
        return SVM_Models;
    }

    private static double SVM_Accuracy(JavaRDD<LabeledPoint> labeled_raw, String identifier) {

        JavaRDD<LabeledPoint>[] splits = labeled_raw.randomSplit(new double[]{0.7, 0.3}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];

        SVMModel model = SVMWithSGD.train(training.rdd(), 200, 0.1, 0.01).setThreshold(0.5);
        model.clearThreshold();
        JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test
                .map(p -> new Tuple2<>(model.predict(p.features()), p.label()));
        BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
        double SVM_classic_accuracy = 100 * metrics.areaUnderROC();
        System.out.println("SVM " + identifier + " Accuracy based on %70-%30 splitting is = \t" + SVM_classic_accuracy + "\n");
        return SVM_classic_accuracy;
    }
}

/*
        JavaRDD<LabeledPoint>[] svm_along_pos = splitting_10(alongside_pos, jsc);
        JavaRDD<LabeledPoint>[] svm_along_neg = splitting_10(alongside_neg, jsc);
        JavaRDD<LabeledPoint>[] svm_build_pos = splitting_10(building_pos, jsc);
        JavaRDD<LabeledPoint>[] svm_build_neg = splitting_10(building_neg, jsc);
        JavaRDD<LabeledPoint>[] svm_road_pos = splitting_10(road_pos, jsc);
        JavaRDD<LabeledPoint>[] svm_road_neg = splitting_10(road_neg, jsc);
        JavaRDD<LabeledPoint>[] svm_vegetation_pos = splitting_10(vegetation_pos, jsc);
        JavaRDD<LabeledPoint>[] svm_vegetation_neg = splitting_10(vegetation_neg, jsc);
        JavaRDD<LabeledPoint>[] svm_water_pos = splitting_10(water_pos, jsc);
        JavaRDD<LabeledPoint>[] svm_water_neg = splitting_10(water_neg, jsc);


        double SVM_CV_Alongside_Result = Evaluation_Section(svm_along_pos, svm_build_neg, svm_road_neg,
                svm_vegetation_neg, svm_water_neg,  "Alongside Cross Validation", "SVM",jsc);

        double SVM_CV_Build_Result = Evaluation_Section(svm_along_neg, svm_build_pos, svm_road_neg, svm_vegetation_neg,
                svm_water_neg,  "Building Cross Validation", "SVM",jsc);

        double SVM_CV_Road_Result = Evaluation_Section(svm_along_neg, svm_build_neg, svm_road_pos, svm_vegetation_neg,
                svm_water_neg,  "Road Cross Validation", "SVM",jsc);

        double SVM_CV_Vegetation_Result = Evaluation_Section(svm_along_neg, svm_build_neg, svm_road_neg,
                svm_vegetation_pos, svm_water_neg,  "Vegetation Cross Validation", "SVM",jsc);

        double SVM_CV_Water_Result = Evaluation_Section(svm_along_neg, svm_build_neg, svm_road_neg, svm_vegetation_neg,
                svm_water_pos,  "Water Cross Validation", "SVM",jsc);

        double[] SVM_CV_Results = {SVM_CV_Alongside_Result, SVM_CV_Build_Result, SVM_CV_Road_Result,
                SVM_CV_Vegetation_Result, SVM_CV_Water_Result};
 */