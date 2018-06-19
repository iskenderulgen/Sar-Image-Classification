package Sar_Image_Analysis.ML_Models;

import Sar_Image_Analysis.Cross_Validation_Sar;
import Sar_Image_Analysis.Pre_Process;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

public class SVM_Modelling {
    // One vs Rest Approach used in this modelling
    public static SVMModel[] SVM_Model (String alongside_path, String building_path, String road_path,
                                        String vegetation_path, String water_path, JavaSparkContext jsc) {

        long Labeling_Clock_Starts = System.currentTimeMillis();
        System.out.println("\nSVM Labeling Phase Starts");
        JavaRDD<LabeledPoint> alongside_pos = Pre_Process.labelingdata(alongside_path, 1.0, jsc);
        JavaRDD<LabeledPoint> alongside_neg = Pre_Process.labelingdata(alongside_path, 0.0, jsc);

        JavaRDD<LabeledPoint> building_pos = Pre_Process.labelingdata(building_path,1.0,jsc);
        JavaRDD<LabeledPoint> building_neg = Pre_Process.labelingdata(building_path,0.0,jsc);

        JavaRDD<LabeledPoint> road_pos = Pre_Process.labelingdata(road_path,1.0,jsc);
        JavaRDD<LabeledPoint> road_neg = Pre_Process.labelingdata(road_path,0.0,jsc);

        JavaRDD<LabeledPoint> vegetation_pos = Pre_Process.labelingdata(vegetation_path,1.0,jsc);
        JavaRDD<LabeledPoint> vegetation_neg = Pre_Process.labelingdata(vegetation_path,0.0,jsc);

        JavaRDD<LabeledPoint> water_pos = Pre_Process.labelingdata(water_path,1.0,jsc);
        JavaRDD<LabeledPoint> water_neg = Pre_Process.labelingdata(water_path,0.0,jsc);

        JavaRDD<LabeledPoint> alongside_labeled_SVM = alongside_pos.union(building_neg).union(road_neg).union(vegetation_neg).union(water_neg);
        JavaRDD<LabeledPoint> building_labeled_SVM =  alongside_neg.union(building_pos).union(road_neg).union(vegetation_neg).union(water_neg);
        JavaRDD<LabeledPoint> road_labeled_SVM = alongside_neg.union(building_neg).union(road_pos).union(vegetation_neg).union(water_neg);
        JavaRDD<LabeledPoint> vegetation_labeled_SVM = alongside_neg.union(building_neg).union(road_neg).union(vegetation_pos).union(water_neg);
        JavaRDD<LabeledPoint> water_labeled_SVM = alongside_neg.union(building_neg).union(road_neg).union(vegetation_neg).union(water_pos);
        System.out.println("SVM Labeling Phase Ends\n");
        long Labeling_Clock_Ends = System.currentTimeMillis();
        long labeling_time = Labeling_Clock_Ends - Labeling_Clock_Starts;
        System.out.println("Total taime elapsed for SVM labeling section is =  " + labeling_time + "  Milisec\t" + (labeling_time / 1000) + "  Seconds\t" + labeling_time / 60000 + "  Minutes");


        long train_start = System.currentTimeMillis();
        SVMModel SVM_Alongside_Model =  SVMWithSGD.train(alongside_labeled_SVM.rdd(),100).setThreshold(0.5);
        SVMModel SVM_building_Model =   SVMWithSGD.train(building_labeled_SVM.rdd(),100).setThreshold(0.5);
        SVMModel SVM_road_Model =  SVMWithSGD.train(road_labeled_SVM.rdd(),100).setThreshold(0.5);
        SVMModel SVM_vegetation_Model =  SVMWithSGD.train(vegetation_labeled_SVM.rdd(),100).setThreshold(0.5);
        SVMModel SVM_water_Model =  SVMWithSGD.train(water_labeled_SVM.rdd(),100).setThreshold(0.5);
        long train_end = System.currentTimeMillis();
        long train_time = train_end - train_start;
        System.out.println("Total time for train phase = "+train_time+"  Milisec\t" + (train_time / 1000) + "  Seconds\t" + train_time / 60000 + "  Minutes");

        //Since we don't use returning value, we can just call the function and print the results
        SVM_Accuracy(alongside_labeled_SVM,"Alongside");
        SVM_Accuracy(building_labeled_SVM,"Building");
        SVM_Accuracy(road_labeled_SVM,"Road");
        SVM_Accuracy(vegetation_labeled_SVM,"Vegetation");
        SVM_Accuracy(water_labeled_SVM,"Water");

        Cross_Validation_Sar.SVM_CV_Accuracy(alongside_pos,alongside_neg,building_pos,building_neg,road_pos,road_neg,vegetation_pos,vegetation_neg,water_pos,water_neg,jsc,"SVM 10 Cross Validation","SVM");

        SVMModel SVM_Models []={SVM_Alongside_Model,SVM_building_Model,SVM_road_Model,SVM_vegetation_Model,SVM_water_Model};
        return SVM_Models;
    }

    private static void SVM_Accuracy (JavaRDD<LabeledPoint> labeledraw, String identifier){

        JavaRDD<LabeledPoint>[] splits = labeledraw.randomSplit(new double[]{0.7, 0.3}, 11L);
        JavaRDD<LabeledPoint> training = splits[0].cache();
        JavaRDD<LabeledPoint> test = splits[1];

        SVMModel model = SVMWithSGD.train(training.rdd(), 100).setThreshold(0.5);
        JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test
                .map(p ->new Tuple2<>(model.predict(p.features()), p.label()));
        BinaryClassificationMetrics metrics =new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
        double auROC = metrics.areaUnderROC();
        System.out.println("SVM "+ identifier + " Accuracy based on %70-%30 splitting is = \t" + (100*auROC)+"\n");
    }
}
