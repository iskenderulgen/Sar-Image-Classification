package Sar_Image_Analysis;

import Sar_Image_Analysis.ML_Models.Naive_Bayes_Modelling;
import Sar_Image_Analysis.ML_Models.SVM_Modelling;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.util.LongAccumulator;

public class Main_Sar_Analysis {
    /* This Section Contains The Bigger Feature Vector Files To Test Larger Environment. It Also Splittet %60-%40
       then Analyzed to measure Evaluation Metrics. Also each  set trained and predicted over total Big set to measure
       real accuaricy. Result will be given when code executed */
    private static final String Main_Path = "C:\\Users\\ULGEN\\Documents\\Idea_Workspace\\Sar_Classify\\Sar_Datasets\\";
    private static final String alongside_path = Main_Path + "alongside\\training\\alongside.csv";
    private static final String building_path = Main_Path + "building\\training\\building.csv";
    private static final String road_path = Main_Path + "road\\training\\road.csv";
    private static final String vegetation_path = Main_Path + "vegetation\\training\\vegetation.csv";
    private static final String water_path = Main_Path + "water\\training\\water.csv";
    private static final String Sar_image_set1K = Main_Path + "Noised_Big_Sets\\Total_Set_1K.csv";
    private static final String Sar_image_set10K = Main_Path + "Noised_Big_Sets\\Total_Set_10K.csv";
    private static final String Sar_image_set100K = Main_Path + "Noised_Big_Sets\\Total_Set_100K.csv";
    private static final String Sar_image_set1000K = Main_Path + "Noised_Big_Sets\\Total_Set_1M.csv";
    private static final String Sar_image_set10M = Main_Path + "Noised_Big_Sets\\Total_Set_10M.csv";
    private static final String Sar_image_set90M = Main_Path + "Noised_Big_Sets\\Total_Sar_Set.csv";

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("Sar_Analysis");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        jsc.setLogLevel("ERROR");

        LongAccumulator accumulator_alongside = jsc.sc().longAccumulator();
        LongAccumulator accumulator_building = jsc.sc().longAccumulator();
        LongAccumulator accumulator_road = jsc.sc().longAccumulator();
        LongAccumulator accumulator_vegetation = jsc.sc().longAccumulator();
        LongAccumulator accumulator_water = jsc.sc().longAccumulator();
 /*
        NaiveBayesModel NB_Model = Naive_Bayes_Modelling.NB_Modelling(alongside_path, building_path, road_path,
                vegetation_path, water_path, jsc);

        Analysis_Section.NaiveBayes_Analysis("1K Sar Set", Sar_image_set1K, NB_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        Analysis_Section.NaiveBayes_Analysis("10K Sar Set", Sar_image_set10K, NB_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        Analysis_Section.NaiveBayes_Analysis("100K Sar Set", Sar_image_set100K, NB_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        Analysis_Section.NaiveBayes_Analysis("1000K Sar Set", Sar_image_set1000K, NB_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        Analysis_Section.NaiveBayes_Analysis("10M Sar Set", Sar_image_set10M, NB_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        Analysis_Section.NaiveBayes_Analysis("90M Sar Set", Sar_image_set90M, NB_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);*/

        SVMModel[] SVM_Models = SVM_Modelling.SVM_Model(alongside_path, building_path, road_path, vegetation_path,
                water_path, jsc);

        String ident[] = {"Alongside", "building", "road", "vegetation", "water"};
        for (int i = 0; i < 5; i++) {
            Analysis_Section.SVM_Analysis(SVM_Models, "1K Sar Set", ident[i], Sar_image_set1K, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc, i);
            Analysis_Section.SVM_Analysis(SVM_Models, "10K Sar Set", ident[i], Sar_image_set10K, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc, i);
            Analysis_Section.SVM_Analysis(SVM_Models, "100K Sar Set", ident[i], Sar_image_set100K, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc, i);
            Analysis_Section.SVM_Analysis(SVM_Models, "1000K Sar Set", ident[i], Sar_image_set1000K, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc, i);
            Analysis_Section.SVM_Analysis(SVM_Models, "10M Sar Set", ident[i], Sar_image_set10M, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc, i);
            Analysis_Section.SVM_Analysis(SVM_Models, "100M Sar Set", ident[i], Sar_image_set90M, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc, i);
        }
    }
}
//MLUtils.saveAsLibSVMFile(collected_labels.rdd(),"C:\\Users\\ULGEN\\Documents\\IdeaWorkspace\\Sar_Classify\\Sar_New_vectorcollected");