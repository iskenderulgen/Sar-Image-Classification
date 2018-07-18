package Sar_Image_Analysis;

import Sar_Image_Analysis.ML_Models.DTree;
import Sar_Image_Analysis.ML_Models.NB_Model_Train;
import Sar_Image_Analysis.ML_Models.SVM_Model_Train;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.util.LongAccumulator;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;



public class Main_Sar_Analysis {

    private static final String Main_Path = "C:\\Users\\ULGEN\\Documents\\Idea_Workspace\\Sar_Classify\\Sar_Datasets\\";
    private static final String Results_Path = "C:\\Users\\ULGEN\\Desktop\\Total_Results\\";
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

    public static void main(String[] args) throws IOException {
        SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("Sar_Analysis");
            //.set("spark.serializer","org.apache.spark.serializer.KryoSerializer");
            //.set("spark.driver.memory","24g")
            //.set("spark.executor.memory","1g");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        jsc.setLogLevel("ERROR");

        long Total_Analysis_Time_Start = System.currentTimeMillis();

        PrintStream outputs = new PrintStream(new FileOutputStream(new File(Results_Path+"Total_results.txt")));
        System.setOut(outputs);

        LongAccumulator accumulator_alongside = jsc.sc().longAccumulator();
        LongAccumulator accumulator_building = jsc.sc().longAccumulator();
        LongAccumulator accumulator_road = jsc.sc().longAccumulator();
        LongAccumulator accumulator_vegetation = jsc.sc().longAccumulator();
        LongAccumulator accumulator_water = jsc.sc().longAccumulator();



        NaiveBayesModel NB_Model = NB_Model_Train.NB_Modelling(alongside_path, building_path, road_path,
                vegetation_path, water_path, jsc, 1.0,Results_Path);

        long[] NB_Time = new long[6];
        Analysis_Section.NaiveBayes_Analysis("1K Sar Set", Sar_image_set1K, NB_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] NB_K1_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        NB_Time[0] = Analysis_Section.analysis_time;
        Analysis_Section.NaiveBayes_Analysis("10K Sar Set", Sar_image_set10K, NB_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] NB_K10_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        NB_Time[1] = Analysis_Section.analysis_time;
        Analysis_Section.NaiveBayes_Analysis("100K Sar Set", Sar_image_set100K, NB_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] NB_K100_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        NB_Time[2] = Analysis_Section.analysis_time;
        Analysis_Section.NaiveBayes_Analysis("1000K Sar Set", Sar_image_set1000K, NB_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] NB_K1000_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        NB_Time[3] = Analysis_Section.analysis_time;
        Analysis_Section.NaiveBayes_Analysis("10M Sar Set", Sar_image_set10M, NB_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] NB_M10_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        NB_Time[4] = Analysis_Section.analysis_time;
        Analysis_Section.NaiveBayes_Analysis("90M Sar Set", Sar_image_set90M, NB_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] NB_M90_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        NB_Time[5] = Analysis_Section.analysis_time;
        Charting.Result_Chart("Naive_Bayes", NB_K1_Results, NB_K10_Results, NB_K100_Results, NB_K1000_Results, NB_M10_Results, NB_M90_Results,Results_Path);





        SVMModel[] SVM_Models = SVM_Model_Train.SVM_Model(alongside_path, building_path, road_path, vegetation_path,
                water_path, jsc,Results_Path);

        long[] SVM_Time = new long[6];
        String[] Model_Identifier = {"Alongside", "building", "road", "vegetation", "water"};
        Analysis_Section.SVM_Analysis(SVM_Models, "1K Sar Set", Model_Identifier, Sar_image_set1K, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] SVM_K1_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        SVM_Time[0] = Analysis_Section.analysis_time;
        Analysis_Section.SVM_Analysis(SVM_Models, "10K Sar Set", Model_Identifier, Sar_image_set10K, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] SVM_K10_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        SVM_Time[1] = Analysis_Section.analysis_time;
        Analysis_Section.SVM_Analysis(SVM_Models, "100K Sar Set", Model_Identifier, Sar_image_set100K, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] SVM_K100_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        SVM_Time[2] = Analysis_Section.analysis_time;
        Analysis_Section.SVM_Analysis(SVM_Models, "1000K Sar Set", Model_Identifier, Sar_image_set1000K, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] SVM_K1000_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        SVM_Time[3] = Analysis_Section.analysis_time;
        Analysis_Section.SVM_Analysis(SVM_Models, "10M Sar Set", Model_Identifier, Sar_image_set10M, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] SVM_M10_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        SVM_Time[4] = Analysis_Section.analysis_time;
        Analysis_Section.SVM_Analysis(SVM_Models, "90M Sar Set", Model_Identifier, Sar_image_set90M, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] SVM_M90_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        SVM_Time[5] = Analysis_Section.analysis_time;
        Charting.Result_Chart("SVM", SVM_K1_Results, SVM_K10_Results, SVM_K100_Results, SVM_K1000_Results, SVM_M10_Results, SVM_M90_Results,Results_Path);




        DecisionTreeModel DTree_Model = DTree.Dtree_Model(alongside_path,building_path,road_path,vegetation_path,water_path,jsc,Results_Path);

        long[] Dtree_Time = new long[6];
        Analysis_Section.DTree_Analysis("1K Sar Set", Sar_image_set1K, DTree_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] Dtree_K1_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        Dtree_Time[0] = Analysis_Section.analysis_time;
        Analysis_Section.DTree_Analysis("10K Sar Set", Sar_image_set10K, DTree_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] Dtree_K10_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        Dtree_Time[1] = Analysis_Section.analysis_time;
        Analysis_Section.DTree_Analysis("100K Sar Set", Sar_image_set100K, DTree_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] Dtree_K100_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        Dtree_Time[2] = Analysis_Section.analysis_time;
        Analysis_Section.DTree_Analysis("1000K Sar Set", Sar_image_set1000K, DTree_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] Dtree_K1000_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        Dtree_Time[3] = Analysis_Section.analysis_time;
        Analysis_Section.DTree_Analysis("10M Sar Set", Sar_image_set10M, DTree_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] Dtree_M10_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        Dtree_Time[4] = Analysis_Section.analysis_time;
        Analysis_Section.DTree_Analysis("90M Sar Set", Sar_image_set90M, DTree_Model, accumulator_alongside, accumulator_building, accumulator_road, accumulator_vegetation, accumulator_water, jsc);
        double[] Dtree_M90_Results = {Analysis_Section.along_result, Analysis_Section.build_result, Analysis_Section.road_result, Analysis_Section.vegetation_result, Analysis_Section.water_result,};
        Dtree_Time[5] = Analysis_Section.analysis_time;
        Charting.Result_Chart("Decision_Tree", Dtree_K1_Results, Dtree_K10_Results, Dtree_K100_Results, Dtree_K1000_Results, Dtree_M10_Results, Dtree_M90_Results,Results_Path);


        Charting.Analysis_time_chart(NB_Time, SVM_Time,Dtree_Time,Results_Path);


        long Total_Analysis_Time_Ends = System.currentTimeMillis();
        long analysis_time = Total_Analysis_Time_Ends - Total_Analysis_Time_Start;
        System.out.println("Total taime elapsed is =  " + analysis_time + "  Milisec\t" + (analysis_time / 1000) + "  Seconds\t" + analysis_time / 60000 + "  Minutes");
    }
}
//MLUtils.saveAsLibSVMFile(collected_labels.rdd(),"C:\\Users\\ULGEN\\Documents\\IdeaWorkspace\\Sar_Classify\\Sar_New_vectorcollected");
//System.out.println("labeled data'nın veri tipi ="+labeleddata.getClass().getName());