package Sar_Image_Analysis;

import Sar_Image_Analysis.ML_Models.DTree;
import Sar_Image_Analysis.ML_Models.NB_Model_Train;
import Sar_Image_Analysis.ML_Models.RF_Tree;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.util.LongAccumulator;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


public class Main_Sar_Analysis {

    private static final String Main_Path = "C:\\Users\\ULGEN\\Documents\\Idea_Workspace\\Sar_Classify\\Sar_Datasets\\";
    private static final String Results_Path = "C:\\Users\\ULGEN\\Desktop\\Total_Results\\";
    private static final String alongside_path = Main_Path + "Categories\\alongside\\training\\alongside_denoised_5.csv";
    private static final String building_path = Main_Path + "Categories\\building\\training\\building_denoised_5.csv";
    private static final String road_path = Main_Path + "Categories\\road\\training\\road_denoised_5.csv";
    private static final String vegetation_path = Main_Path + "Categories\\vegetation\\training\\vegetation_denoised_5.csv";
    private static final String water_path = Main_Path + "Categories\\water\\training\\water_denoised_5.csv";
    private static final String SAR_1_Img = Main_Path + "Denoised_Big_Set\\Total_Set_1.csv";
    private static final String SAR_10_Img = Main_Path + "Denoised_Big_Set\\total_set_16k.csv";
    private static final String SAR_100_Img = Main_Path + "Denoised_Big_Set\\total_set_166k.csv";
    private static final String SAR_1K_Img = Main_Path + "Denoised_Big_Set\\total_set_1.6M.csv";
    private static final String SAR_10K_Img = Main_Path + "Denoised_Big_Set\\total_set_16M.csv";
    private static final String Sar_100K_Img = Main_Path + "Denoised_Big_Set\\total_set_166M.csv";

    public static void main(String[] args) throws IOException {
        SparkConf conf = new SparkConf().setMaster("local[4]").setAppName("Sar_Analysis")
                .set("spark.driver.memory", "5g")
                .set("spark.executor.memory", "1g");
        //.set("spark.serializer","org.apache.spark.serializer.KryoSerializer");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        jsc.setLogLevel("ERROR");
        long Total_Analysis_Time_Start = System.currentTimeMillis();

        LongAccumulator acc_alongside = jsc.sc().longAccumulator();
        LongAccumulator acc_building = jsc.sc().longAccumulator();
        LongAccumulator acc_road = jsc.sc().longAccumulator();
        LongAccumulator acc_vegetation = jsc.sc().longAccumulator();
        LongAccumulator acc_water = jsc.sc().longAccumulator();

        //In here we define the most used variable as lists or instances to pass it to everywhere and
        //be able to manipulate from one location
        String[] Training_Paths = {alongside_path, building_path, road_path, vegetation_path, water_path};

        double NB_Lambda = 1.0;
        int Tree_NumClass = 6;
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        String impurity = "gini";
        int Tree_MaxDepth = 10;
        int Tree_MaxBins = 32;
        int RF_Tree_NumTree = 10;
        String RF_Feature_Subset = "auto";

        //PrintStream outputs = new PrintStream(new FileOutputStream(new File(Results_Path+"Total_results.txt")));
        //System.setOut(outputs);

        JavaRDD<Vector> Vectorized_1_Img = Pre_Process.vectorised(jsc, SAR_1_Img).cache();
        JavaRDD<Vector> Vectorized_10_Img = Pre_Process.vectorised(jsc, SAR_10_Img).cache();
        JavaRDD<Vector> Vectorized_100_Img = Pre_Process.vectorised(jsc, SAR_100_Img).cache();
        JavaRDD<Vector> Vectorized_1K_Img = Pre_Process.vectorised(jsc, SAR_1K_Img).cache();
        JavaRDD<Vector> Vectorized_10K_Img = Pre_Process.vectorised(jsc, SAR_10K_Img).cache();
        JavaRDD<Vector> Vectorized_100K_Img = Pre_Process.vectorised(jsc, Sar_100K_Img).cache();

        System.out.println("\n////////////////////////////  NAIVE BAYES STARTS HERE  ///////////////////////////////\n");
        NaiveBayesModel NB_Model = NB_Model_Train.NB_Modelling(Training_Paths, jsc, NB_Lambda, Results_Path);
        //Array contains 5 results respected to Analysis order and analysis time at the end of the array
        double[] NB_Result_1 = Analysis_Section.NaiveBayes_Analysis("1 Sar Img", Vectorized_1_Img,
                NB_Model, acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] NB_Result_10 = Analysis_Section.NaiveBayes_Analysis("10 Sar Img", Vectorized_10_Img,
                NB_Model, acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] NB_Result_100 = Analysis_Section.NaiveBayes_Analysis("100 Sar Img", Vectorized_100_Img,
                NB_Model, acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] NB_Result_1K = Analysis_Section.NaiveBayes_Analysis("1K Sar Img", Vectorized_1K_Img,
                NB_Model, acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] NB_Result_10K = Analysis_Section.NaiveBayes_Analysis("10K Sar Img", Vectorized_10K_Img,
                NB_Model, acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] NB_Result_100K = Analysis_Section.NaiveBayes_Analysis("100K Sar Img", Vectorized_100K_Img,
                NB_Model, acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        Charting.Result_Chart("Naive_Bayes", NB_Result_1, NB_Result_10, NB_Result_100, NB_Result_1K,
                NB_Result_10K, NB_Result_100K, Results_Path);
        System.out.println("\n////////////////////////////    NAIVE BAYES ENDS HERE //////////////////////////////\n");


        System.out.println("\n////////////////////////   DECISION TREE STARTS HERE  //// //////////////////////////\n");
        DecisionTreeModel DTree_Model = DTree.DTree_Model(Training_Paths, jsc, categoricalFeaturesInfo, Tree_NumClass,
                impurity, Tree_MaxDepth, Tree_MaxBins, Results_Path);
        double[] DTree_Result_1 = Analysis_Section.DTree_Analysis("1 Sar Img", Vectorized_1_Img, DTree_Model,
                acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] DTree_Result_10 = Analysis_Section.DTree_Analysis("10 Sar Img", Vectorized_10_Img, DTree_Model,
                acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] DTree_Result_100 = Analysis_Section.DTree_Analysis("100 Sar Img", Vectorized_100_Img, DTree_Model,
                acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] DTree_Result_1K = Analysis_Section.DTree_Analysis("1K Sar Img", Vectorized_1K_Img, DTree_Model,
                acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] DTree_Result_10K = Analysis_Section.DTree_Analysis("10K Sar Img", Vectorized_10K_Img, DTree_Model,
                acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] DTree_Result_100K = Analysis_Section.DTree_Analysis("100K Sar Img", Vectorized_100K_Img, DTree_Model,
                acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        Charting.Result_Chart("Decision_Tree", DTree_Result_1, DTree_Result_10, DTree_Result_100,
                DTree_Result_1K, DTree_Result_10K, DTree_Result_100K, Results_Path);
        System.out.println("\n////////////////////////////   DECISION TREE ENDS HERE      ///////////////////////\n");


        System.out.println("\n//////////////////////   RANDOM FOREST TREE STARTS HERE      ////////////////////////\n");
        RandomForestModel RFTree_Model = RF_Tree.RF_Tree_Modelling(Training_Paths, jsc, categoricalFeaturesInfo,
                Tree_NumClass, RF_Tree_NumTree, RF_Feature_Subset, impurity, Tree_MaxDepth, Tree_MaxBins, Results_Path);
        double[] RFTree_Result_1 = Analysis_Section.RF_Tree_Analysis("1 Sar Set", Vectorized_1_Img,
                RFTree_Model, acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] RFTree_Result_10 = Analysis_Section.RF_Tree_Analysis("10 Sar Set", Vectorized_10_Img,
                RFTree_Model, acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] RFTree_Result_100 = Analysis_Section.RF_Tree_Analysis("100 Sar Set", Vectorized_100_Img,
                RFTree_Model, acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] RF_tree_1M_Results = Analysis_Section.RF_Tree_Analysis("1K Sar Set", Vectorized_1K_Img,
                RFTree_Model, acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] RF_tree_10M_Results = Analysis_Section.RF_Tree_Analysis("10K Sar Set", Vectorized_10K_Img,
                RFTree_Model, acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        double[] RF_tree_90M_Results = Analysis_Section.RF_Tree_Analysis("100K Sar Set", Vectorized_100K_Img,
                RFTree_Model, acc_alongside, acc_building, acc_road, acc_vegetation, acc_water);
        Charting.Result_Chart("Random_Forest_Tree", RFTree_Result_1, RFTree_Result_10, RFTree_Result_100,
                RF_tree_1M_Results, RF_tree_10M_Results, RF_tree_90M_Results, Results_Path);
        System.out.println("\n//////////////////////   RANDOM FOREST TREE ENDS HERE    ////////////////////////////\n");

/*
        double[] NB_Time = {NB_Result_1[5], NB_Result_10[5], NB_Result_100[5], NB_Result_1K[5], NB_Result_10K[5],
                NB_Result_100K[5]};

        double[] DTree_Time = {DTree_Result_1[5], DTree_Result_10[5], DTree_Result_100[5], DTree_Result_1K[5],
                DTree_Result_10K[5], DTree_Result_100K[5]};

        double[] RFTree_Time = {RFTree_Result_1[5], RFTree_Result_10[5], RFTree_Result_100[5], RF_tree_1M_Results[5],
                RF_tree_10M_Results[5], RF_tree_90M_Results[5]};

        Charting.Analysis_time_chart(NB_Time, DTree_Time, RFTree_Time, Results_Path);

*/
        long Total_Analysis_Time_Ends = System.currentTimeMillis();
        long analysis_time = Total_Analysis_Time_Ends - Total_Analysis_Time_Start;
        System.out.println("Total taime elapsed is =  " + analysis_time + "  Milisec\t" + (analysis_time / 1000) + "  Seconds\t" + analysis_time / 60000 + "  Minutes");
    }
}
//MLUtils.saveAsLibSVMFile(collected_labels.rdd(),"C:\\Users\\ULGEN\\Documents\\IdeaWorkspace\\Sar_Classify\\Sar_New_vectorcollected");
//System.out.println("labeled data'nın veri tipi ="+labeleddata.getClass().getName());


/*
        SVMModel[] SVM_Models = SVM_Model_Train.SVM_Model(alongside_path, building_path, road_path, vegetation_path,
                water_path, jsc,Results_Path);

        String[] Model_Identifier = {"Alongside", "building", "road", "vegetation", "water"};
        double[] SVM_1K_Results = Analysis_Section.SVM_Analysis(SVM_Models, "1K Sar Set", Model_Identifier,
                Vectorized_1_Pic, acc_alongside, acc_building, acc_road,
                acc_vegetation, acc_water);
        double[] SVM_10K_Results = Analysis_Section.SVM_Analysis(SVM_Models, "10K Sar Set", Model_Identifier,
                Vectorized_10_Pic, acc_alongside, acc_building, acc_road,
                acc_vegetation, acc_water);
        double[] SVM_100K_Results = Analysis_Section.SVM_Analysis(SVM_Models, "100K Sar Set", Model_Identifier,
                Vectorized_100_Pic, acc_alongside, acc_building, acc_road,
                acc_vegetation, acc_water);
        double[] SVM_1M_Results = Analysis_Section.SVM_Analysis(SVM_Models, "1M Sar Set", Model_Identifier,
                Vectorized_1K_Pic, acc_alongside, acc_building, acc_road,
                acc_vegetation, acc_water);
        double[] SVM_10M_Results = Analysis_Section.SVM_Analysis(SVM_Models, "10M Sar Set", Model_Identifier,
                Vectorized_10K_Pic, acc_alongside, acc_building, acc_road,
                acc_vegetation, acc_water);
        double[] SVM_90M_Results = Analysis_Section.SVM_Analysis(SVM_Models, "90M Sar Set", Model_Identifier,
                Vectorized_100K_Pic, acc_alongside, acc_building, acc_road,
                acc_vegetation, acc_water);

        Charting.Result_Chart("SVM", SVM_1K_Results, SVM_10K_Results, SVM_100K_Results, SVM_1M_Results,
                SVM_10M_Results, SVM_90M_Results,Results_Path);

        double[] SVM_Time = {SVM_1K_Results[5], SVM_10K_Results[5],SVM_100K_Results[5],SVM_1M_Results[5],
                SVM_10M_Results[5],SVM_90M_Results[5]};
*/
