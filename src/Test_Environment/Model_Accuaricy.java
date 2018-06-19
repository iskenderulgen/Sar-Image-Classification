package Test_Environment;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

public class Model_Accuaricy {

    /* This Section Contains The 5x5 Feature Vector Files To Test Minimal Environment. It Also Splittet %60-%40
       then Analyzed to measure Evaluation Metrics. Also each 5 set trained and predicted over total 25 set to measure
       real accuaricy. Result will be given when code executed */
    private static final String Main_Path = "C:\\Users\\ULGEN\\Documents\\Idea_Workspace\\Sar_Classify\\Sar_Datasets\\";

    private static final String alongside_path      = Main_Path + "alongside\\training\\alongside.csv";
    private static final String building_path       = Main_Path + "building\\training\\building.csv";
    private static final String road_path           = Main_Path + "road\\training\\road.csv";
    private static final String vegetation_path     = Main_Path + "vegetation\\training\\vegetation.csv";
    private static final String water_path          = Main_Path + "water\\training\\water.csv";

    private static final String alongside_path_5    = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised5\\alongside_denoised.csv";
    private static final String building_path_5     = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised5\\building_denoised.csv";
    private static final String road_path_5         = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised5\\road_denoised.csv";
    private static final String vegetation_path_5   = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised5\\vegetation_denoised.csv";
    private static final String water_path_5        = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised5\\water_denoised.csv";

    private static final String alongside_path_10   = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised10\\alongside_denoised.csv";
    private static final String building_path_10    = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised10\\building_denoised.csv";
    private static final String road_path_10        = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised10\\road_denoised.csv";
    private static final String vegetation_path_10  = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised10\\vegetation_denoised.csv";
    private static final String water_path_10       = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised10\\water_denoised.csv";


    private static final String alongside_path_15   = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised15\\alongside_denoised.csv";
    private static final String building_path_15    = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised15\\building_denoised.csv";
    private static final String road_path_15        = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised15\\road_denoised.csv";
    private static final String vegetation_path_15  = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised15\\vegetation_denoised.csv";
    private static final String water_path_15       = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised15\\water_denoised.csv";

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("Sar_Analysis");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example")
                .config(conf)
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

        JavaRDD<LabeledPoint> totaled_set_noised = unionized_label(alongside_path,building_path,road_path,vegetation_path,water_path,sc," Niosed Data   ");
        JavaRDD<LabeledPoint> totaled_set_5 = unionized_label(alongside_path_5,building_path_5,road_path_5,vegetation_path_5,water_path_5,sc," Denoise Factor 5   ");
        JavaRDD<LabeledPoint> totaled_set_10 = unionized_label(alongside_path_10,building_path_10,road_path_10,vegetation_path_10,water_path_10,sc," Denoise Factor 10   ");
        JavaRDD<LabeledPoint> totaled_set_15 = unionized_label(alongside_path_15,building_path_15,road_path_15,vegetation_path_15,water_path_15,sc," Denoise Factor 15    ");

        double lambda_value  = 1.0;
        double train_value_1 = 0.6; double train_value_2 = 0.7; double train_value_3 = 0.8;
        double test_value_1  = 0.4; double test_value_2  = 0.3; double test_value_3  = 0.2;

        System.out.println("\n//////////////////////////////////////////////////////\n");
        System.out.println("Noised Data Accuracy Parameters Train = 0.6 , Test = 0.4  =  "+NB_Accuaricy(totaled_set_noised,lambda_value,train_value_1,test_value_1));
        System.out.println("Noised Data Accuracy Parameters Train = 0.7 , Test = 0.3  =  "+NB_Accuaricy(totaled_set_noised,lambda_value,train_value_2,test_value_2));
        System.out.println("Noised Data Accuracy Parameters Train = 0.8 , Test = 0.2  =  "+NB_Accuaricy(totaled_set_noised,lambda_value,train_value_3,test_value_3));

        System.out.println("\n//////////////////////////////////////////////////////\n");
        System.out.println("Denoise Factor 5 Accuracy Parameters Train = 0.6 , Test = 0.4  =  "+NB_Accuaricy(totaled_set_5,lambda_value,train_value_1,test_value_1));
        System.out.println("Denoise Factor 5 Accuracy Parameters Train = 0.7 , Test = 0.3  =  "+NB_Accuaricy(totaled_set_5,lambda_value,train_value_2,test_value_2));
        System.out.println("Denoise Factor 5 Accuracy Parameters Train = 0.8 , Test = 0.2  =  "+NB_Accuaricy(totaled_set_5,lambda_value,train_value_3,test_value_3));

        System.out.println("\n//////////////////////////////////////////////////////\n");
        System.out.println("Denoise Factor 10 Accuracy Parameters Train = 0.6 , Test = 0.4  =  "+NB_Accuaricy(totaled_set_10,lambda_value,train_value_1,test_value_1));
        System.out.println("Denoise Factor 10 Accuracy Parameters Train = 0.7 , Test = 0.3  =  "+NB_Accuaricy(totaled_set_10,lambda_value,train_value_2,test_value_2));
        System.out.println("Denoise Factor 10 Accuracy Parameters Train = 0.8 , Test = 0.2  =  "+NB_Accuaricy(totaled_set_10,lambda_value,train_value_3,test_value_3));

        System.out.println("\n//////////////////////////////////////////////////////\n");
        System.out.println("Denoise Factor 15 Accuracy Parameters Train = 0.6 , Test = 0.4  =  "+NB_Accuaricy(totaled_set_15,lambda_value,train_value_1,test_value_1));
        System.out.println("Denoise Factor 15 Accuracy Parameters Train = 0.7 , Test = 0.3  =  "+NB_Accuaricy(totaled_set_15,lambda_value,train_value_2,test_value_2));
        System.out.println("Denoise Factor 15 Accuracy Parameters Train = 0.8 , Test = 0.2  =  "+NB_Accuaricy(totaled_set_15,lambda_value,train_value_3,test_value_3));
    }

    static JavaRDD<LabeledPoint> unionized_label (String alongside,String building, String road, String vegetation, String water, JavaSparkContext sc, String identfy){
        System.out.println("\n"+identfy+"Labeling Phase Starts");
        JavaRDD<LabeledPoint> alongside_labeled = labelingdata(alongside, 1.0, sc);
        JavaRDD<LabeledPoint> building_labeled = labelingdata(building, 2.0, sc);
        JavaRDD<LabeledPoint> road_labeled = labelingdata(road, 3.0, sc);
        JavaRDD<LabeledPoint> vegetation_labeled = labelingdata(vegetation, 4.0, sc);
        JavaRDD<LabeledPoint> water_labeled = labelingdata(water, 5.0, sc);
        System.out.println(identfy+"Labeling Phase Ends\n");
        //This section unionize each labeled set to train them
        JavaRDD<LabeledPoint> collected_labels = alongside_labeled
                .union(building_labeled)
                .union(road_labeled)
                .union(vegetation_labeled)
                .union(water_labeled);
        return collected_labels;
    }

    static JavaRDD<LabeledPoint> labelingdata(String datapath, double label, JavaSparkContext javasparkcontext) {
        JavaRDD<String> data = javasparkcontext.textFile(datapath);
        //in here we create sub function to split by line and label it.
        JavaRDD<LabeledPoint> labeleddata = data.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String line) {
                String featureString[] = line.trim().split(",");
                double[] Vectors_Double = new double[featureString.length];
                int i = 0;
                for (String s : featureString) {
                    if (s.trim().equals(""))
                        continue;
                    Vectors_Double[i++] = Double.parseDouble(s.trim());
                }
                return new LabeledPoint(label, Vectors.dense(Vectors_Double));
            }
        });
        System.out.println("Labeled Data Count and also Dataset count is ="+labeleddata.count());
        //MLUtils.saveAsLibSVMFile(labeleddata.rdd(), "C:\\Users\\ULGEN\\Documents\\IdeaWorkspace\\Sar_Classify\\Sar_New_vector");
        //System.out.println("labeled data'nın veri tipi ="+labeleddata.getClass().getName());
        return labeleddata;
    }

    static double NB_Accuaricy (JavaRDD<LabeledPoint> datalabel, double lamda,double train_value, double test_value) {
        JavaRDD<LabeledPoint>[] tmp = datalabel.randomSplit(new double[]{train_value, test_value}, 11L);
        JavaRDD<LabeledPoint> training = tmp[0]; // training set
        JavaRDD<LabeledPoint> test = tmp[1]; // test set

        NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
        JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        return 100 * metrics.accuracy();
    }
}
