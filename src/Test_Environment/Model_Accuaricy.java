package Test_Environment;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.LongAccumulator;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;

public class Model_Accuaricy {

    /* This Section Contains The 5x5 Feature Vector Files To Test Minimal Environment. It Also Splittet %60-%40
       then Analyzed to measure Evaluation Metrics. Also each 5 set trained and predicted over total 25 set to measure
       real accuaricy. Result will be given when code executed */
    private static final String Main_Path = "C:\\Users\\ULGEN\\Documents\\Idea_Workspace\\Sar_Classify\\Sar_Datasets\\";
    private static final String Results_Path = "C:\\Users\\ULGEN\\Desktop\\Total_Results\\";

    private static final String alongside_path = Main_Path + "alongside\\training\\alongside.csv";
    private static final String building_path = Main_Path + "building\\training\\building.csv";
    private static final String road_path = Main_Path + "road\\training\\road.csv";
    private static final String vegetation_path = Main_Path + "vegetation\\training\\vegetation.csv";
    private static final String water_path = Main_Path + "water\\training\\water.csv";

    private static final String alongside_path_5 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised5\\alongside_denoised.csv";
    private static final String building_path_5 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised5\\building_denoised.csv";
    private static final String road_path_5 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised5\\road_denoised.csv";
    private static final String vegetation_path_5 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised5\\vegetation_denoised.csv";
    private static final String water_path_5 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised5\\water_denoised.csv";

    private static final String alongside_path_10 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised10\\alongside_denoised.csv";
    private static final String building_path_10 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised10\\building_denoised.csv";
    private static final String road_path_10 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised10\\road_denoised.csv";
    private static final String vegetation_path_10 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised10\\vegetation_denoised.csv";
    private static final String water_path_10 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised10\\water_denoised.csv";


    private static final String alongside_path_15 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised15\\alongside_denoised.csv";
    private static final String building_path_15 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised15\\building_denoised.csv";
    private static final String road_path_15 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised15\\road_denoised.csv";
    private static final String vegetation_path_15 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised15\\vegetation_denoised.csv";
    private static final String water_path_15 = Main_Path + "Sar_Different_Denoise_Parameters\\features_denoised15\\water_denoised.csv";
    private static int[] numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // fn stands for fold_number
    private static int[] svm_local_numbers = {0, 1, 2, 3, 4};

    public static void main(String[] args) throws FileNotFoundException {
        SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("Sar_Analysis");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession
                .builder()
                .appName("Java Spark SQL basic example")
                .config(conf)
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");

        long Total_Analysis_Time_Start = System.currentTimeMillis();


        String[] noised_paths = {alongside_path, building_path, road_path, vegetation_path, water_path};
        String[] denoised_5_paths = {alongside_path_5, building_path_5, road_path_5, vegetation_path_5, water_path_5};
        String[] denoised_10_paths = {alongside_path_10, building_path_10, road_path_10, vegetation_path_10, water_path_10};
        String[] denoised_15_paths = {alongside_path_15, building_path_15, road_path_15, vegetation_path_15, water_path_15};

        double train_value_1 = 0.6; double train_value_2 = 0.7; double train_value_3 = 0.8;
        double test_value_1 = 0.4; double test_value_2 = 0.3; double test_value_3 = 0.2;

        double[] train_values = {train_value_1,train_value_2,train_value_3};
        double[] test_values = {test_value_1,test_value_2,test_value_3};

        double[] Multiclass_labels = {1, 2, 3, 4, 5};
        double[] SVM_labels   = {1,0,0,0,0};
        double[] SVM_labels_2 = {0,1,0,0,0};
        double[] SVM_labels_3 = {0,0,1,0,0};
        double[] SVM_labels_4 = {0,0,0,1,0};
        double[] SVM_labels_5 = {0,0,0,0,1};
        String[] Model_Identifier = {"Alongside", "Building", "Road", "Vegetation", "Water"};
        String[] noise_identifier = {"Noised Data","Denoised Factor 5 Data","Denoised Factor 10 Data","Denoised Factor 15 Data"};
        PrintStream outputs = new PrintStream(new FileOutputStream(new File(Results_Path+"All_Accuracy_Results.txt")));
        System.setOut(outputs);

        System.out.println("\n////////////////////////////    Naive Bayes Starts Here ///////////////////////////////\n");

        NB_Accuaricy(noised_paths, jsc, train_values, test_values,"Noised Data");
        NB_Accuaricy(denoised_5_paths, jsc, train_values, test_values,"Denoise Factor 5");
        NB_Accuaricy(denoised_10_paths, jsc, train_values, test_values,"Denoise Factor 10");
        NB_Accuaricy(denoised_15_paths, jsc, train_values, test_values,"Denoise Factor 15");

        Evaluation_Section(noised_paths, Multiclass_labels, "Naive Bayes Cross Validation Noised Data", "Naive_Bayes", jsc);
        Evaluation_Section(denoised_5_paths, Multiclass_labels, "Naive Bayes Cross Validation Denoised Factor 5 Data", "Naive_Bayes", jsc);
        Evaluation_Section(denoised_10_paths, Multiclass_labels, "Naive Bayes Cross Validation Denoised Factor 10 Data", "Naive_Bayes", jsc);
        Evaluation_Section(denoised_15_paths, Multiclass_labels, "Naive Bayes Cross Validation Denoised Factor 15 Data", "Naive_Bayes", jsc);

        System.out.println("\n////////////////////////////    Naive Bayes Ends Here //////////////////////////////\n");


        System.out.println("\n////////////////////////////    SVM Starts Here       //////////////////////////////\n");

        SVM_Accuracy(noised_paths,SVM_labels,jsc,train_values,test_values,"SVM Noised Data",Model_Identifier);
        SVM_Accuracy(denoised_5_paths,SVM_labels,jsc,train_values,test_values,"SVM Denoised Factor 5 Data",Model_Identifier);
        SVM_Accuracy(denoised_10_paths,SVM_labels,jsc,train_values,test_values,"SVM Denoised Factor 10 Data",Model_Identifier);
        SVM_Accuracy(denoised_15_paths,SVM_labels,jsc,train_values,test_values,"SVM Denoised Factor 15 Data",Model_Identifier);

        String[][] total_paths = {noised_paths,denoised_5_paths,denoised_10_paths,denoised_15_paths};
        double[][] SVM_Total_labels = {SVM_labels,SVM_labels_2,SVM_labels_3,SVM_labels_4,SVM_labels_5};
        for(int i =0;i<4;i++){
            for (int j =0;j<5;j++){
                Evaluation_Section(total_paths[i], SVM_Total_labels[j], "SVM Cross "+Model_Identifier[j]+" "+noise_identifier[i], "SVM", jsc);
            }
        }

        System.out.println("\n////////////////////////////    SVM ENDS Here       //////////////////////////////\n");

        System.out.println("\n////////////////////////////   Decision TRee Starts Here       //////////////////////////////\n");

        DTree_Accuracy(noised_paths,jsc,train_values,test_values,"Noised Data");
        DTree_Accuracy(denoised_5_paths,jsc,train_values,test_values,"Denoised Factor 5");
        DTree_Accuracy(denoised_10_paths,jsc,train_values,test_values,"Denoised Factor 10");
        DTree_Accuracy(denoised_15_paths,jsc,train_values,test_values,"Denoised Factor 15");

        Evaluation_Section(noised_paths, Multiclass_labels, "Decision TRee Cross Validation Noised Data", "Decision_Tree", jsc);
        Evaluation_Section(denoised_5_paths, Multiclass_labels, "Decision TRee Cross Validation Denoised Factor 5 Data", "Decision_Tree", jsc);
        Evaluation_Section(denoised_10_paths, Multiclass_labels, "Decision TRee Cross Validation Denoised Factor 10 Data", "Decision_Tree", jsc);
        Evaluation_Section(denoised_15_paths, Multiclass_labels, "Decision TRee Cross Validation Denoised Factor 15 Data", "Decision_Tree", jsc);

        System.out.println("\n////////////////////////////   Decision TRee Ends Here       //////////////////////////////\n");

        long Total_Analysis_Time_Ends = System.currentTimeMillis();
        long analysis_time = Total_Analysis_Time_Ends - Total_Analysis_Time_Start;
        System.out.println("Total taime elapsed is =  " + analysis_time + "  Milisec\t" + (analysis_time / 1000) + "  Seconds\t" + analysis_time / 60000 + "  Minutes");

    }


    private static JavaRDD<LabeledPoint> labelingdata(String datapath, double label, JavaSparkContext javasparkcontext) {

        JavaRDD<String> data = javasparkcontext.textFile(datapath);
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
        return labeleddata;
    }

    private static void NB_Accuaricy(String[] paths, JavaSparkContext jsc, double[] train_values, double[] test_values, String identifier) {

        JavaRDD<LabeledPoint> collected_labels = labelingdata(paths[0], 1.0, jsc)
                .union(labelingdata(paths[1], 2.0, jsc))
                .union(labelingdata(paths[2], 3.0, jsc))
                .union(labelingdata(paths[3], 4.0, jsc))
                .union(labelingdata(paths[4], 5.0, jsc));

        for(int j =0;j<3;j++) {
            JavaRDD<LabeledPoint>[] splits = collected_labels.randomSplit(new double[]{train_values[j], test_values[j]}, 11L);
            JavaRDD<LabeledPoint> training = splits[0]; // training set
            JavaRDD<LabeledPoint> test = splits[1]; // test set
            NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
            JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
                    new Tuple2<>(model.predict(p.features()), p.label()));
            MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

            System.out.println(identifier + " Niave Bayes Accuracy Parameters Train = " + train_values[j] + "  Test = " + test_values[j] + " = " + 100 * metrics.accuracy());
        }
        System.out.println("\n");
    }

    private static void SVM_Accuracy (String[] paths, double[] labels, JavaSparkContext jsc, double[] train_values, double[] test_values,String identifier,String[] model_identifier){

        int[] fn = svm_local_numbers.clone();
        for (int i=0; i<5;i++) {
            for(int j =0;j<3;j++) {
                JavaRDD<LabeledPoint> collected_labels = labelingdata(paths[0], labels[fn[0]], jsc)
                        .union(labelingdata(paths[1], labels[fn[1]], jsc))
                        .union(labelingdata(paths[2], labels[fn[2]], jsc))
                        .union(labelingdata(paths[3], labels[fn[3]], jsc))
                        .union(labelingdata(paths[4], labels[fn[4]], jsc));

                JavaRDD<LabeledPoint>[] splits = collected_labels.randomSplit(new double[]{train_values[j], test_values[j]}, 11L);
                JavaRDD<LabeledPoint> training = splits[0].cache();
                JavaRDD<LabeledPoint> test = splits[1];
                SVMModel model = (SVMWithSGD.train(training.rdd(), 1,10.0,1.0).setThreshold(0.5)).clearThreshold();
                model.clearThreshold();
                JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test
                        .map(p -> new Tuple2<>(model.predict(p.features()), p.label()));
                BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));

                System.out.println(identifier+"  "+model_identifier[i]+"  SVM Accuracy Parameters Train = " + train_values[j] + "  Test = " + test_values[j] + " = " + 100 * metrics.areaUnderROC());
            }
            System.out.println("\n");
            if (i == 4) swap(fn, 3, 4);
            else swap(fn, 0, i + 1);
        }
    }


    private static void DTree_Accuracy(String[] paths, JavaSparkContext jsc, double[] train_values, double[] test_values, String identifier) {

        JavaRDD<LabeledPoint> collected_labels = labelingdata(paths[0], 1.0, jsc)
                .union(labelingdata(paths[1], 2.0, jsc))
                .union(labelingdata(paths[2], 3.0, jsc))
                .union(labelingdata(paths[3], 4.0, jsc))
                .union(labelingdata(paths[4], 5.0, jsc));

        for(int j =0;j<3;j++) {
            JavaRDD<LabeledPoint>[] splits = collected_labels.randomSplit(new double[]{train_values[j], test_values[j]}, 11L);
            JavaRDD<LabeledPoint> training = splits[0]; // training set
            JavaRDD<LabeledPoint> test = splits[1]; // test set

            Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
            DecisionTreeModel Dtree_model_accuracy = DecisionTree.trainClassifier(training,6,categoricalFeaturesInfo,"gini",5,32);
            JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(p -> new Tuple2<>(Dtree_model_accuracy.predict(p.features()), p.label()));
            double accuracy = predictionAndLabel.filter(pl -> pl._1().equals(pl._2())).count() / (double) test.count();

            System.out.println(identifier + " Decision Tree Accuracy Parameters Train = " + train_values[j] + "  Test = " + test_values[j] + " = " + (100 * accuracy));
        }
        System.out.println("\n");
    }


    private static JavaRDD<LabeledPoint>[] splitting_10(String path, double label, JavaSparkContext jsc) {
        JavaRDD<LabeledPoint> data = labelingdata(path, label, jsc);
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

        JavaRDD<LabeledPoint>[] sonuc = new JavaRDD[]{data1, data2, data3, data4, data5, data6, data7, data8, data9, data10};
        return sonuc;
    }

    private static void Evaluation_Section(String[] paths, double[] labels, String identifier,
                                           String Cross_identifier, JavaSparkContext jsc) {

        JavaRDD<LabeledPoint>[] sonuc_along = splitting_10(paths[0], labels[0], jsc);
        JavaRDD<LabeledPoint>[] sonuc_build = splitting_10(paths[1], labels[1], jsc);
        JavaRDD<LabeledPoint>[] sonuc_road = splitting_10(paths[2], labels[2], jsc);
        JavaRDD<LabeledPoint>[] sonuc_vege = splitting_10(paths[3], labels[3], jsc);
        JavaRDD<LabeledPoint>[] sonuc_water = splitting_10(paths[4], labels[4], jsc);


        long CV_Clock_Starts = System.currentTimeMillis();
        int[] fn = numbers.clone(); // fn stands for fold_number
        LongAccumulator Cross_Validation_Total = jsc.sc().longAccumulator();

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
                    SVMModel model = SVMWithSGD.train(train_un.rdd(), 1,10.0,1.0).setThreshold(0.5);
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

        System.out.println("\nK = 10 Fold " + identifier + " Total Analysis Result is = " + (Cross_Validation_Total.value() / 10));
        long CV_Clock_Ends = System.currentTimeMillis();
        long analysis_time = CV_Clock_Ends - CV_Clock_Starts;
        System.out.println("Total taime elapsed for " + identifier + " Cross Validation section is =  " + analysis_time + "  Milisec\t" + (analysis_time / 1000) + "  Seconds\t" + analysis_time / 60000 + "  Minutes\n\n");
        System.out.println("/////////////////////////////////  " + identifier + "  ////////////////////////////////////////\n\n");
    }

    private static void swap(int[] a, int i, int j) {
        Object temp = a[i];
        a[i] = a[j];
        a[j] = (int) temp;
    }
}