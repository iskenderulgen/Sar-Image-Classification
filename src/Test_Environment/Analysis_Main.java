package Test_Environment;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.util.LongAccumulator;
import scala.Tuple2;

public class Analysis_Main {
    private static final String Main_Path = "C:\\Users\\ULGEN\\Documents\\Idea_Workspace\\Sar_Classify\\Sar_Datasets\\";
    private static final String alongside_path = Main_Path + "alongside/training/alongside.csv";
    private static final String building_path = Main_Path + "building/training/building.csv";
    private static final String road_path = Main_Path + "road/training/road.csv";
    private static final String vegetation_path = Main_Path + "vegetation/training/vegetation.csv";
    private static final String water_path = Main_Path + "water/training/water.csv";
    private static final String Sar_image_set1K = Main_Path + "Noised_Big_Sets\\Total_Set_1K.csv";
    private static final String Sar_image_set10K = Main_Path + "Noised_Big_Sets\\Total_Set_10K.csv";
    private static final String Sar_image_set100K = Main_Path + "Noised_Big_Sets\\Total_Set_100K.csv";
    private static final String Sar_image_set1000K = Main_Path + "Noised_Big_Sets\\Total_Set_1M.csv";
    private static final String Sar_image_set10M = Main_Path + "Noised_Big_Sets\\Total_Set_10M.csv";
    private static final String Sar_image_set90M = Main_Path + "Noised_Big_Sets\\Total_Sar_Set.csv";

    public static void main(String[] args) {

        SparkConf conf = new SparkConf()
                .setMaster("local[*]")
                .setAppName("Sar_Analysis");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        jsc.setLogLevel("ERROR");
        LongAccumulator accumulator_alongside = jsc.sc().longAccumulator();
        LongAccumulator accumulator_building = jsc.sc().longAccumulator();
        LongAccumulator accumulator_road = jsc.sc().longAccumulator();
        LongAccumulator accumulator_vegetation = jsc.sc().longAccumulator();
        LongAccumulator accumulator_water = jsc.sc().longAccumulator();

        long Labeling_Clock_Starts = System.currentTimeMillis();
        System.out.println("\nLabeling Phase Starts");
        JavaRDD<LabeledPoint> alongside_labeled = labelingdata(alongside_path, 1.0, jsc);
        JavaRDD<LabeledPoint> building_labeled = labelingdata(building_path, 2.0, jsc);
        JavaRDD<LabeledPoint> road_labeled = labelingdata(road_path, 3.0, jsc);
        JavaRDD<LabeledPoint> vegetation_labeled = labelingdata(vegetation_path, 4.0, jsc);
        JavaRDD<LabeledPoint> water_labeled = labelingdata(water_path, 5.0, jsc);
        System.out.println("Labeling Phase Ends\n");
        long Labeling_Clock_Ends = System.currentTimeMillis();
        long labeling_time = Labeling_Clock_Ends - Labeling_Clock_Starts;
        System.out.println("Total taime elapsed for labeling section is =  " + labeling_time + "  Milisec\t" + (labeling_time / 1000) + "  Seconds\t" + labeling_time / 60000 + "  Minutes");

        JavaRDD<LabeledPoint> collected_labels = alongside_labeled
                .union(building_labeled)
                .union(road_labeled)
                .union(vegetation_labeled)
                .union(water_labeled);

        JavaRDD<Vector> Sar_Vectorized1K = vectorizing(jsc, Sar_image_set1K);
        JavaRDD<Vector> Sar_Vectorized10K = vectorizing(jsc, Sar_image_set10K);
        JavaRDD<Vector> Sar_Vectorized100K = vectorizing(jsc, Sar_image_set100K);
        JavaRDD<Vector> Sar_Vectorized1000K = vectorizing(jsc, Sar_image_set1000K);
        JavaRDD<Vector> Sar_Vectorized10M = vectorizing(jsc, Sar_image_set10M);
        JavaRDD<Vector> Sar_Vectorized90M = vectorizing(jsc, Sar_image_set90M);

        long train_start = System.currentTimeMillis();
        NaiveBayesModel NB_Model = NaiveBayes.train(collected_labels.rdd(), 1.0);
        long train_end = System.currentTimeMillis();
        System.out.println("Total time for train phase ="+ (train_end-train_start));
        System.out.println("Naive Bayes Accuracy based on %30-%70 splitting = \t" + NB_Accuaricy(collected_labels, 1.0)+"\n\n");
        CV_Accuaricy(alongside_labeled,building_labeled,road_labeled,vegetation_labeled,water_labeled,jsc,"Noised Set");

        Analysis_Section(Sar_Vectorized1K,NB_Model,accumulator_alongside,accumulator_building,accumulator_road,accumulator_vegetation,accumulator_water,"total Set (1K Row)");
        Analysis_Section(Sar_Vectorized10K,NB_Model,accumulator_alongside,accumulator_building,accumulator_road,accumulator_vegetation,accumulator_water,"total Set (10K Row)");
        Analysis_Section(Sar_Vectorized100K,NB_Model,accumulator_alongside,accumulator_building,accumulator_road,accumulator_vegetation,accumulator_water,"total Set (100K Row)");
        Analysis_Section(Sar_Vectorized1000K,NB_Model,accumulator_alongside,accumulator_building,accumulator_road,accumulator_vegetation,accumulator_water,"total Set (1000K-1M Row)");
        Analysis_Section(Sar_Vectorized10M,NB_Model,accumulator_alongside,accumulator_building,accumulator_road,accumulator_vegetation,accumulator_water,"total Set (10M Row)");
        Analysis_Section(Sar_Vectorized90M,NB_Model,accumulator_alongside,accumulator_building,accumulator_road,accumulator_vegetation,accumulator_water,"total Set (90M Row)");
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
        System.out.println("Labeled Data Count and also Dataset count is ="+labeleddata.count());
        return labeleddata;
    }

    private static JavaRDD<Vector> vectorizing(org.apache.spark.api.java.JavaSparkContext sparkcontext, String rddtovector) {
        JavaRDD<String> Sar_analyze_data = sparkcontext.textFile(rddtovector);
        JavaRDD<Vector> Sar_Vector = Sar_analyze_data.map(new Function<String, Vector>() {
            @Override
            public Vector call(String line) throws Exception {
                String featureString[] = line.trim().split(",");
                double[] v = new double[featureString.length];
                int i = 0;
                for (String s : featureString) {
                    if (s.trim().equals(""))
                        continue;
                    v[i++] = Double.parseDouble(s.trim());
                }
                return  Vectors.dense(v);
            }
        });
        return Sar_Vector;
    }

    private static void Analysis_Section (JavaRDD<Vector> Sar_Dataset, NaiveBayesModel NB_Model, LongAccumulator accumulator_alongside,
                                  LongAccumulator accumulator_building, LongAccumulator accumulator_road, LongAccumulator accumulator_vegetation,
                                  LongAccumulator accumulator_water , String Identifier ){
        System.out.println(Identifier + " Dataset Naive Bayes Result Section Beginning");
        long Total_Data_Analyzed = Sar_Dataset.count();
        System.out.println(" Total Value of the data to be analyzed = \t"+Total_Data_Analyzed);
        long Analysis_Clock_Starts = System.currentTimeMillis();
        Sar_Dataset.foreach((line) ->{
                double returned = NB_Model.predict(line);
                switch (((int) returned)) {
                    case 1:
                        accumulator_alongside.add(1);
                        break;
                    case 2:
                        accumulator_building.add(1);
                        break;
                    case 3:
                        accumulator_road.add(1);
                        break;
                    case 4:
                        accumulator_vegetation.add(1);
                        break;
                    case 5:
                        accumulator_water.add(1);
                        break;
                }
        });
        long Analysis_Clock_Ends = System.currentTimeMillis();
        long analysis_time = Analysis_Clock_Ends-Analysis_Clock_Starts;
        System.out.println("Total taime elapsed for analysis section is =  "+analysis_time+"  Milisec\t"+(analysis_time/1000)+"  Seconds\t"+analysis_time/60000+"  Minutes");
        System.out.println(Identifier+" Total lines in Sar Set = " + Total_Data_Analyzed);
        System.out.println(Identifier+" Alongside counter   = " + accumulator_alongside.value() +"\t\t"+
                Identifier+" Alongside percent= " + (100 *  ((double) accumulator_alongside.value() / Total_Data_Analyzed)));
        System.out.println(Identifier+" building counter    = " + accumulator_building.value() +"\t\t"+
                Identifier+" building percent= " + (100 * ((double) accumulator_building.value() / Total_Data_Analyzed)));
        System.out.println(Identifier+" road counter        = " + accumulator_road.value() +"\t\t"+
                Identifier+" road percent= " + (100 * ((double) accumulator_road.value() / Total_Data_Analyzed)));
        System.out.println(Identifier+" vegetation counter  = " + accumulator_vegetation.value() +"\t\t"+
                Identifier+" vegetation percent= " + (100 * ((double) accumulator_vegetation.value() / Total_Data_Analyzed)));
        System.out.println(Identifier+" water counter       = " + accumulator_water.value() +"\t\t"+
                Identifier+" water percent= " + (100 * ((double) accumulator_water.value() / Total_Data_Analyzed)));
        System.out.println(Identifier+" Dataset Naive Bayes Result Section Ending");
        System.out.println(Identifier+" Sar Image Analysis Ends Here\n");
        long Analysed_Vectors = accumulator_alongside.value()+ accumulator_building.value()+ accumulator_road.value()+ accumulator_vegetation.value()+ accumulator_water.value();

        if(Total_Data_Analyzed == Analysed_Vectors){
            System.out.println("Total Dataset Count And Analysed Vector's Counts Are Equal. No Failure\n\n");
        }
        else
            System.out.println("Total Dataset Count And Analysed Vector's Counts Are Not Equal."+
                    "Difference is = "+ (Total_Data_Analyzed -Analysed_Vectors)+
                    "\nIf Difference is not big this may caused by Unclassified Data... "+
                    "\nif Difference is bigger this may casued by input failure.\n\n");
        System.out.println("////////////////////////////////////////////////////////////////////////////////////////////\n\n");
        accumulator_alongside.reset();
        accumulator_building.reset();
        accumulator_road.reset();
        accumulator_vegetation.reset();
        accumulator_water.reset();
    }


    private static double NB_Accuaricy (JavaRDD<LabeledPoint> datalabel, double lamda) {
        JavaRDD<LabeledPoint>[] tmp = datalabel.randomSplit(new double[]{0.7, 0.3}, 11L);
        JavaRDD<LabeledPoint> training = tmp[0]; // training set
        JavaRDD<LabeledPoint> test = tmp[1]; // test set
        NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
        JavaPairRDD<Object, Object> predictionAndLabels = test.mapToPair(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        return 100 * metrics.accuracy();
    }

    static void CV_Accuaricy(JavaRDD<LabeledPoint> alongside_labeled, JavaRDD<LabeledPoint> building_labeled,
                             JavaRDD<LabeledPoint> road_labeled, JavaRDD<LabeledPoint> vegetation_labeled,
                             JavaRDD<LabeledPoint> water_labeled, JavaSparkContext jsc, String identifier) {

        JavaRDD<LabeledPoint> sonuc_along[] = splitting_10(alongside_labeled, jsc);
        JavaRDD<LabeledPoint> sonuc_build[] = splitting_10(building_labeled, jsc);
        JavaRDD<LabeledPoint> sonuc_road[] = splitting_10(road_labeled, jsc);
        JavaRDD<LabeledPoint> sonuc_vege[] = splitting_10(vegetation_labeled, jsc);
        JavaRDD<LabeledPoint> sonuc_water[] = splitting_10(water_labeled, jsc);

        double sonuc_1 = (int) NB_Cross_Accuaricy(sonuc_along,sonuc_build,sonuc_road,sonuc_vege,sonuc_water, 0,1,2,3,4,5,6,7,8,9);
        double sonuc_2 = (int) NB_Cross_Accuaricy(sonuc_along,sonuc_build,sonuc_road,sonuc_vege,sonuc_water, 1,0,2,3,4,5,6,7,8,9);
        double sonuc_3 = (int) NB_Cross_Accuaricy(sonuc_along,sonuc_build,sonuc_road,sonuc_vege,sonuc_water, 2,1,0,3,4,5,6,7,8,9);
        double sonuc_4 = (int) NB_Cross_Accuaricy(sonuc_along,sonuc_build,sonuc_road,sonuc_vege,sonuc_water, 3,1,2,0,4,5,6,7,8,9);
        double sonuc_5 = (int) NB_Cross_Accuaricy(sonuc_along,sonuc_build,sonuc_road,sonuc_vege,sonuc_water, 4,1,2,3,0,5,6,7,8,9);
        double sonuc_6 = (int) NB_Cross_Accuaricy(sonuc_along,sonuc_build,sonuc_road,sonuc_vege,sonuc_water, 5,1,2,3,4,0,6,7,8,9);
        double sonuc_7 = (int) NB_Cross_Accuaricy(sonuc_along,sonuc_build,sonuc_road,sonuc_vege,sonuc_water, 6,1,2,3,4,5,0,7,8,9);
        double sonuc_8 = (int) NB_Cross_Accuaricy(sonuc_along,sonuc_build,sonuc_road,sonuc_vege,sonuc_water, 7,1,2,3,4,5,6,0,8,9);
        double sonuc_9 = (int) NB_Cross_Accuaricy(sonuc_along,sonuc_build,sonuc_road,sonuc_vege,sonuc_water, 8,1,2,3,4,5,6,7,0,9);
        double sonuc_10 = (int) NB_Cross_Accuaricy(sonuc_along,sonuc_build,sonuc_road,sonuc_vege,sonuc_water,9,1,2,3,4,5,6,7,8,0);

        System.out.println("////////////////////////////////////////////////////////////////////////////////////////////\n\n");
        System.out.println("K = 1. "+identifier+" Fold Analysis Result is = "+sonuc_1);
        System.out.println("K = 2. "+identifier+" Fold Analysis Result is = "+sonuc_2);
        System.out.println("K = 3. "+identifier+" Fold Analysis Result is = "+sonuc_3);
        System.out.println("K = 4. "+identifier+" Fold Analysis Result is = "+sonuc_4);
        System.out.println("K = 5. "+identifier+" Fold Analysis Result is = "+sonuc_5);
        System.out.println("K = 6. "+identifier+" Fold Analysis Result is = "+sonuc_6);
        System.out.println("K = 7. "+identifier+" Fold Analysis Result is = "+sonuc_7);
        System.out.println("K = 8. "+identifier+" Fold Analysis Result is = "+sonuc_8);
        System.out.println("K = 9. "+identifier+" Fold Analysis Result is = "+sonuc_9);
        System.out.println("K = 10. "+identifier+" Fold Analysis Result is = "+sonuc_10);
        System.out.println("K = 10 Fold "+identifier+ " Total Analysis Result is = "+(sonuc_1+sonuc_2+sonuc_3+sonuc_4+
                sonuc_5+sonuc_6+sonuc_7+sonuc_8+sonuc_9+sonuc_10)/10);
        System.out.println("////////////////////////////////////////////////////////////////////////////////////////////\n\n");
    }

    private static  JavaRDD<LabeledPoint>[] splitting_10(JavaRDD<LabeledPoint> data, JavaSparkContext jsc ){
        long sayi = data.count();
        int i = (int) sayi/10;
        JavaRDD<LabeledPoint> data1 = jsc.parallelize(data.take(i));
        JavaRDD<LabeledPoint> data2 = jsc.parallelize(data.take(2*i)).subtract(jsc.parallelize(data.take(i)));
        JavaRDD<LabeledPoint> data3 = jsc.parallelize(data.take(3*i)).subtract(jsc.parallelize(data.take(2*i)));
        JavaRDD<LabeledPoint> data4 = jsc.parallelize(data.take(4*i)).subtract(jsc.parallelize(data.take(3*i)));
        JavaRDD<LabeledPoint> data5 = jsc.parallelize(data.take(5*i)).subtract(jsc.parallelize(data.take(4*i)));
        JavaRDD<LabeledPoint> data6 = jsc.parallelize(data.take(6*i)).subtract(jsc.parallelize(data.take(5*i)));
        JavaRDD<LabeledPoint> data7 = jsc.parallelize(data.take(7*i)).subtract(jsc.parallelize(data.take(6*i)));
        JavaRDD<LabeledPoint> data8 = jsc.parallelize(data.take(8*i)).subtract(jsc.parallelize(data.take(7*i)));
        JavaRDD<LabeledPoint> data9 = jsc.parallelize(data.take(9*i)).subtract(jsc.parallelize(data.take(8*i)));
        JavaRDD<LabeledPoint> data10 = data.subtract(jsc.parallelize(data.take(9*i)));

        JavaRDD<LabeledPoint> sonuc[] = new JavaRDD[] {data1,data2,data3,data4,data5,data6,data7,data8,data9,data10};
        return  sonuc;
    }

    private static double NB_Cross_Accuaricy(JavaRDD<LabeledPoint>[] sonuc_along,JavaRDD<LabeledPoint>[] sonuc_build,
                                             JavaRDD<LabeledPoint>[] sonuc_road,JavaRDD<LabeledPoint>[] sonuc_vege,
                                             JavaRDD<LabeledPoint>[] sonuc_water, int a, int b, int c,int d,int e,
                                             int f,int g, int h, int j,int k )
    {
        JavaRDD<LabeledPoint> test_un =  sonuc_along[a].union(sonuc_build[a]).union(sonuc_road[a]).union(sonuc_vege[a]).union(sonuc_water[a]);

        JavaRDD<LabeledPoint> train_un = sonuc_along[b].union(sonuc_along[c]).union(sonuc_along[d]).union(sonuc_along[e]).union(sonuc_along[f]).union(sonuc_along[g]).union(sonuc_along[h]).union(sonuc_along[j]).union(sonuc_along[k])
                .union(sonuc_build[b]).union(sonuc_build[c]).union(sonuc_build[d]).union(sonuc_build[e]).union(sonuc_build[f]).union(sonuc_build[g]).union(sonuc_build[h]).union(sonuc_build[j]).union(sonuc_build[k])
                .union(sonuc_road[b]).union(sonuc_road[c]).union(sonuc_road[d]).union(sonuc_road[e]).union(sonuc_road[f]).union(sonuc_road[g]).union(sonuc_road[h]).union(sonuc_road[j]).union(sonuc_road[k])
                .union(sonuc_vege[b]).union(sonuc_vege[c]).union(sonuc_vege[d]).union(sonuc_vege[e]).union(sonuc_vege[f]).union(sonuc_vege[g]).union(sonuc_vege[h]).union(sonuc_vege[j]).union(sonuc_vege[k])
                .union(sonuc_water[b]).union(sonuc_water[c]).union(sonuc_water[d]).union(sonuc_water[e]).union(sonuc_water[f]).union(sonuc_water[g]).union(sonuc_water[h]).union(sonuc_water[j]).union(sonuc_water[k]);

        NaiveBayesModel model = NaiveBayes.train(train_un.rdd(), 1.0);
        JavaPairRDD<Object, Object> predictionAndLabels = test_un.mapToPair(p ->
                new Tuple2<>(model.predict(p.features()), p.label()));
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
        return 100 * metrics.accuracy();
    }
}

// MLUtils.saveAsLibSVMFile(testdata,"C:\\Users\\ULGEN\\Documents\\Idea_Workspace\\Sar_Classify\\training_set\\class2");
//MLUtils.saveAsLibSVMFile(labeleddata.rdd(), "C:\\Users\\ULGEN\\Documents\\IdeaWorkspace\\Sar_Classify\\Sar_New_vector");
//System.out.println("labeled data'nın veri tipi ="+labeleddata.getClass().getName());