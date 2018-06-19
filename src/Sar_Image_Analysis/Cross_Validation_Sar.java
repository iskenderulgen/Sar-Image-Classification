package Sar_Image_Analysis;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.classification.SVMWithSGD;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;


public class Cross_Validation_Sar {
    private static double total = 0;
    private static int index = 1;
    private static int[] numbers = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}; // fn stands for fold_number

    public static void NB_CV_Accuaricy(JavaRDD<LabeledPoint> alongside_labeled, JavaRDD<LabeledPoint> building_labeled,
                                       JavaRDD<LabeledPoint> road_labeled, JavaRDD<LabeledPoint> vegetation_labeled,
                                       JavaRDD<LabeledPoint> water_labeled, JavaSparkContext jsc, String identifier, String Cross_identifier) {

        JavaRDD<LabeledPoint> nb_sonuc_along[] = splitting_10(alongside_labeled, jsc);
        JavaRDD<LabeledPoint> nb_sonuc_build[] = splitting_10(building_labeled, jsc);
        JavaRDD<LabeledPoint> nb_sonuc_road[] = splitting_10(road_labeled, jsc);
        JavaRDD<LabeledPoint> nb_sonuc_vege[] = splitting_10(vegetation_labeled, jsc);
        JavaRDD<LabeledPoint> nb_sonuc_water[] = splitting_10(water_labeled, jsc);

        System.out.println("////////////////////////////////////////////////////////////////////////////////////////////\n\n");
        int [] fn  = numbers.clone();
        for (int i = 0; i < 10; i++) {
            //System.out.println(Arrays.toString(fn));
            Evaluation_Section(nb_sonuc_along, nb_sonuc_build, nb_sonuc_road, nb_sonuc_vege, nb_sonuc_water, fn[0], fn[1], fn[2], fn[3], fn[4], fn[5], fn[6], fn[7], fn[8], fn[9], index, identifier, Cross_identifier);
            if (i == 9) break;
            swap(fn, 0, i + 1);
            index++;
        }
        System.out.println("K = 10 Fold " + identifier + " Total Analysis Result is = " + (total / 10));
        total = 0;
        index=1;
        System.out.println("////////////////////////////////////////////////////////////////////////////////////////////\n\n");
    }

    public static void SVM_CV_Accuracy(JavaRDD<LabeledPoint> alongside_pos, JavaRDD<LabeledPoint> alongside_neg,
                                       JavaRDD<LabeledPoint> building_pos, JavaRDD<LabeledPoint> building_neg,
                                       JavaRDD<LabeledPoint> road_pos, JavaRDD<LabeledPoint> road_neg,
                                       JavaRDD<LabeledPoint> vegetation_pos, JavaRDD<LabeledPoint> vegetation_neg,
                                       JavaRDD<LabeledPoint> water_pos, JavaRDD<LabeledPoint> water_neg,
                                       JavaSparkContext jsc, String identifier,String Cross_identifier) {


        JavaRDD<LabeledPoint> svm_along_pos[] = splitting_10(alongside_pos, jsc);
        JavaRDD<LabeledPoint> svm_along_neg[] = splitting_10(alongside_neg, jsc);
        JavaRDD<LabeledPoint> svm_build_pos[] = splitting_10(building_pos, jsc);
        JavaRDD<LabeledPoint> svm_build_neg[] = splitting_10(building_neg, jsc);
        JavaRDD<LabeledPoint> svm_road_pos[] = splitting_10(road_pos, jsc);
        JavaRDD<LabeledPoint> svm_road_neg[] = splitting_10(road_neg, jsc);
        JavaRDD<LabeledPoint> svm_vege_pos[] = splitting_10(vegetation_pos, jsc);
        JavaRDD<LabeledPoint> svm_vege_neg[] = splitting_10(vegetation_neg, jsc);
        JavaRDD<LabeledPoint> svm_water_pos[] = splitting_10(water_pos, jsc);
        JavaRDD<LabeledPoint> svm_water_neg[] = splitting_10(water_neg, jsc);


        Evaluation_Section(svm_along_pos, svm_build_neg, svm_road_neg, svm_vege_neg, svm_water_neg, 0,1,2,3,4,5,6,7,8,9, index, identifier, Cross_identifier);
        Evaluation_Section(svm_along_pos, svm_build_neg, svm_road_neg, svm_vege_neg, svm_water_neg, 1,0,2,3,4,5,6,7,8,9, index, identifier, Cross_identifier);
        Evaluation_Section(svm_along_pos, svm_build_neg, svm_road_neg, svm_vege_neg, svm_water_neg, 2,1,0,3,4,5,6,7,8,9, index, identifier, Cross_identifier);
        Evaluation_Section(svm_along_pos, svm_build_neg, svm_road_neg, svm_vege_neg, svm_water_neg, 3,1,2,0,4,5,6,7,8,9, index, identifier, Cross_identifier);
        Evaluation_Section(svm_along_pos, svm_build_neg, svm_road_neg, svm_vege_neg, svm_water_neg, 4,1,2,3,0,5,6,7,8,9, index, identifier, Cross_identifier);
        Evaluation_Section(svm_along_pos, svm_build_neg, svm_road_neg, svm_vege_neg, svm_water_neg, 5,1,2,3,4,0,6,7,8,9, index, identifier, Cross_identifier);
        Evaluation_Section(svm_along_pos, svm_build_neg, svm_road_neg, svm_vege_neg, svm_water_neg, 6,1,2,3,4,5,0,7,8,9, index, identifier, Cross_identifier);
        Evaluation_Section(svm_along_pos, svm_build_neg, svm_road_neg, svm_vege_neg, svm_water_neg, 7,1,2,3,4,5,6,0,8,9, index, identifier, Cross_identifier);
        Evaluation_Section(svm_along_pos, svm_build_neg, svm_road_neg, svm_vege_neg, svm_water_neg, 8,1,2,3,4,5,6,7,0,9, index, identifier, Cross_identifier);
        Evaluation_Section(svm_along_pos, svm_build_neg, svm_road_neg, svm_vege_neg, svm_water_neg, 9,1,2,3,4,5,6,7,8,0, index, identifier, Cross_identifier);
        System.out.println("K = 10 Fold " + identifier + " Total Analysis Result is = " + (total / 10));



        System.out.println("/////////////////////////////////    Alongside   ////////////////////////////////////\n\n");
        int[] fn_a = numbers.clone(); // fn stands for fold_number
        for (int i = 0; i < 10; i++) {
            Evaluation_Section(svm_along_pos, svm_build_neg, svm_road_neg, svm_vege_neg, svm_water_neg,
                    fn_a[0], fn_a[1], fn_a[2], fn_a[3], fn_a[4], fn_a[5], fn_a[6], fn_a[7], fn_a[8],
                    fn_a[9], index, identifier, Cross_identifier);
            if (i == 9) break;
            swap(fn_a, 0, i + 1);
            index++;
        }
        System.out.println("K = 10 Fold " + identifier + " Alongside Total Analysis Result is = " + (total / 10));
        total = 0;
        index=1;
        System.out.println("/////////////////////////////////    Alongside   ////////////////////////////////////\n\n");



        /*System.out.println("/////////////////////////////////    building   /////////////////////////////////////\n\n");
        int[] fn_b = numbers.clone(); // fn stands for fold_number
        for (int i = 0; i < 10; i++) {
            Evaluation_Section(svm_along_neg, svm_build_pos, svm_road_neg, svm_vege_neg, svm_water_neg,
                    fn_b[0], fn_b[1], fn_b[2], fn_b[3], fn_b[4], fn_b[5], fn_b[6], fn_b[7], fn_b[8], fn_b[9],
                    index, identifier, Cross_identifier);
            if (i == 9) break;
            swap(fn_b, 0, i + 1);
            index++;
        }
        System.out.println("K = 10 Fold " + identifier + " Building Total Analysis Result is = " + (total / 10));
        total = 0;
        index=1;
        System.out.println("/////////////////////////////////    building   /////////////////////////////////////\n\n");



        System.out.println("/////////////////////////////////    road   /////////////////////////////////////////\n\n");
        int[] fn_r = numbers.clone(); // fn stands for fold_number
        for (int i = 0; i < 10; i++) {
            Evaluation_Section(svm_along_neg, svm_build_neg, svm_road_pos, svm_vege_neg, svm_water_neg,
                    fn_r[0], fn_r[1], fn_r[2], fn_r[3], fn_r[4], fn_r[5], fn_r[6], fn_r[7], fn_r[8], fn_r[9],
                    index, identifier, Cross_identifier);
            if (i == 9) break;
            swap(fn_r, 0, i + 1);
            index++;
        }
        System.out.println("K = 10 Fold " + identifier + " Total Analysis Result is = " + (total / 10));
        total = 0;
        index=1;
        System.out.println("/////////////////////////////////    road   /////////////////////////////////////////\n\n");



        System.out.println("/////////////////////////////////    vegetation   ///////////////////////////////////\n\n");
        int[] fn_v = numbers.clone(); // fn stands for fold_number
        for (int i = 0; i < 10; i++) {
            Evaluation_Section(svm_along_neg, svm_build_neg, svm_road_neg, svm_vege_pos, svm_water_neg,
                    fn_v[0], fn_v[1], fn_v[2], fn_v[3], fn_v[4], fn_v[5], fn_v[6], fn_v[7], fn_v[8], fn_v[9],
                    index, identifier, Cross_identifier);
            if (i == 9) break;
            swap(fn_v, 0, i + 1);
            index++;
        }
        System.out.println("K = 10 Fold " + identifier + " Total Analysis Result is = " + (total / 10));
        total = 0;
        index=1;
        System.out.println("/////////////////////////////////    vegetation   ///////////////////////////////////\n\n");



        System.out.println("/////////////////////////////////    water   ////////////////////////////////////////\n\n");
        int[] fn_w = numbers.clone(); // fn stands for fold_number
        for (int i = 0; i < 10; i++) {
            Evaluation_Section(svm_along_neg, svm_build_neg, svm_road_neg, svm_vege_neg, svm_water_pos,
                    fn_w[0], fn_w[1], fn_w[2], fn_w[3], fn_w[4], fn_w[5], fn_w[6], fn_w[7], fn_w[8], fn_w[9],
                    index, identifier, Cross_identifier);
            if (i == 9) break;
            swap(fn_w, 0, i + 1);
            index++;
        }
        System.out.println("K = 10 Fold " + identifier + " Total Analysis Result is = " + (total / 10));
        total = 0;
        index=1;
        System.out.println("/////////////////////////////////    water   ////////////////////////////////////////\n\n");
*/


    }


    private static void Evaluation_Section(JavaRDD<LabeledPoint>[] sonuc_along, JavaRDD<LabeledPoint>[] sonuc_build,
                                           JavaRDD<LabeledPoint>[] sonuc_road, JavaRDD<LabeledPoint>[] sonuc_vege,
                                           JavaRDD<LabeledPoint>[] sonuc_water, int a, int b, int c, int d, int e, int f,
                                           int g, int h, int j, int k, int index, String identifier, String Cross_identifier) {


        JavaRDD<LabeledPoint> test_un = sonuc_along[a].union(sonuc_build[a]).union(sonuc_road[a])
                .union(sonuc_vege[a]).union(sonuc_water[a]);


        JavaRDD<LabeledPoint> train_un = sonuc_along[b].union(sonuc_along[c]).union(sonuc_along[d]).union(sonuc_along[e]).union(sonuc_along[f]).union(sonuc_along[g]).union(sonuc_along[h]).union(sonuc_along[j]).union(sonuc_along[k])
                .union(sonuc_build[b]).union(sonuc_build[c]).union(sonuc_build[d]).union(sonuc_build[e]).union(sonuc_build[f]).union(sonuc_build[g]).union(sonuc_build[h]).union(sonuc_build[j]).union(sonuc_build[k])
                .union(sonuc_road[b]).union(sonuc_road[c]).union(sonuc_road[d]).union(sonuc_road[e]).union(sonuc_road[f]).union(sonuc_road[g]).union(sonuc_road[h]).union(sonuc_road[j]).union(sonuc_road[k])
                .union(sonuc_vege[b]).union(sonuc_vege[c]).union(sonuc_vege[d]).union(sonuc_vege[e]).union(sonuc_vege[f]).union(sonuc_vege[g]).union(sonuc_vege[h]).union(sonuc_vege[j]).union(sonuc_vege[k])
                .union(sonuc_water[b]).union(sonuc_water[c]).union(sonuc_water[d]).union(sonuc_water[e]).union(sonuc_water[f]).union(sonuc_water[g]).union(sonuc_water[h]).union(sonuc_water[j]).union(sonuc_water[k]);


        if (Cross_identifier.equals("Naive_Bayes")) {
            NaiveBayesModel model = NaiveBayes.train(train_un.rdd(), 1.0);
            JavaPairRDD<Object, Object> predictionAndLabels = test_un.mapToPair(p ->
                    new Tuple2<>(model.predict(p.features()), p.label()));
            MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
            System.out.println("K = " + (index) + ". " + identifier + " Fold Analysis Result is = " + 100 * metrics.accuracy());
            total += 100 * metrics.accuracy();
        }

        else if (Cross_identifier.equals("SVM")) {
            SVMModel model = SVMWithSGD.train(train_un.rdd(), 5).setThreshold(0.5);
            JavaRDD<Tuple2<Object, Object>> scoreAndLabels = test_un
                    .map(p -> new Tuple2<>(model.predict(p.features()), p.label()));
            BinaryClassificationMetrics metrics = new BinaryClassificationMetrics(JavaRDD.toRDD(scoreAndLabels));
            System.out.println("K = " + (index) + ". " + identifier + " Fold Analysis Result is = " + 100 * metrics.areaUnderROC());
            total += 100 * metrics.areaUnderROC();
        }
    }

    private static JavaRDD<LabeledPoint>[] splitting_10(JavaRDD<LabeledPoint> data, JavaSparkContext jsc) {
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

        JavaRDD<LabeledPoint> sonuc[] = new JavaRDD[]{data1, data2, data3, data4, data5, data6, data7, data8, data9, data10};
        return sonuc;
    }

    private static void swap(int[] a, int i, int j) {
        Object temp = a[i];
        a[i] = a[j];
        a[j] = (int) temp;
    }
}