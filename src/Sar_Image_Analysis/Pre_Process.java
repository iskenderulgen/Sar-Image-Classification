package Sar_Image_Analysis;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;


public class Pre_Process {

    //It takes the data and labels.
    public static JavaRDD<LabeledPoint> labeling_data(String datapath, double label, JavaSparkContext javasparkcontext) {
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
                    //System.out.println(Arrays.toString(Vectors_Double));
                }
                return new LabeledPoint(label, Vectors.dense(Vectors_Double));
            }
        });
        return labeleddata;
    }

    static JavaRDD<Vector> vectorised(JavaSparkContext jsc, String Datapath) {
        JavaRDD<String> Sar_analyze_data = jsc.textFile(Datapath);
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
                return Vectors.dense(v);
            }
        });
        return Sar_Vector;
    }

    public static JavaRDD<LabeledPoint>[] splitting_10(JavaRDD<LabeledPoint> data, JavaSparkContext jsc) {
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
}