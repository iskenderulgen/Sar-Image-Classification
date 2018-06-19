package Sar_Image_Analysis;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;

public class Pre_Process {

        //It takes the data and labels.
        public static JavaRDD<LabeledPoint> labelingdata(String datapath, double label, JavaSparkContext javasparkcontext) {
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
            return labeleddata;
        }

        static JavaRDD<Vector> vectorizing(JavaSparkContext jsc,String Datapath) {
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
                    return  Vectors.dense(v);
                }
            });
            return Sar_Vector;
        }
}



