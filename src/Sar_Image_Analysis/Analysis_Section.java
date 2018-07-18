package Sar_Image_Analysis;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.util.LongAccumulator;

class Analysis_Section {
    static double along_result = 0, build_result = 0, road_result = 0, vegetation_result = 0, water_result = 0;
    static long analysis_time = 0;
    static void NaiveBayes_Analysis(String Identifier, String Sar_Dataset, NaiveBayesModel NB_Model,
                                    LongAccumulator accumulator_alongside, LongAccumulator accumulator_building,
                                    LongAccumulator accumulator_road, LongAccumulator accumulator_vegetation,
                                    LongAccumulator accumulator_water, JavaSparkContext jsc) {
        //This section turns integer features to double expressions so that spark can analyze them more correctly.
        JavaRDD<Vector> Sar_Vectorized = Pre_Process.vectorizing(jsc, Sar_Dataset);
        System.out.println("///////////   "+Identifier+"   Dataset Naive Bayes Result Section BEGINS  ///////\n\n");
        long Total_Data_Analyzed = Sar_Vectorized.count();
        System.out.println(" Total Value of the data to be analyzed = \t" + Total_Data_Analyzed);
        long Analysis_Clock_Starts = System.currentTimeMillis();
        Sar_Vectorized.foreach((line) -> {
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
        along_result = 100 * ((double) accumulator_alongside.value() / Total_Data_Analyzed);
        build_result = 100 * ((double) accumulator_building.value() / Total_Data_Analyzed);
        road_result = 100 * ((double) accumulator_road.value() / Total_Data_Analyzed);
        vegetation_result = 100 * ((double) accumulator_vegetation.value() / Total_Data_Analyzed);
        water_result = 100 * ((double) accumulator_water.value() / Total_Data_Analyzed);

        long Analysis_Clock_Ends = System.currentTimeMillis();
        analysis_time = Analysis_Clock_Ends - Analysis_Clock_Starts;
        System.out.println("Total taime elapsed for analysis section is =  " + analysis_time + "  Milisec\t" + (analysis_time / 1000) + "  Seconds\t" + analysis_time / 60000 + "  Minutes");
        System.out.println(Identifier + " Total lines in Sar Set  =\t" + Total_Data_Analyzed);
        System.out.println(Identifier + " Alongside counter   =\t" + accumulator_alongside.value() + "\t\t" +
                Identifier + " Alongside percent  =\t" + (100 * ((double) accumulator_alongside.value() / Total_Data_Analyzed)));
        System.out.println(Identifier + " building counter    =\t" + accumulator_building.value() + "\t\t" +
                Identifier + " building percent   =\t" + (100 * ((double) accumulator_building.value() / Total_Data_Analyzed)));
        System.out.println(Identifier + " road counter        =\t" + accumulator_road.value() + "\t\t" +
                Identifier + " road percent       =\t" + (100 * ((double) accumulator_road.value() / Total_Data_Analyzed)));
        System.out.println(Identifier + " vegetation counter  =\t" + accumulator_vegetation.value() + "\t\t" +
                Identifier + " vegetation percent =\t" + (100 * ((double) accumulator_vegetation.value() / Total_Data_Analyzed)));
        System.out.println(Identifier + " water counter       =\t" + accumulator_water.value() + "\t\t" +
                Identifier + " water percent      =\t" + (100 * ((double) accumulator_water.value() / Total_Data_Analyzed)));
        long Analysed_Vectors = accumulator_alongside.value() + accumulator_building.value() + accumulator_road.value() + accumulator_vegetation.value() + accumulator_water.value();

        if (Total_Data_Analyzed == Analysed_Vectors) {
            System.out.println("Total Dataset Count And Analysed Vector's Counts Are Equal. No Failure\n\n");
        } else
            System.out.println("Total Dataset Count And Analysed Vector's Counts Are Not Equal." +
                    "Difference is = " + (Total_Data_Analyzed - Analysed_Vectors) +
                    "\nIf Difference is not big this may caused by Unclassified Data... " +
                    "\nif Difference is bigger, this may casued by input failure. \n\n");
        System.out.println("///////////   "+  Identifier +"  Dataset Naive Bayes Result Section ENDS  ///////\n\n");

        accumulator_alongside.reset();
        accumulator_building.reset();
        accumulator_road.reset();
        accumulator_vegetation.reset();
        accumulator_water.reset();
    }

    static void SVM_Analysis(SVMModel[] SVM_Model, String Identifier, String[] Model_Identifier, String Sar_Dataset,
                             LongAccumulator accumulator_alongside, LongAccumulator accumulator_building,
                             LongAccumulator accumulator_road, LongAccumulator accumulator_vegetation,
                             LongAccumulator accumulator_water, JavaSparkContext jsc) {

        System.out.println("///////////   "+Identifier+"   Dataset Support Vector Machine Result Section BEGINS  ///////\n\n");
        JavaRDD<Vector> Sar_Vectorized = Pre_Process.vectorizing(jsc, Sar_Dataset);
        long Total_Data_Analyzed = Sar_Vectorized.count();
        System.out.println(" Total Value of the data to be analyzed = \t" + Total_Data_Analyzed);
        long Analysis_Clock_Starts = System.currentTimeMillis();
        for (int i = 0; i < 5; i++) {
            String Identifier_2 = Model_Identifier[i];
            SVMModel model = SVM_Model[i];
            Sar_Vectorized.foreach((line) -> {
                double returned = model.predict(line);
                if (Identifier_2.equals("Alongside") && returned == 1.0)
                    accumulator_alongside.add(1);

                else if (Identifier_2.equals("building") && returned == 1.0)
                    accumulator_building.add(1);

                else if (Identifier_2.equals("road") && returned == 1.0)
                    accumulator_road.add(1);

                else if (Identifier_2.equals("vegetation") && returned == 1.0)
                    accumulator_vegetation.add(1);

                else if (Identifier_2.equals("water") && returned == 1.0)
                    accumulator_water.add(1);
            });
        }
        along_result = 100 * ((double) accumulator_alongside.value() / Total_Data_Analyzed);
        build_result = 100 * ((double) accumulator_building.value() / Total_Data_Analyzed);
        road_result = 100 * ((double) accumulator_road.value() / Total_Data_Analyzed);
        vegetation_result = 100 * ((double) accumulator_vegetation.value() / Total_Data_Analyzed);
        water_result = 100 * ((double) accumulator_water.value() / Total_Data_Analyzed);

        long Analysis_Clock_Ends = System.currentTimeMillis();
        analysis_time = Analysis_Clock_Ends - Analysis_Clock_Starts;        analysis_time = Analysis_Clock_Ends - Analysis_Clock_Starts;
        System.out.println("Total taime elapsed for analysis section is =  " + analysis_time + "  Milisec\t" + (analysis_time / 1000) + "  Seconds\t" + analysis_time / 60000 + "  Minutes");
        System.out.println(Identifier + " Total lines in Sar Set  =\t" + Total_Data_Analyzed);
        System.out.println(Identifier + " Alongside counter   =\t" + accumulator_alongside.value() + "\t\t" +
                Identifier + " Alongside percent  =\t" + (100 * ((double) accumulator_alongside.value() / Total_Data_Analyzed)));
        System.out.println(Identifier + " building counter    =\t" + accumulator_building.value() + "\t\t" +
                Identifier + " building percent   =\t" + (100 * ((double) accumulator_building.value() / Total_Data_Analyzed)));
        System.out.println(Identifier + " road counter        =\t" + accumulator_road.value() + "\t\t" +
                Identifier + " road percent       =\t" + (100 * ((double) accumulator_road.value() / Total_Data_Analyzed)));
        System.out.println(Identifier + " vegetation counter  =\t" + accumulator_vegetation.value() + "\t\t" +
                Identifier + " vegetation percent =\t" + (100 * ((double) accumulator_vegetation.value() / Total_Data_Analyzed)));
        System.out.println(Identifier + " water counter       =\t" + accumulator_water.value() + "\t\t" +
                Identifier + " water percent      =\t" + (100 * ((double) accumulator_water.value() / Total_Data_Analyzed))+"\n\n");
        System.out.println("///////////    "+Identifier+"   Dataset Support Vector Machine Result Section ENDS  ///////\n\n");

        accumulator_alongside.reset();
        accumulator_building.reset();
        accumulator_road.reset();
        accumulator_vegetation.reset();
        accumulator_water.reset();
    }

    static void DTree_Analysis (String Identifier, String Sar_Dataset, DecisionTreeModel DTree_Model,
                                LongAccumulator accumulator_alongside, LongAccumulator accumulator_building,
                                LongAccumulator accumulator_road, LongAccumulator accumulator_vegetation,
                                LongAccumulator accumulator_water, JavaSparkContext jsc){

        JavaRDD<Vector> Sar_Vectorized = Pre_Process.vectorizing(jsc, Sar_Dataset);
        System.out.println("///////////   "+Identifier+"   Dataset Decision Tree Result Section BEGINS  ///////\n\n");
        long Total_Data_Analyzed = Sar_Vectorized.count();
        System.out.println(" Total Value of the data to be analyzed = \t" + Total_Data_Analyzed);
        long Analysis_Clock_Starts = System.currentTimeMillis();
        Sar_Vectorized.foreach((line) -> {
            double returned = DTree_Model.predict(line);
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
        along_result = 100 * ((double) accumulator_alongside.value() / Total_Data_Analyzed);
        build_result = 100 * ((double) accumulator_building.value() / Total_Data_Analyzed);
        road_result = 100 * ((double) accumulator_road.value() / Total_Data_Analyzed);
        vegetation_result = 100 * ((double) accumulator_vegetation.value() / Total_Data_Analyzed);
        water_result = 100 * ((double) accumulator_water.value() / Total_Data_Analyzed);

        long Analysis_Clock_Ends = System.currentTimeMillis();
        analysis_time = Analysis_Clock_Ends - Analysis_Clock_Starts;
        System.out.println("Total taime elapsed for analysis section is =  " + analysis_time + "  Milisec\t" + (analysis_time / 1000) + "  Seconds\t" + analysis_time / 60000 + "  Minutes");
        System.out.println(Identifier + " Total lines in Sar Set  =\t" + Total_Data_Analyzed);
        System.out.println(Identifier + " Alongside counter   =\t" + accumulator_alongside.value() + "\t\t" +
                Identifier + " Alongside percent  =\t" + (100 * ((double) accumulator_alongside.value() / Total_Data_Analyzed)));
        System.out.println(Identifier + " building counter    =\t" + accumulator_building.value() + "\t\t" +
                Identifier + " building percent   =\t" + (100 * ((double) accumulator_building.value() / Total_Data_Analyzed)));
        System.out.println(Identifier + " road counter        =\t" + accumulator_road.value() + "\t\t" +
                Identifier + " road percent       =\t" + (100 * ((double) accumulator_road.value() / Total_Data_Analyzed)));
        System.out.println(Identifier + " vegetation counter  =\t" + accumulator_vegetation.value() + "\t\t" +
                Identifier + " vegetation percent =\t" + (100 * ((double) accumulator_vegetation.value() / Total_Data_Analyzed)));
        System.out.println(Identifier + " water counter       =\t" + accumulator_water.value() + "\t\t" +
                Identifier + " water percent      =\t" + (100 * ((double) accumulator_water.value() / Total_Data_Analyzed)));
        System.out.println("///////////   "+  Identifier +"  Dataset Decision Tree Result Section ENDS  ///////\n\n");

        accumulator_alongside.reset();
        accumulator_building.reset();
        accumulator_road.reset();
        accumulator_vegetation.reset();
        accumulator_water.reset();
    }
}