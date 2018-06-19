package Sar_Image_Analysis;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.classification.SVMModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.util.LongAccumulator;

class Analysis_Section {
    static void NaiveBayes_Analysis(String Identifier,String Sar_Dataset,NaiveBayesModel NB_Model,
                                    LongAccumulator accumulator_alongside,LongAccumulator accumulator_building,
                                    LongAccumulator accumulator_road,LongAccumulator accumulator_vegetation,
                                    LongAccumulator accumulator_water,JavaSparkContext jsc)
    {
        //This section turns integer features to double expressions so that spark can analyze them more correctly.
        JavaRDD<Vector> Sar_Vectorized = Pre_Process.vectorizing(jsc, Sar_Dataset);
        System.out.println(Identifier + " Dataset Naive Bayes Result Section Beginning");
        long Total_Data_Analyzed = Sar_Vectorized.count();
        System.out.println(" Total Value of the data to be analyzed = \t"+Total_Data_Analyzed);
        long Analysis_Clock_Starts = System.currentTimeMillis();
        Sar_Vectorized.foreach((line) ->{
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
        System.out.println(Identifier+" Total lines in Sar Set  =\t" + Total_Data_Analyzed);
        System.out.println(Identifier+" Alongside counter   =\t" + accumulator_alongside.value() +"\t\t"+
                Identifier+" Alongside percent  =\t" + (100 *  ((double) accumulator_alongside.value() / Total_Data_Analyzed)));
        System.out.println(Identifier+" building counter    =\t" + accumulator_building.value() +"\t\t"+
                Identifier+" building percent   =\t" + (100 * ((double) accumulator_building.value() / Total_Data_Analyzed)));
        System.out.println(Identifier+" road counter        =\t" + accumulator_road.value() +"\t\t"+
                Identifier+" road percent       =\t" + (100 * ((double) accumulator_road.value() / Total_Data_Analyzed)));
        System.out.println(Identifier+" vegetation counter  =\t" + accumulator_vegetation.value() +"\t\t"+
                Identifier+" vegetation percent =\t" + (100 * ((double) accumulator_vegetation.value() / Total_Data_Analyzed)));
        System.out.println(Identifier+" water counter       =\t" + accumulator_water.value() +"\t\t"+
                Identifier+" water percent      =\t" + (100 * ((double) accumulator_water.value() / Total_Data_Analyzed)));
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
                    "\nif Difference is bigger, this may casued by input failure. \n\n");
        System.out.println("////////////////////////////////////////////////////////////////////////////////////////////\n\n");
        accumulator_alongside.reset();
        accumulator_building.reset();
        accumulator_road.reset();
        accumulator_vegetation.reset();
        accumulator_water.reset();
    }
    static void SVM_Analysis (SVMModel[] SVM_Model, String Identifier,String Identifier_2,String Sar_Dataset,
                              LongAccumulator accumulator_alongside,LongAccumulator accumulator_building,
                              LongAccumulator accumulator_road,LongAccumulator accumulator_vegetation,
                              LongAccumulator accumulator_water,JavaSparkContext jsc, int Model_index ){

        JavaRDD<Vector> Sar_Vectorized = Pre_Process.vectorizing(jsc, Sar_Dataset);
        System.out.println(Identifier + " Dataset Support Vector Machine Result Section Beginning");
        long Total_Data_Analyzed = Sar_Vectorized.count();
        System.out.println(" Total Value of the data to be analyzed = \t"+Total_Data_Analyzed);
        long Analysis_Clock_Starts = System.currentTimeMillis();

        Sar_Vectorized.foreach((line) ->{
            double returned = SVM_Model[Model_index].predict(line);
            if(Identifier_2.equals("Alongside") &&  returned == 1.0)
                        accumulator_alongside.add(1);

            else if(Identifier_2.equals("building") && returned == 1.0)
                        accumulator_building.add(1);

            else if(Identifier_2.equals("road") && returned == 1.0)
                        accumulator_road.add(1);

            else if(Identifier_2.equals("vegetation") && returned == 1.0)
                        accumulator_vegetation.add(1);

            else if(Identifier_2.equals("water") && returned == 1.0)
                        accumulator_water.add(1);
        });

        long Analysis_Clock_Ends = System.currentTimeMillis();
        long analysis_time = Analysis_Clock_Ends-Analysis_Clock_Starts;
        System.out.println("Total taime elapsed for analysis section is =  "+analysis_time+"  Milisec\t"+(analysis_time/1000)+"  Seconds\t"+analysis_time/60000+"  Minutes");
        System.out.println(Identifier+" Total lines in Sar Set  =\t" + Total_Data_Analyzed);
        System.out.println(Identifier+" Alongside counter   =\t" + accumulator_alongside.value() +"\t\t"+
                Identifier+" Alongside percent  =\t" + (100 *  ((double) accumulator_alongside.value() / Total_Data_Analyzed)));
        System.out.println(Identifier+" building counter    =\t" + accumulator_building.value() +"\t\t"+
                Identifier+" building percent   =\t" + (100 * ((double) accumulator_building.value() / Total_Data_Analyzed)));
        System.out.println(Identifier+" road counter        =\t" + accumulator_road.value() +"\t\t"+
                Identifier+" road percent       =\t" + (100 * ((double) accumulator_road.value() / Total_Data_Analyzed)));
        System.out.println(Identifier+" vegetation counter  =\t" + accumulator_vegetation.value() +"\t\t"+
                Identifier+" vegetation percent =\t" + (100 * ((double) accumulator_vegetation.value() / Total_Data_Analyzed)));
        System.out.println(Identifier+" water counter       =\t" + accumulator_water.value() +"\t\t"+
                Identifier+" water percent      =\t" + (100 * ((double) accumulator_water.value() / Total_Data_Analyzed)));
        System.out.println(Identifier+" Dataset"+Identifier_2+"  Result Section Ending");
        System.out.println(Identifier+" Sar Image Analysis Ends Here\n");
        System.out.println("////////////////////////////////////////////////////////////////////////////////////////////\n\n");
        accumulator_alongside.reset();
        accumulator_building.reset();
        accumulator_road.reset();
        accumulator_vegetation.reset();
        accumulator_water.reset();
    }
}



