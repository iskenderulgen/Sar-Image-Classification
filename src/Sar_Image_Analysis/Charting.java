package Sar_Image_Analysis;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.chart.ChartUtilities;


import java.awt.*;
import java.io.File;
import java.io.IOException;

public class Charting {

    public static void NB_Model_Chart(double classic_accuracy, double cross_validation,String path) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        dataset.setValue(classic_accuracy, "%70 - %30 Accuracy", "Accuracy");
        dataset.setValue(cross_validation, "Cross Validation Accuracy", "Accuracy");
        JFreeChart chart = ChartFactory.createBarChart("Naive Bayes Accuracy Results", "Accuracy Types",
                "Results Corresponding as Percent", dataset, PlotOrientation.VERTICAL, true, true, false);
        chart.setBackgroundPaint(Color.white);
        chart.getTitle().setPaint(Color.blue);
        CategoryPlot p = chart.getCategoryPlot();
        p.setRangeGridlinePaint(Color.GREEN);
        ChartFrame frame1 = new ChartFrame("Accuracy Graphs", chart);
        frame1.setVisible(true);
        frame1.setSize(600, 600);

//        ChartUtilities.saveChartAsPNG(new File(path+"NB_Model_Chart.png"), chart, 700, 700);
    }

    public static void SVM_Model_Chart(double[] classic_accuracy, double[] Cross_validation_accuracy,String path) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        dataset.setValue(classic_accuracy[0], "%70 - %30 Accuracy", "Alongside");
        dataset.setValue(Cross_validation_accuracy[0], "Cross Validation", "Alongside");

        dataset.setValue(classic_accuracy[1], "%70 - %30 Accuracy", "Building");
        dataset.setValue(Cross_validation_accuracy[1], "Cross Validation", "Building");

        dataset.setValue(classic_accuracy[2], "%70 - %30 Accuracy", "Road");
        dataset.setValue(Cross_validation_accuracy[2], "Cross Validation", "Road");

        dataset.setValue(classic_accuracy[3], "%70 - %30 Accuracy", "Vegetation");
        dataset.setValue(Cross_validation_accuracy[3], "Cross Validation", "Vegetation");

        dataset.setValue(classic_accuracy[4], "%70 - %30 Accuracy", "Water");
        dataset.setValue(Cross_validation_accuracy[4], "Cross Validation", "Water");

        JFreeChart chart = ChartFactory.createBarChart(" SVM One - Rest Accuracy Results", "Accuracy Types",
                "Results Corresponding as Percent", dataset, PlotOrientation.VERTICAL, true, true, false);
        chart.setBackgroundPaint(Color.white);
        chart.getTitle().setPaint(Color.blue);
        CategoryPlot p = chart.getCategoryPlot();
        p.setRangeGridlinePaint(Color.GREEN);

        ChartFrame frame1 = new ChartFrame("Accuracy Graphs", chart);
        frame1.setVisible(true);
        frame1.setSize(600, 600);
        //ChartUtilities.saveChartAsPNG(new File(path+"SVM_Model_Chart.png"), chart, 700, 700);
    }

    public static void DTree_Model_Chart(double classic_accuracy, double cross_validation,String path) throws IOException {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        dataset.setValue(classic_accuracy, "%70 - %30 Accuracy", "Accuracy");
        dataset.setValue(cross_validation, "Cross Validation Accuracy", "Accuracy");
        JFreeChart chart = ChartFactory.createBarChart("Decision Tree Accuracy Results", "Accuracy Types",
                "Results Corresponding as Percent", dataset, PlotOrientation.VERTICAL, true, true, false);
        chart.setBackgroundPaint(Color.white);
        chart.getTitle().setPaint(Color.blue);
        CategoryPlot p = chart.getCategoryPlot();
        p.setRangeGridlinePaint(Color.GREEN);
        ChartFrame frame1 = new ChartFrame("Accuracy Graphs", chart);
        frame1.setVisible(true);
        frame1.setSize(600, 600);

       // ChartUtilities.saveChartAsPNG(new File(path+"Dtree_Model_Chart.png"), chart, 700, 700);
    }



    static void Result_Chart(String identifier, double[] K1_Results, double[] K10_Results, double[] K100_Results,
                             double[] K1000_Results, double[] M10_results, double[] M90_results,String path) throws IOException {

        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        dataset.setValue(K1_Results[0], "Alongside", "1K Data");
        dataset.setValue(K1_Results[1], "Building", "1K Data");
        dataset.setValue(K1_Results[2], "Road", "1K Data");
        dataset.setValue(K1_Results[3], "Vegetation", "1K Data");
        dataset.setValue(K1_Results[4], "Water", "1K Data");

        dataset.setValue(K10_Results[0], "Alongside", "10K Data");
        dataset.setValue(K10_Results[1], "Building", "10K Data");
        dataset.setValue(K10_Results[2], "Road", "10K Data");
        dataset.setValue(K10_Results[3], "Vegetation", "10K Data");
        dataset.setValue(K10_Results[4], "Water", "10K Data");

        dataset.setValue(K100_Results[0], "Alongside", "100K Data");
        dataset.setValue(K100_Results[1], "Building", "100K Data");
        dataset.setValue(K100_Results[2], "Road", "100K Data");
        dataset.setValue(K100_Results[3], "Vegetation", "100K Data");
        dataset.setValue(K100_Results[4], "Water", "100K Data");

        dataset.setValue(K1000_Results[0], "Alongside", "1M Data");
        dataset.setValue(K1000_Results[1], "Building", "1M Data");
        dataset.setValue(K1000_Results[2], "Road", "1M Data");
        dataset.setValue(K1000_Results[3], "Vegetation", "1M Data");
        dataset.setValue(K1000_Results[4], "Water", "1M Data");

        dataset.setValue(M10_results[0], "Alongside", "10M Data");
        dataset.setValue(M10_results[1], "Building", "10M Data");
        dataset.setValue(M10_results[2], "Road", "10M Data");
        dataset.setValue(M10_results[3], "Vegetation", "10M Data");
        dataset.setValue(M10_results[4], "Water", "10M Data");

        dataset.setValue(M90_results[0], "Alongside", "90M Data");
        dataset.setValue(M90_results[1], "Building", "90M Data");
        dataset.setValue(M90_results[2], "Road", "90M Data");
        dataset.setValue(M90_results[3], "Vegetation", "90M Data");
        dataset.setValue(M90_results[4], "Water", "90M Data");


        JFreeChart chart = ChartFactory.createBarChart(identifier + "  Sar Results", "Total Data Analysed",
                "Results Corresponding as Percent", dataset, PlotOrientation.VERTICAL, true, true, false);
        chart.setBackgroundPaint(Color.white);
        chart.getTitle().setPaint(Color.blue);
        CategoryPlot p = chart.getCategoryPlot();
        p.setRangeGridlinePaint(Color.GREEN);
        ChartFrame frame1 = new ChartFrame("Analysis Graphs", chart);
        frame1.setVisible(true);
        frame1.setSize(600, 600);
/*
        switch (identifier){
            case "Naive_Bayes":{
                ChartUtilities.saveChartAsPNG(new File(path+"Naive_Bayes_Results_Chart.png"), chart, 700, 700);
            }
            case "SVM":{
                ChartUtilities.saveChartAsPNG(new File(path + "SVM_Results_Chart.png"), chart, 700, 700);
            }
            case "Decision_Tree":{
                ChartUtilities.saveChartAsPNG(new File(path + "Decision_Tree_Results.png"), chart, 700, 700);

            }
        }
        */
    }

    static void Analysis_time_chart(long[] NB_time, long[] SVM_time, long[] Dtree_time,String path) throws IOException {

        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        dataset.setValue(NB_time[0]/1000, "Naive Bayes", "1K Data");
        dataset.setValue(SVM_time[0]/1000, "Support Vector Machine", "1K Data");
        dataset.setValue(Dtree_time[0]/1000, "Decision Tree", "1K Data");


        dataset.setValue(NB_time[1]/1000, "Naive Bayes", "10K Data");
        dataset.setValue(SVM_time[1]/1000, "Support Vector Machine", "10K Data");
        dataset.setValue(Dtree_time[1]/1000, "Decision Tree", "10K Data");


        dataset.setValue(NB_time[2]/1000, "Naive Bayes", "100K Data");
        dataset.setValue(SVM_time[2]/1000, "Support Vector Machine", "100K Data");
        dataset.setValue(Dtree_time[2]/1000, "Decision Tree", "100K Data");


        dataset.setValue(NB_time[3]/1000, "Naive Bayes", "1M Data");
        dataset.setValue(SVM_time[3]/1000, "Support Vector Machine", "1M Data");
        dataset.setValue(Dtree_time[3]/1000, "Decision Tree", "1M Data");


        dataset.setValue(NB_time[4]/1000, "Naive Bayes", "10M Data");
        dataset.setValue(SVM_time[4]/1000, "Support Vector Machine", "10M Data");
        dataset.setValue(Dtree_time[4]/1000, "Decision Tree", "10M Data");


        dataset.setValue(NB_time[5]/1000, "Naive Bayes", "90M Data");
        dataset.setValue(SVM_time[5]/1000, "Support Vector Machine", "90M Data");
        dataset.setValue(Dtree_time[5]/1000, "Decision Tree", "90M Data");


        JFreeChart chart = ChartFactory.createBarChart(" Total Analysis Time Comparision", "Time Model Types",
                "Results Corresponding as Seconds", dataset, PlotOrientation.VERTICAL, true, true, false);
        chart.setBackgroundPaint(Color.white);
        chart.getTitle().setPaint(Color.blue);
        CategoryPlot p = chart.getCategoryPlot();
        p.setRangeGridlinePaint(Color.GREEN);
        ChartFrame frame1 = new ChartFrame("Time Graphs", chart);
        frame1.setVisible(true);
        frame1.setSize(600, 600);
       //ChartUtilities.saveChartAsPNG(new File(path+"Time_Results.png"), chart, 700, 700);
    }
}