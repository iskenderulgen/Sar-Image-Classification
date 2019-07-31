# Sar_Analysis
Patched Sar Image Analysis based on Apache Spark Structure. This project based on well-known data analysis framework Apache Spark. Apache Spark is an advanced version of Hadoop which stores the data inside the RAM instead of HDD. This new approach, makes the analysis up 100x faster then the common approaches.
Our focus is to conduct analysis on Remote Sensing images such as SAR. Due to SAR images are too large, we divided the images in to patches and used GLCM (Gray Level Co-Occurrence Matrix method as we convert images to vectors. Then we imported the vectors in to Apache Spark Environment and conducted Naive Bayes, Decision-Trees, Random-Forset Trees and to have some insights about possible deep learning approach, Logistic Regression.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Things you need to install the software 

```
- Apache Spark (Current Version is 2.4.3)
- Java Development Kit (Spark supports Java 8 At the moment)
- winutils.exe (For windows based environment, Spark needs hadoop winutils file)
- Least 16gb ram for better performance
```
### Installing

For java development environment, IntelliJ IDEA would be good choise for starters. If u are student, u can freely get a copy with your school email. After u set up IDE, import Java 8 development environment to your project and then import Spark 2.4.3 bins in to same project folder. After setting all the environments

```
- Create a spark context and spark session objects.
- import dataset to be analyzed.
```
### Running the tests
```
Model accuracy test is based on 60-40 / 70-30 / 80-20 divison and Cross-Validation. Since spark 2.0, cross validation is available under main guide with 'spark-dataset' option.
```
As you run the analysis u can easly follow the process using web-ui provided by spark in localhost. Also, one can limit the processor size and ram amount using spark-context to have variety of results under different environments. 

