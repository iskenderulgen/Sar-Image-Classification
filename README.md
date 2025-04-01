# SAR Image Texture Classification using Apache Spark
## Overview
This project demonstrates fast classification of SAR (Synthetic Aperture Radar) image patches using machine learning models implemented with Apache Spark. SAR images provide valuable information for remote sensing applications but are challenging to process due to their size and complexity. This repository provides a streamlined Jupyter notebook implementation of our original research, which was conducted using a 1+4 (Master-Worker) Spark cluster to process approximately 1 billion rows of data.

The analysis pipeline includes:

- Loading pre-processed SAR image patches
- Training multiple classification models
- Performance evaluation using cross-validation
- Application to full-scale SAR images

By leveraging Apache Spark's distributed computing capabilities, this project achieves significantly faster processing compared to traditional methods.

## Prerequisites

- PySpark library (tested with version 3.5.5)
- Python 3.11
- Java 8+ (required for Spark)
- At least 8GB RAM (16GB recommended for larger datasets)

## Dataset Structure
The project uses pre-processed SAR image patches divided into five classes:

- Alongside (man-made structures along roads/waters)
- Building
- Road
- Vegetation
- Water

Each class has its own CSV file with GLCM features extracted from image patches.

# Implementation Details

## 1. Spark Session Configuration

```
spark = (
    SparkSession.builder.master("local")
    .appName("Sar_Image_Analysis")
    .config("spark.driver.cores", "2")
    .config("spark.driver.memory", "2g")
    .config("spark.executor.cores", "2")
    .config("spark.executor.memory", "2g")
    .config("spark.driver.maxResultSize", "3g")
    .config("spark.executor.instances", "4")
    .getOrCreate()
)
```
The configuration parameters can be adjusted based on your hardware capabilities. For production environments, you might want to increase memory allocation or number of executor instances.

## 2. Data Loading and Preparation

Data from multiple classes is loaded and combined with appropriate labels:

```
alongside = spark.read.csv("data/alongside.csv").withColumn("string_label", F.lit("alongside"))
building = spark.read.csv("data/building.csv").withColumn("string_label", F.lit("building"))
road = spark.read.csv("data/road.csv").withColumn("string_label", F.lit("road"))
vegetation = spark.read.csv("data/vegetation.csv").withColumn("string_label", F.lit("vegetation"))
water = spark.read.csv("data/water.csv").withColumn("string_label", F.lit("water"))

# Combine all datasets
training_dataset = alongside.union(building).union(road).union(vegetation).union(water)
```


## 3. Feature Engineering

Features are converted to appropriate types and assembled into feature vectors:

```
# Convert string features to FloatType
feature_columns = training_dataset.columns[:-1]
training_dataset = training_dataset.select(
    *[F.col(c).cast(FloatType()).alias(c) for c in feature_columns],
    F.col("string_label")
)

# Create feature vectors
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
training_dataset = assembler.transform(training_dataset).select("features", "string_label")

# Convert string labels to numeric indices
string_indexer = StringIndexer(inputCol="string_label", outputCol="label", stringOrderType="alphabetAsc")
training_dataset = string_indexer.fit(training_dataset).transform(training_dataset)
```

## 4. Model Training and Evaluation
The project implements and compares four machine learning models:

- Naive Bayes
- Random Forest
- Decision Tree
- Logistic Regression

Each model is trained and evaluated on a train-test split of the data:

```
# Split data
train, test = training_dataset.randomSplit([0.8, 0.2], seed=42)

# Example: Train a Decision Tree model
model = DecisionTreeClassifier(
    featuresCol="features", labelCol="label", maxDepth=10
).fit(train)

# Evaluate model
predictions = model.transform(test)
accuracy = evaluator.evaluate(predictions)
```

## 5. Cross-Validation
To ensure robust evaluation, 10-fold cross-validation is performed on all models:

```
# Simplified cross-validation code
for model, name in models:
    cv = CrossValidator(
        estimator=Pipeline(stages=[model]),
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=10
    )
    cvModel = cv.fit(training_dataset)
    accuracy = evaluator.evaluate(cvModel.transform(training_dataset))
    print(f"{name}: Accuracy = {accuracy*100:.2f}%")
```

## 6. Full Image Classification
After selecting the best model, it can be applied to classify a full SAR image:

```
# Read full SAR image data
indiana_df = spark.read.parquet("data/indiana.parquet")

# Preprocess the data
indiana_df = indiana_df.select(*[F.col(c).cast(FloatType()).alias(c) for c in indiana_df.columns])
assembler = VectorAssembler(inputCols=indiana_df.columns, outputCol="features")
indiana_df = assembler.transform(indiana_df).select("features")

# Apply the trained model
predictions = model.transform(indiana_df)

# Analyze results by category percentages
result = predictions.groupBy("prediction").count() \
    .withColumn("percentage", F.round(F.col("count") * 100 / predictions.count()))
```
- The indiana data is not included into repository due to having large file size.

# Results
The models typically achieve classification accuracies between 85-95% depending on the specific dataset and parameters.

Classifiers performance comparison:

- Decision Tree: Often provides the best balance of accuracy and computational efficiency
- Random Forest: Typically achieves highest accuracy but requires more computation time
- Naive Bayes: Fastest but may have lower accuracy
- Logistic Regression: Good for understanding feature importance

## Citation
If you use this code in your research, please cite:

```
@article{zcan2020FastTC,
  title={Fast texture classification of denoised SAR image patches using GLCM on Spark},
  author={Caner {\"O}zcan and Okan K. Ersoy and Iskender Ulgen Ogul},
  journal={Turkish J. Electr. Eng. Comput. Sci.},
  year={2020},
  volume={28},
  pages={182-195},
  url={https://api.semanticscholar.org/CorpusID:214245308}
}
```