 Real-Time Sentiment Analysis on Twitter Data

## Overview
This project implements a robust data pipeline for real-time sentiment analysis on Twitter data, leveraging Apache Spark and Databricks for scalable stream processing. It showcases the application of big data technologies to gain deep insights into public opinions on social media.

## Key Components

### 1. **Data Collection**
- Integrated with the Twitter API to continuously fetch and process live tweet data.
- Processed 200K tweets in real-time.

### 2. **Data Processing**
- Developed a multi-stage processing pipeline using PySpark to cleanse, transform, and enrich raw data into actionable insights.
- Implemented in stages from bronze to gold-level data using Delta Lake for efficient data management.

### 3. **Real-Time Analytics**
- Utilized Spark Streaming to analyze and score tweet sentiments in real-time.
- Employed a pre-trained BERT-based sentiment analysis model from Hugging Face.

### 4. **ML Integration and Monitoring**
- Integrated MLflow for model management and performance tracking.
- Ensured high accuracy and reliability of sentiment predictions.

### 5. **Visualization and Reporting**
- Built real-time dashboards using Databricks notebooks and DB SQL for sentiment trend monitoring and analytical reporting.

### 6. **Optimization and Performance Tuning**
- Enhanced system efficiency by optimizing Spark configurations and addressing performance bottlenecks.

## Skills Developed
This project enhanced my skills in:
- Big data architectures
- Stream processing using Apache Spark and Databricks
- Real-time data analytics and sentiment analysis
- Model management with MLflow
- Data visualization with Databricks and DB SQL

## Conclusion
This project not only bolstered my skills in big data and stream processing but also provided valuable insights into public opinions on social media. It demonstrates the power of real-time data analytics in understanding consumer behavior and shaping business strategies.
