# 🏭 Manufacturing Downtime Analysis Project

## 📌 Project Overview
This project aims to analyze manufacturing downtime in order to identify the main causes of production interruptions and improve operational efficiency.  

The project will combine **SQL, Python, and Data Visualization tools** to clean, analyze, and forecast downtime trends in manufacturing operations.

Currently, the project is in the **data cleaning and preprocessing stage using SQL**.

---

## 🎯 Project Objectives

- Analyze downtime patterns in manufacturing operations.
- Identify the most common downtime causes.
- Measure the impact of downtime on production efficiency.
- Forecast future downtime using machine learning models.
- Build a visualization dashboard to present key insights.

---

## 🛠️ Tools & Technologies

The following tools will be used throughout the project:

- **SQL** → Data cleaning and querying  
- **Python**
  - Pandas → Data preprocessing
  - Matplotlib → Data visualization
  - Scikit-learn → Forecasting models
- **Power BI / Tableau** → Dashboard visualization
- **Jupyter Notebook** → Data analysis workflow

---

## 🧩 Data Model

The project will implement a **Star Schema data model** to structure the dataset for analytical queries.

### Fact Table
**Fact_Downtime**
- Downtime Duration
- Production Time
- Batch Information
- Foreign keys linking to dimension tables

### Dimension Tables
- **Dim_Date**
- **Dim_Machine**
- **Dim_Product**
- **Dim_Operator**
- **Dim_Downtime_Cause**

This schema will improve query performance and support business intelligence reporting.

---

## 📊 Planned Analysis Questions

The project will explore questions such as:

- What are the most frequent downtime causes?
- Which machines experience the highest downtime?
- What time periods have the most downtime?
- How does downtime affect production output?
- Which products are associated with higher downtime?

---

## 📈 Forecasting (Planned)

Machine learning models will be developed to:

- Predict downtime for the next operational day
- Estimate production batches based on downtime trends

Models will be implemented using **Scikit-learn**.

---

## 📊 Dashboard (Planned)

A visualization dashboard will be built to display:

- Total Downtime
- Average Production Time
- Downtime by Machine
- Downtime by Cause
- Production Trends

The dashboard will help decision makers monitor manufacturing performance.

---

## 🔎 Key Insights

Insights will be added after completing the data analysis phase.\

---

## 🤖 Machine Learning (Planned)

Machine learning techniques will be applied to predict manufacturing downtime and improve operational decision-making.

The model will be trained using historical production and downtime data to identify patterns and forecast potential interruptions.

### Planned Steps
- Feature engineering from production and downtime data
- Train machine learning models to predict downtime
- Evaluate model performance using appropriate metrics
- Use the predictions to support production planning

### Possible Models
- Linear Regression
- Decision Tree
- Random Forest
- Support Vector Regression (SVR)

The results of the models will later be integrated into the final analysis and visualization dashboard.
