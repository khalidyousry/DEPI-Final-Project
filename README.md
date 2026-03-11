🏭 Manufacturing Downtime & Predictive Maintenance Analysis
📌 Project Overview
This project focuses on analyzing manufacturing performance and machine health to mitigate downtime and improve operational efficiency. By leveraging the Predictive Maintenance Dataset, I investigate technical sensor data (Temperature, Speed, Torque) to identify patterns that lead to machine failure.
The project follows a full analytical lifecycle: from deep-dive exploration with SQL and interactive visualization with Excel, to planned advanced forecasting using Python.
🎯 Project Objectives
Analyze Failure Patterns: Identify the primary technical causes of machine downtime.
Operational Benchmarking: Compare failure rates across different product quality grades (High, Medium, Low).
Threshold Identification: Determine critical operational limits for Torque and Temperature to prevent crashes.
Downtime Forecasting: (In-Progress) Build Machine Learning models to predict failures before they occur.
Decision Support: Provide actionable insights for maintenance scheduling.
🛠️ Tools & Technologies
SQL (MS SQL Server/MySQL): Data cleaning, integrity checks, and complex aggregations.
Microsoft Excel (Power Pivot & DAX): Built an interactive professional dashboard for real-time monitoring.
Python (Planned): - Pandas & NumPy: For feature engineering.
Scikit-Learn: For classification and regression models.
Power BI / Tableau (Planned): To scale the visualization for enterprise-level reporting.
🧩 Data Schema & Architecture
To optimize the analysis, the raw sensor data is structured into a logical model:
Fact Table: Fact_Maintenance
UDI, Air_Temp, Process_Temp, Rotational_Speed, Torque, Tool_Wear, Target (Failure/No Failure).
Dimension Tables (Derived):
Dim_Product: Product_ID, Type (L, M, H quality).
Dim_Failure_Type: Detailed categories (Power Failure, Overstrain, Tool Wear, etc.).
🔎 Key Insights (Phase 1: SQL & Excel)
Based on the initial analysis of 10,000 records:
Major Failure Driver: "Power Failure" and "Heat Dissipation" were identified as the most frequent causes of interruptions.
Quality Correlation: Low-grade products (L) exhibit a significantly higher failure frequency compared to High-grade (H) products.
Mechanical Stress: Failures show a strong correlation with high Torque values, specifically when exceeding 60 Nm.
Tool Wear Impact: The analysis identified a specific "Tool Wear" threshold where the probability of failure increases by nearly 40%.
📈 Machine Learning Roadmap (In-Progress)
I am currently working on the predictive phase using Python:
Feature Engineering: Creating a "Temperature Difference" feature.
Classification: Training a Random Forest Classifier to predict the Target (Failure vs. No Failure).
Evaluation: Using Precision-Recall curves, as the dataset is imbalanced (low failure frequency).
