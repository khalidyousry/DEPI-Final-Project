# 🏭 Manufacturing Downtime & Predictive Maintenance Analysis

<table width="100%">
  <tr>
    <td width="30%" valign="top">
      <p align="center">
        <img src="https://img.shields.io/badge/Status-Phase_1_Completed-green?style=for-the-badge" width="100%">
      </p>
      <h3>🛠️ Tech Stack</h3>
      <ul>
        <li><b>Database:</b> SQL (Data Cleaning & EDA)</li>
        <li><b>BI Tool:</b> MS Excel (Power Pivot & DAX)</li>
        <li><b>Planned:</b> Python (Scikit-Learn)</li>
      </ul>
      <hr>
      <h3>📊 Project Metrics</h3>
      <ul>
        <li><b>Dataset Size:</b> 10,000 Records</li>
        <li><b>Failure Rate:</b> 3.39%</li>
        <li><b>Industry:</b> Smart Manufacturing</li>
      </ul>
      <hr>
      <h3>🔗 Connect with Me</h3>
      
    </td>
    
    <td width="70%" valign="top">
      <h2>📌 Project Overview</h2>
      <p>This project, part of <b>"The Zero-Downtime Initiative"</b>, focuses on analyzing machine sensor data to identify and predict manufacturing failures. By analyzing variables like Temperature, Torque, and Tool Wear, we transform raw data into actionable maintenance strategies to minimize production interruptions.</p>
      
      <h3>🚀 Milestones Achieved</h3>
      <ul>
        <li><b>Data Engineering (SQL):</b> Developed structured queries to handle 10k rows, performing data integrity checks and identifying failure correlations.</li>
        <li><b>Interactive UI (Excel):</b> Built a professional dashboard using Power Pivot, providing a real-time view of machine health and operational KPIs.</li>
      </ul>
      
      <h3>💡 Key Analytical Insights</h3>
      <blockquote>
        <b>1. Critical Failure Driver:</b> "Power Failure" and "Heat Dissipation" are responsible for over 60% of total downtime.<br>
        <b>2. Risk Threshold:</b> Machine failure probability increases by <b>40%</b> when Torque exceeds <b>60 Nm</b>.<br>
        <b>3. Quality Impact:</b> Low-grade products (L) exhibit higher failure volatility compared to High-grade (H) products.
      </blockquote>

      <h3>📈 Machine Learning Roadmap (Phase 2)</h3>
      <p>Currently developing a <b>Predictive Model</b> using Python to forecast failures before they occur.
      <ul>
        <li>Handling class imbalance with SMOTE.</li>
        <li>Training Random Forest & XGBoost classifiers.</li>
      </ul>
      </p>
    </td>
  </tr>
</table>

---



## 💻 SQL Analysis Snippet
```sql
-- Analysis of Failure Types and Operational Limits
SELECT 
    failure_type,
    COUNT(*) AS total_failures,
    AVG(rotational_speed_rpm) AS avg_speed,
    AVG(torque_nm) AS avg_torque
FROM maintenance
WHERE Target = 1
GROUP BY failure_type
ORDER BY total_failures DESC;
