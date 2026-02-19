# Turning Chaos into Victory: Optimizing Random Team Composition in League of Legends ARAM Matches

### 📊 Project Overview
This project investigates whether team composition in League of Legends (ARAM mode) can predict match outcomes more effectively than random chance. By analyzing a dataset of over **30,000 matches** retrieved via the Riot Games API, I developed predictive models to identify higher-probability of win based solely on team champion composition.

### 📄 Documentation
Detailed reports and presentations can be found in the `/docs` folder:
* [**Executive Summary**](./docs/Executive Summary.pdf) – High-level findings and strategic recommendations.
* [**Full Research Paper**](./docs/Full Paper.pdf) – Deep dive into methodology, data cleaning, and statistical analysis.
* [**Final Presentation**](./docs/Presentation.pdf) – Visual breakdown of the project lifecycle.

### 🛠️ Technical Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib/Seaborn
* **Models:** Logistic Regression, Random Forest
* **Data Source:** Riot Games API

### 🚀 Analysis Workflow
The code is organized sequentially to ensure reproducibility:
1.  **`01_data_collection.py`**: Scripts for API interaction and raw data collection.
2.  **`02_data_preparation.py`**: Handling formatting, cleaning, and aggregation of raw data.
3.  **`03_data_exploration.py`**: Exploratory Data Analysis visualizing role distributions and win rates.
4.  **`04_data_model.py`**: Implementation of Logistic Regression model.
5.  **`05_data_model_rf.py`**: Implementation of the Random Forest model.

### 💡 Key Insights
* **The "Mage & Tank" Meta:** Teams prioritizing Mages and Tanks saw a **5-10% increase** in win probability.
* **The Assassin Pitfall:** High-density Assassin compositions consistently underperformed across the sample set.
* **Predictive Accuracy:** Machine Learning models successfully outperformed the baseline (random guessing), proving that "comp diff" is statistically measurable.
* **Unexplained Variance:** Despite a slightly higher than chance accuracy, the models were unable to account for nearly all of the variance in win probability. This is by design as this project only examined the pre-match champion line-up.

---
*Developed as a Data Science Capstone Project for the Western Governors University M.S. Data Analytics degree.*