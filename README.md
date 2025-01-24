
# Automated Anomaly Detection for Predictive Maintenance

## Project Overview

This project aims to build a machine-learning pipeline to detect anomalies in equipment data for predictive maintenance. This project helps prevent breakdowns, reduce risks, and optimize maintenance schedules by identifying potential system failures in advance.

The solution involves:
- **Data Analysis**: Exploratory data analysis (EDA) to understand patterns and trends in the dataset.
- **Feature Engineering**: Cleaning and transforming the data for improved model performance.
- **Model Training**: Building a machine learning model for binary anomaly detection.
- **Model Optimization**: Fine-tuning the model using hyperparameter tuning.
- **Deployment Plan**: Outlining steps to integrate the trained model into production.

---

## Prerequisites

Before running the code, ensure the following libraries are installed in your Python environment:
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`

You can install them using:

```bash
pip install pandas matplotlib seaborn scikit-learn joblib
```

---

## Dataset

The dataset `AnomaData.xlsx` contains:
- **Features**: Various predictor columns to evaluate system conditions.
- **Target Column (`y`)**: Binary labels (1 = anomaly, 0 = no anomaly).

Place the dataset in the same directory as the script.

---

## How to Run

1. **Download the Code**  
   Save the provided Python script (`anomaly_detection.py`) in the same directory as the dataset.

2. **Run the Script**  
   Execute the script in a Python environment (e.g., Jupyter Notebook, VSCode, or terminal):
   ```bash
   python anomaly_detection.py
   ```

3. **Outputs**  
   The script will:
   - Perform exploratory data analysis (EDA).
   - Train a machine learning model to predict anomalies.
   - Optimize the model with hyperparameter tuning.
   - Generate evaluation metrics like accuracy, confusion matrix, and classification report.
   - Save the trained model as `anomaly_detection_model.pkl`.

---

## Key Components

1. **Exploratory Data Analysis (EDA)**  
   - Identifies missing values, outliers, and correlations between features.
   - Visualizes data distribution and class imbalance.

2. **Feature Engineering**  
   - Handles missing values.
   - Converts the `date` column to datetime (if present).
   - Selects relevant features for model training.

3. **Model Training**  
   - Trains a Random Forest Classifier.
   - Evaluates model performance using metrics like accuracy and precision.

4. **Hyperparameter Tuning**  
   - Uses `GridSearchCV` to optimize the model.
   - Reports the best parameters for improved performance.

5. **Model Deployment Plan**  
   - Saves the trained model as a `.pkl` file.
   - Provides a high-level outline for deploying the model in production.

---

## Next Steps

To deploy the model:
1. Create an API (e.g., using Flask or FastAPI) to serve predictions for new data.
2. Integrate the model into your production system.
3. Monitor model performance and periodically retrain using updated data.

---

## Directory Structure

```
.
├── AnomaData.xlsx          # Dataset file
├── anomaly_detection.py    # Main script
├── README.md               # Documentation
└── anomaly_detection_model.pkl # Trained model file (output)
```

---

