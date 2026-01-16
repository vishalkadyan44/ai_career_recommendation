# Personalized Career Path Recommendation System

## Project Overview
This project focuses on building a **Personalized Career Path Recommendation System** using Machine Learning techniques.  
The system analyzes student-related data such as academic background, skills, interests, and extracurricular activities to recommend a suitable future career domain.

The project follows a **modular and structured approach**, separating data processing, model training, evaluation, and application layers.

---

## Problem Statement
Students often struggle to identify an appropriate career path based on their academic performance and interests.  
This project aims to provide a **data-driven career recommendation system** that assists students in making informed career decisions.

---

## Objective
- Analyze student data using machine learning
- Predict suitable career domains
- Compare multiple ML models
- Provide interpretable and structured results

---

## Dataset Information
- **Source:** Public Kaggle Dataset  
- **Type:** Structured tabular data  
- **Format:** CSV  

### Dataset Location
- data/raw/career_data.csv

## Project Track
ai_learning/
│
├── data/
│ ├── raw/ # Original dataset
│ ├── processed/ # Cleaned and encoded data
│ └── models/ # Saved models
│
├── src/
│ ├── data_cleaning.py # Data preprocessing
│ ├── feature_engineering.py
│ ├── eda.py # Exploratory Data Analysis
│ ├── models/ # Model training scripts
│ └── visualizations/ # Generated plots
│
├── notebook/
│ └── ai_learning.ipynb # Main evaluation notebook
│
├── reports/ # Confusion matrices & reports
│
├── app_streamlit.py # Streamlit application
├── requirements.txt
└── README.md



---

## Data Processing Steps
1. Data loading from CSV file
2. Handling missing values
3. Encoding categorical features
4. Feature scaling (where required)
5. Saving cleaned and encoded datasets

## Processed files:
- data/processed/career_data_cleaned.csv
- data/processed/career_data_encoded.csv



---

## Models Used
The following Machine Learning models were implemented and evaluated:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- XGBoost (optional)

Each model was trained using the same feature set to ensure fair comparison.

---

## Evaluation Metrics
Model performance was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Evaluation results and plots are stored in:



---

## Application
A simple **Streamlit-based application** is included to demonstrate model inference on new inputs.

Run the app using:
```bash
streamlit run app_streamlit.py
