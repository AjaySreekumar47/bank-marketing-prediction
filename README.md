# Bank Marketing Term Deposit Prediction

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling Approach](#modeling-approach)
- [Results and Evaluation](#results-and-evaluation)
- [Key Insights](#key-insights)
- [Usage Examples](#usage-examples)
- [Future Work](#future-work)
- [License](#license)
- [Contact](#contact)

## 🎯 Project Overview

This project develops a machine learning system that predicts whether clients of a Portuguese banking institution will subscribe to term deposits based on data from direct marketing campaigns. The goal is to identify patterns in client behavior and develop models that can accurately predict subscription likelihood, allowing the bank to optimize their marketing efforts.

The project implements a complete machine learning pipeline including:
- Data exploration and visualization
- Feature engineering and preprocessing
- Model building and evaluation
- Handling class imbalance
- Hyperparameter tuning
- Model deployment considerations

## 📊 Dataset Description

The dataset contains information from a Portuguese bank's direct marketing campaigns (phone calls) from May 2008 to November 2010. It includes 41,188 records with 20 input features and a binary output variable indicating whether the client subscribed to a term deposit.

### Input Features
- **Client Information**: 
  - `age`: Client's age (numeric)
  - `job`: Type of job (categorical)
  - `marital`: Marital status (categorical)
  - `education`: Education level (categorical)
  - `default`: Has credit in default? (categorical)
  - `housing`: Has housing loan? (categorical)
  - `loan`: Has personal loan? (categorical)

- **Campaign Information**:
  - `contact`: Contact communication type (categorical)
  - `month`: Last contact month of year (categorical)
  - `day_of_week`: Last contact day of the week (categorical)
  - `duration`: Last contact duration in seconds (numeric)
  - `campaign`: Number of contacts performed during this campaign (numeric)
  - `pdays`: Number of days since client was last contacted (numeric)
  - `previous`: Number of contacts before this campaign (numeric)
  - `poutcome`: Outcome of the previous marketing campaign (categorical)

- **Economic Context**:
  - `emp.var.rate`: Employment variation rate (numeric)
  - `cons.price.idx`: Consumer price index (numeric)
  - `cons.conf.idx`: Consumer confidence index (numeric)
  - `euribor3m`: Euribor 3 month rate (numeric)
  - `nr.employed`: Number of employees (numeric)

### Target Variable
- `y`: Has the client subscribed to a term deposit? (binary: 'yes', 'no')

## 💻 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/bank-marketing-prediction.git
cd bank-marketing-prediction

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## 📁 Project Structure

```
bank-marketing-prediction/
│
├── data/
│   ├── bankadditionalfull.csv      # Full dataset
│   └── data_description.md         # Detailed data description
│
├── notebooks/
│   ├── 1_exploratory_data_analysis.ipynb  # Data exploration
│   ├── 2_preprocessing_and_feature_engineering.ipynb
│   ├── 3_model_building_and_evaluation.ipynb
│   └── 4_model_optimization.ipynb
│
├── src/
│   ├── preprocessing.py            # Data preprocessing utilities
│   ├── feature_engineering.py      # Feature engineering functions
│   ├── model.py                    # Model implementation
│   └── evaluation.py               # Evaluation metrics and utilities
│
├── images/                         # Visualizations and plots
├── models/                         # Saved model files
├── README.md                       # Project documentation
├── requirements.txt                # Required packages
└── LICENSE                         # License information
```

## 🔍 Exploratory Data Analysis

The EDA process revealed several key insights:

1. **Class Imbalance**: Only about 11.27% of clients subscribed to term deposits, indicating significant class imbalance.

2. **Age Distribution**: Subscription rates vary notably by age group, with clients over 60 showing the highest subscription rate (45.49%).

3. **Job Types**: Different professions show varying subscription patterns, with students, retired individuals, and management professionals more likely to subscribe.

4. **Economic Indicators**: Strong correlations exist between economic factors (especially Euribor rates and employment variation) and subscription decisions.

5. **Previous Campaign Outcomes**: Clients who succeeded in previous campaigns are significantly more likely to subscribe again.

## 🧠 Modeling Approach

The project implements and compares multiple classification models:

1. **Logistic Regression**: As a baseline model with good interpretability.

2. **Random Forest**: To capture complex, non-linear relationships in the data.

3. **Gradient Boosting**: For potentially higher predictive performance.

Each model is implemented within a scikit-learn pipeline that includes:
- Preprocessing steps (imputation, encoding, scaling)
- Model training and prediction
- Hyperparameter tuning
- Evaluation metrics suitable for imbalanced classification

Additionally, SMOTE (Synthetic Minority Over-sampling Technique) is applied to address the class imbalance issue.

## 📈 Results and Evaluation

The models were evaluated using metrics appropriate for imbalanced classification:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.85 | 0.62 | 0.65 | 0.63 | 0.86 |
| Random Forest | 0.87 | 0.68 | 0.67 | 0.67 | 0.90 |
| Gradient Boosting | 0.88 | 0.70 | 0.68 | 0.69 | 0.91 |
| RF with SMOTE | 0.84 | 0.59 | 0.77 | 0.67 | 0.88 |

The Gradient Boosting model achieved the best overall performance, with Random Forest being a close second. When SMOTE was applied, recall improved significantly at the cost of some precision.

## 💡 Key Insights

1. **Feature Importance**: The duration of the last contact is highly predictive but would not be available in a real-world prediction scenario before making the call.

2. **Economic Context**: Macroeconomic indicators have a strong influence on client decisions, suggesting that marketing campaigns should be timed according to economic conditions.

3. **Previous Interactions**: Clients with successful previous interactions are much more likely to subscribe, highlighting the importance of relationship building.

4. **Demographic Targeting**: Older clients, higher education levels, and certain job categories show higher subscription rates, which could inform targeted marketing.

5. **Contact Timing**: The month and day of contact show variations in effectiveness, indicating potential for optimization.

## 🚀 Usage Examples

### Loading the Model and Making Predictions

```python
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('models/best_model.pkl')

# Prepare data for a new client
new_client = pd.DataFrame([{
    'age': 58,
    'job': 'management',
    'marital': 'married',
    'education': 'university.degree',
    'default': 'no',
    'housing': 'yes',
    'loan': 'no',
    'contact': 'cellular',
    'month': 'may',
    'day_of_week': 'mon',
    'duration': 200,
    'campaign': 1,
    'pdays': 999,
    'previous': 0,
    'poutcome': 'nonexistent',
    'emp.var.rate': 1.1,
    'cons.price.idx': 93.994,
    'cons.conf.idx': -36.4,
    'euribor3m': 4.857,
    'nr.employed': 5191.0
}])

# Make prediction
subscription_prob = model.predict_proba(new_client)[:, 1]
print(f"Probability of subscription: {subscription_prob[0]:.2%}")
```

## 🔮 Future Work

1. **Feature Engineering**: Develop more sophisticated features that capture interaction effects between variables.

2. **Advanced Models**: Explore deep learning approaches such as neural networks for potentially higher performance.

3. **Web Application**: Develop a user-friendly interface for marketing teams to input client information and get subscription predictions.

4. **API Development**: Create an API for real-time prediction integration with the bank's systems.

5. **Explainable AI**: Implement methods to better explain model predictions to business stakeholders.

6. **Time Series Analysis**: Incorporate time-dependent patterns and seasonality effects.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📫 Contact

Your Name - [ajaysreekumar.nmims@gmail.com](mailto:ajaysreekumar.nmims@gmail.com)
