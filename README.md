# Bank Marketing Term Deposit Prediction

## Overview
This project develops a machine learning system to predict whether clients of a Portuguese banking institution will subscribe to term deposits. Using a dataset from direct marketing campaigns, the model identifies key factors influencing subscription decisions and provides accurate predictions to optimize marketing strategies.

## Dataset
The dataset contains information from a Portuguese bank's direct marketing campaigns, including:
- Client demographics (age, job, marital status, education)
- Financial attributes (default status, housing loans, personal loans)
- Campaign information (contact type, duration, previous contact outcomes)
- Economic indicators (employment variation rate, consumer price index, Euribor rates)

Target variable: Whether the client subscribed to a term deposit (yes/no)

## Features
- **Comprehensive EDA**: In-depth analysis of client demographics, economic indicators, and their relationships with subscription rates
- **Advanced Preprocessing**: Pipeline-based approach for handling categorical variables, missing values, and feature scaling
- **Multiple Models**: Implementation of Logistic Regression, Random Forest, and Gradient Boosting classifiers
- **Class Imbalance Handling**: SMOTE resampling to address the imbalanced nature of the dataset
- **Model Evaluation**: ROC curves, precision-recall analysis, and confusion matrices to assess model performance
- **Feature Importance Analysis**: Identification of key predictors for subscription behavior
- **Hyperparameter Tuning**: Grid search optimization for model parameters
- **Production-Ready Prediction**: Function for scoring new client data

## Technical Stack
- Python 3.x
- pandas, numpy for data manipulation
- scikit-learn for modeling and evaluation
- imbalanced-learn for handling class imbalance
- matplotlib, seaborn for data visualization

## Key Insights
- The dataset shows significant class imbalance (~11% subscription rate)
- Economic indicators strongly correlate with subscription decisions
- Client age, education level, and job type show varying subscription patterns
- Contact method and previous campaign outcomes significantly influence results
- The model achieves 87% accuracy in predicting subscription behavior

## Usage
1. Clone the repository
2. Install required dependencies: `pip install -r requirements.txt`
3. Run the Jupyter notebook for step-by-step analysis
4. Use the provided prediction function for new client data

## Future Improvements
- Deploy model as a web application
- Implement API for real-time prediction
- Add more advanced feature engineering
- Explore deep learning approaches
- Create an interactive dashboard for marketing insights

## License
MIT

## Acknowledgments
- UCI Machine Learning Repository for the original dataset
- The bank marketing dataset is based on [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014.
