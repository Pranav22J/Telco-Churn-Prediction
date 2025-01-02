# Telco Customer Churn Prediction Project

## Executive Summary
This project addresses a critical business challenge in the telecommunications industry: High customer churn. Using a comprehensive dataset from a California-based telecommunications company, I developed a machine learning solution that predicts customer churn with 78% accuracy and 82% recall, providing actionable insights for customer retention strategies.

## Problem Statement & Business Context
The telecommunications industry faces significant challenges with customer churn, which directly impacts revenue and growth. Telco, a California-based telecommunications company, experiences a concerning 25% churn rate. If this problem were to go unsolved, Telco would be losing sales and revenue. This project aims to:
- Develop a predictive model for early churn identification
- Identify key factors influencing customer churn
- Provide data-driven recommendations for retention strategies
- Enable proactive intervention through an interactive prediction tool

## Dataset Overview
- **Source**: Telco Customer Churn Dataset
- **Scope**: 7,043 customer records
- **Features**: 35 variables including:
  - Demographics (age, gender, location)
  - Service subscriptions (phone, internet, streaming)
  - Account information (tenure, charges, contract type)
  - Customer status (active/churned)
- **Target Variable**: Binary churn indicator (24.35% churn rate), (Yes, No)

## Exploratory Data Analysis Key Findings

### Customer Demographics
- Age distribution is approximately uniform from 19-64 years
- No significant gender bias in customer base or churn rates
- Customers with dependents show lower churn rates (-23% correlation)

### Service Insights
1. **Internet Services**
   - Fiber optic customers show higher churn propensity
   - Customers without internet service have lowest churn rates (7%)

2. **Contract Types**
   - Month-to-month contracts have highest churn risk
   - Long-term contracts (1-2 years) show significantly lower churn

3. **Additional Services**
   - Technical support subscribers show 13% churn rate vs. 27% for non-subscribers
   - Online security and device protection services correlate with lower churn

### Geographic Analysis
- San Diego shows notably higher churn (63%) compared to other cities
- 11 smaller cities show 100% churn rate, requiring immediate attention

## Model Development

### Feature Engineering & Preprocessing
1. **Data Cleaning**
   - Handled missing values affecting 551 records
   - Encoded categorical variables using custom mapping
   - Applied MinMaxScaler for numeric features

2. **Feature Selection**
   - Used Chi-squared test for feature importance
   - Identified top 10 predictive features including:
     - Contract length (highest impact)
     - Number of dependents
     - Tenure
     - Internet service type
     - Payment method

### Model Selection & Evaluation

#### Logistic Regression (Selected Model)
- **Performance Metrics**:
  - Accuracy: 78%
  - Recall: 82%
  - Precision: 53%
  - AUC-ROC: 0.86
- **Cross-validation Score**: 0.82 (σ = 0.02)

#### Random Forest Comparison
- **Performance Metrics**:
  - Accuracy: 77%
  - Recall: 61%
  - Precision: 53%
- **After Hyperparameter Tuning**:
  - Improved recall to 76%
  - Maintained similar accuracy and precision

### Model Selection Rationale
Selected Logistic Regression with 10 features due to:
1. Superior recall rate (82% vs 76%)
2. Better balance of false positives and negatives
3. Strong cross-validation stability
4. Model interpretability for business stakeholders

## Business Impact Analysis

### Cost-Benefit Considerations

#### False Negatives (Critical)
- Missed churn predictions lead to direct revenue loss
- Higher cost impact than false positives
- Model achieves 82% recall, missing only 18% of potential churners

#### False Positives (Manageable)
- Over-prediction of churn (47% false positive rate)
- Lower business impact than false negatives
- May actually strengthen customer relationships through proactive engagement

## Business Recommendations

1. **Contract Strategy**
   - Promote longer-term contracts
   - Develop attractive 1-2 year contract packages
   - Create incentives for month-to-month customers to upgrade

2. **Service Bundles**
   - Package technical support with core services
   - Increase focus on security and device protection offerings
   - Create value-added service bundles

3. **Geographic Focus**
   - Implement targeted retention programs in San Diego
   - Investigate and address issues in high-churn cities
   - Develop location-specific retention strategies

4. **Customer Engagement**
   - Proactive engagement for customers with churn risk factors
   - Enhanced support for fiber optic service customers
   - Special attention to customers without dependents

## Interactive Prediction Tool
- Deployed Streamlit application for real-time churn prediction
- Enables customer service representatives to:
  - Input customer information
  - Receive immediate churn risk assessment
  - Access recommended retention actions
- [Link to Live Application]

## Future Enhancements
1. Model refinement with additional customer interaction data
2. Integration of customer satisfaction metrics
3. Development of automated retention action recommendations
4. Enhanced geographic analysis capabilities

## Technical Implementation
- **Documentation**: Detailed notebooks for EDA and modeling
- **Tech Stack**: Python, Scikit-learn, Pandas, Streamlit
- **Deployment**: Streamlit Cloud
