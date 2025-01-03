# Telco Customer Churn Prediction

## Project Overview
A machine learning solution developed to predict customer churn for a telecommunications company experiencing a 25% churn rate. Using historical customer data, the model achieves 78% accuracy and 82% recall in identifying at-risk customers, enabling proactive retention strategies and potential revenue savings.

## Link to Machine Learning App
[ML Churn Prediction App](https://telco-churn-prediction-nkqgdudoi9jlhtwlikomfy.streamlit.app/)

## Technical Skills Demonstrated
- **Data Analysis**: Exploratory analysis, feature engineering, statistical testing
- **Machine Learning**: Model development, feature selection, hyperparameter tuning
- **Visualization**: Interactive plots, statistical plots, geographic analysis
- **Development**: GUI development, model deployment
- **Business Analysis**: Cost-benefit analysis, actionable recommendations

## Technologies Used
- Python
- Scikit-learn
- Pandas/NumPy
- Plotly/Seaborn
- Streamlit
- Scipy


## Project Structure
```
📁 Telco-Churn-Prediction/
├── 📁 Data/
│   ├── 📁 Filtered_Data/
│   │   ├── features.csv
│   │   ├── target.csv
│   │   └── telco_data.csv
│   └── 📁 Raw_Data/
│       ├── CustomerChurn.xlsx
│       ├── Telco_customer_churn_demographics.xlsx
│       └── Telco_customer_churn_location.xlsx
├── 📁 Docs/
│   └── project-overview.md
├── 📁 Images/
│   ├── 📁 EDA_plots/
│   │   ├── churn_rates_for_telco_services.png
│   │   ├── churned_customers_per_gender.png
│   │   ├── customer_count_and_churn_prop.png
│   │   ├── gender_distribution_telco_dataset.png
│   │   └── telco_customer_count_and_churn.png
│   └── 📁 ML_plots/
│       └── distribution_of_target_variable.png
├── 📁 Notebooks/
│   ├── EDA.ipynb
│   └── ML.ipynb
├── 📁 Streamlit_app/
│   ├── app.py
│   └── requirements.txt
├── .gitattributes
├── .gitignore
└── README.md                 
```

## Key Features
1. **Predictive Modeling**
   - Logistic Regression and Random Forest models
   - Feature importance analysis
   - Model performance optimization
   - Cross-validation testing

2. **Customer Analysis**
   - Demographic profiling
   - Service usage patterns
   - Geographic distribution analysis
   - Churn factor identification

3. **Interactive Prediction Tool**
   - Real-time churn prediction
   - Customer risk assessment
   - Retention recommendation engine

## Data Analysis Highlights
- Contract length is the strongest predictor of churn
- Customers without dependents show 23% higher churn probability
- Fiber optic internet users demonstrate increased churn risk
- Technical support subscribers show 14% lower churn rate
- San Diego market exhibits 63% churn rate versus 25% average

## Model Development & Performance
- **Final Model**: Logistic Regression with 10 key features
- **Performance Metrics**:
  - Accuracy: 78%
  - Recall: 82%
  - Precision: 53%
  - AUC-ROC: 0.86
- **Business Impact**: Model enables identification of 82% of potential churners, allowing for proactive retention efforts

## Interactive Application
- Input customer characteristics
- Receive real-time churn probability
- View key factors influencing prediction
- Access recommended retention actions

## Future Enhancement Possibilities
- Integration of customer satisfaction metrics
- Real-time data processing pipeline
- Automated retention strategy recommendations
- Advanced feature engineering
- Deep learning model implementation

## Data Note
This project uses the Telco Customer Churn dataset from Kaggle. To replicate:
1. Download dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Download suplemmentary dataset from [IBM](https://community.ibm.com/community/user/businessanalytics/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)

## Contact Me
- [LinkedIn](https://www.linkedin.com/in/pranav22j/)
- [Email](pj30447@uga.edu)

---
*This project was developed independently as part of my data science portfolio, demonstrating end-to-end machine learning project implementation and business value creation.*
