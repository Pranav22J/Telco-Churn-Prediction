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
📁 telco-churn-prediction/
├── 📁 notebooks/
│   ├── exploratory_analysis.ipynb      # EDA and visualization
│   └── machine_learning.ipynb          # Model development
├── 📁 streamlit_app/
│   ├── app.py                          # Interactive prediction app
│   └── requirements.txt
├── 📁 docs/
│   └── project_overview.md             # Detailed documentation
📁 data/
├── 📁 raw_data/                  # Original downloaded files
│   ├── CustomerChurn.xlsx
│   ├── Telco_customer_churn.xlsx
│   └── ... (other source files)
├── 📁 filtered_data/             # Processed data
│   ├── Telco_Data.csv           # Cleaned Data
│   ├── features.csv             # Model features
│   └── target.csv               # Target variable
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
