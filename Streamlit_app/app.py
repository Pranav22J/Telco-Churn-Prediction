import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import warnings



def import_data():
    """A function to import the preprocessed data"""

    features = pd.read_csv("https://raw.githubusercontent.com/Pranav22J/Telco-Churn-Prediction/refs/heads/main/Data/Filtered_Data/features.csv", index_col = 0)
    # Read target and convert to 1D array
    target = pd.read_csv("https://raw.githubusercontent.com/Pranav22J/Telco-Churn-Prediction/refs/heads/main/Data/Filtered_Data/target.csv", index_col=0)
    target = target.iloc[:, 0].values  # Get first column as 1D array
    telco_data = pd.read_csv("https://raw.githubusercontent.com/Pranav22J/Telco-Churn-Prediction/refs/heads/main/Data/Filtered_Data/telco_data.csv", index_col=[0])

    return features, target, telco_data

def preprocess_raw_data(features):
    """A function to preprocess the raw data to input into machine learning model"""

    # Top 10 most influential features (as determined in the ML Notebook analysis)
    cols_to_keep = ['Contract_N','Dependents_N','No_Internet_Service_N','Tenure','Payment_Method_N','Fiber_Optic_N',
                    'Tech_Support_N', 'San Diego','Online_Security_N','Partner_N']
    features = features[cols_to_keep]

    return features, cols_to_keep


def create_model(features, target):
    """A function to create the ML Model"""
    
    # Scale the features
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Splitting the data    
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

    # Adressing imbalanced data via oversampling
    oversample = SMOTE()
    X_train, y_train  = oversample.fit_resample(X_train, y_train)

    # Initializing the Log reg model and training it
    logistic_reg = LogisticRegression()
    logistic_reg.fit(X_train, y_train)

    return logistic_reg, scaler

def preprocess_input(input_df, scaler, cols_to_keep):
    """A function to preprocess the user inputs and predict the output"""
    
    # Creating a new dataframe with the same column names used to train the model (to eliminate error in prediction)
    processed_inputs = pd.DataFrame(index=[0], columns = cols_to_keep)

    # Create mappings for encodings
    yes_no_map = {'Yes': 1, 'No': 0}

    # Encode the inputs made by the user
    processed_inputs['Contract_N'] = input_df['contract'].map({'One year': 1, 'Two year': 1, 'Month-to-month': 0})
    processed_inputs['Dependents_N'] = input_df['dependents'].map(yes_no_map)
    processed_inputs['No_Internet_Service_N'] = (input_df['internet_service'] == 'No').astype(int)
    processed_inputs['Tenure'] = input_df['tenure']
    processed_inputs['Payment_Method_N'] = input_df['payment_method'].map({'Electronic check': 0,'Mailed check': 0,'Bank transfer (automatic)': 1,'Credit card (automatic)': 1
    })
    processed_inputs['Fiber_Optic_N'] = (input_df['internet_service'] == 'Fiber optic').astype(int)
    processed_inputs['Tech_Support_N'] = input_df['tech_support'].map(yes_no_map)
    processed_inputs['San Diego'] = (input_df['city'] == 'Yes').astype(int)
    processed_inputs['Online_Security_N'] = input_df['online_security'].map(yes_no_map)
    processed_inputs['Partner_N'] = input_df['partner'].map(yes_no_map)

    # Scale the input features in the same order as the model was trained to eliminate error
    processed_inputs = processed_inputs[cols_to_keep] # Ensure same order
    processed_inputs_scaled = pd.DataFrame(scaler.transform(processed_inputs), columns = cols_to_keep)

    return processed_inputs_scaled

def create_app(model, scaler, cols_to_keep):
    """Function used to create the UI in the streamlit app"""

    # Create the title and description
    st.title("Telco Customer Churn Prediction")
    st.info("A Machine Learning Model Utilizing Logistic Regression to Predict Customer Churn")

    st.subheader("Customer Information")

    # Creating the input boxes
    contract = st.selectbox('Contract Type', ('Month-to-month', 'One year', 'Two year'))
    dependents = st.selectbox('Has Dependents?', ('Yes', 'No'))
    internet_service = st.selectbox("Internet Service", ("DSL", "Fiber optic", "No"))
    payment_method = st.selectbox("Payment Method", ('Electronic check', 'Mailed check', 
                                                   'Bank transfer (automatic)', 'Credit card (automatic)'))
    city = st.selectbox("Lives in San Diego?", ('Yes', 'No'))
    fiber_optic = st.selectbox("Has fiber optic?", ('Yes', 'No'))
    tech_support = st.selectbox('Has tech support?', ('Yes', 'No'))
    online_security = st.selectbox('Has Online Security?', ('Yes', 'No'))
    partner = st.selectbox("Has a partner?", ('Yes', 'No'))
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72)

    # Predict Button
    if st.button("Predict Churn"):
            
        # Storing user input as a dictionary and converting to a dataframe
        input_features = {
            'contract': contract,
            'dependents': dependents,
            'internet_service': internet_service,
            'payment_method': payment_method,
            'city': city,
            'fiber_optic': fiber_optic,
            'tech_support': tech_support,
            'online_security': online_security,
            'partner': partner,
            'tenure': tenure
        }

        # Converting the inputs to a dataframe
        input_df = pd.DataFrame(input_features, index=[0])

        # Preprocess input
        processed_input = preprocess_input(input_df, scaler, cols_to_keep)

        st.subheader("Churn Prediction Probability")

        # Make prediction
        prediction = model.predict(processed_input)
    
        # Create probability of prediction
        prediction_prob = model.predict_proba(processed_input[:1])
        df_prediction_prob = pd.DataFrame(prediction_prob)
        df_prediction_prob.columns = ["Won't Churn",'Will Churn']
        df_prediction_prob = df_prediction_prob * 100

        # Create Dataframe displaying probability of prediction
        st.dataframe(df_prediction_prob, column_config={
            "Won't Churn": st.column_config.ProgressColumn(
                "Won't Churn",
                format='%d%%',
                width = 'medium',
                min_value = 0,
                max_value = 100
            ),"Will Churn": st.column_config.ProgressColumn(
                "Will Churn",
                format='%d%%',
                width = 'medium',
                min_value = 0,
                max_value = 100
            )
        }, hide_index = True)


        # Display prediction
        if prediction[0]==1:
            st.error('Customer may be at risk for churn')
        else:
            st.success('Customer may not be at risk for churn')

def main():
    # Load data
    features, target, telco_data = import_data()

    # Pre process raw data
    features, cols_to_keep = preprocess_raw_data(features)

    # Create the model
    model, scaler = create_model(features, target)

    # Create app
    create_app(model,scaler, cols_to_keep)

if __name__ == "__main__":
    main()