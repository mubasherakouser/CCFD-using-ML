import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
import streamlit as st

# Load data
data = pd.read_csv(r'C:\Users\Mubashera Kouser\Downloads\Credit_Card_Fraud_Detection-main\Credit_Card_Fraud_Detection\creditcard.csv')

# Separate legitimate and fraudulent transactions
legit = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]
legit_new = legit.sample(n=len(fraud), random_state=2)
new_df = pd.concat([legit_new, fraud], axis=0)

# Split the data into X and Y
X = new_df.drop('Class', axis=1)
Y = new_df['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2,stratify=Y, random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, Y_train)

#Check model performance
train_data_acc = accuracy_score(model.predict(X_train), Y_train)
test_data_acc = accuracy_score(model.predict(X_test), Y_test)

# Train Random Forest model
#model = RandomForestClassifier(n_estimators=100, random_state=0)
#model.fit(X_train, Y_train)

# Check model performance
#train_data_acc = accuracy_score(model.predict(X_train), Y_train)
#test_data_acc = accuracy_score(model.predict(X_test), Y_test)


# Set page title and favicon
st.set_page_config(page_title="Credit Card Fraud Detection", page_icon=":credit_card:")

# Web app
st.title("Credit Card Fraud Detection Model")
st.markdown("---")

# Input text box
input_data = st.text_input("**Enter all required features values**")
input_splited = input_data.split(',')

# Detect button
detect = st.button("Detect", key="detect_btn", help="Click to detect transaction")

# Perform detection
if detect:
    # Get input feature values
    features = np.array(input_splited, dtype=np.float64)
    # Make prediction
    prediction = model.predict(features.reshape(1,-1))
    # Display result
    if prediction[0] == 0:
        st.success("**Non Fraudulent Transaction**")
    else:
        st.error("**Fraudulent Transaction**")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background: black;
        color: white;
    }
    .stTextInput input {
        border: 3px solid orange;
    }
    .stbutton button{
        background-color: black;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .css-14bw00s button:hover {
        background-color: #333 !important;
    }
    .stMarkdown {
        font-size: 24px;
    }
    </style>
    """,
    unsafe_allow_html=True
)
