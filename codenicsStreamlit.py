# libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



df=pd.read_csv('diabetes.csv')
 # Heading
st.title("Diabetes Prediction App ")
st.sidebar.header("Patient Data")
st.subheader("Description Stats of Data")
st.write(df.describe())

# Data Split into X,y and Train Test split
X=df.drop("Outcome",axis=1)
y=df.iloc[:,-1]

X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#function

def user_report():
    # variable=st.sidebar("colum name",min,max,start value)
    pregnancies= st.sidebar.slider("Pregnancies",0,17,2)
    glocuse=st.sidebar.slider("Glucose",0,300,100)
    bp=st.sidebar.slider("Blood Pressure",0,122,80)
    sk=st.sidebar.slider("SkinThickness",0,99,12)
    insulin=st.sidebar.slider("Insulin",0,846,80)
    bmi=st.sidebar.slider("BMI",0,67,5)
    dpf=st.sidebar.slider("DiabetesPedigreeFunction",0.07,2.42,0.37)
    age=st.sidebar.slider("Age",21,81,33)

    user_report_data= {
        "Pregnancies" : pregnancies,
        "Glucose" : glocuse,
        "BloodPressure": bp,
        "SkinThickness":sk,
        "Insulin" : insulin,
        "BMI" : bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age }
    user_data=pd.DataFrame(user_report_data,index=[0])
    return user_data

# Patient Data
user_data=user_report()
st.subheader("Patient Data")
st.write("user_data")

#Model
rc=RandomForestClassifier()
rc.fit(X_train,y_train)
user_result=rc.predict(user_data)

# Visualization 
st.title("Visualized Patient Data")

# color function
if user_result[0]==0 :
    color='blue'
else :
    color ='red'

# Age vs Pregnancies 
st.header("Pregnancies Count Graph(Other vs Yours)")
fig_preg=plt.figure()
ax1=axis=sns.scatterplot(x='Age', y='Pregnancies',data=df, hue='Outcome', palette="Greens")
ax2=sns.scatterplot(x=user_data["Age"],y=user_data["Pregnancies"], s=150, color= color)
plt.xticks(np.arange(10,100,5))
plt.yticks(np.arange(0,20,2))
plt.title("0- Health & 1- Diabetics")
st.pyplot(fig_preg)

#output
st.header("Your Report : ")
output=''
if user_result[0]==0:
    output = "You are Healthy"
    st.balloons()
else :
    output ="Metha kam kao"
    st.warning("Sugar, Sugar")
st.title(output)
# st.subheader("Accuracy: ")
# st.write(str(accuracy_score(y_test,rc.predict(X_test))*100+"%"))