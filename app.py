import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

st.title("Passenger Survival Chance in the Titanic Journey")

pClass=st.slider("Enter the Passenger class for the user",min_value=1,max_value=3)
sex=st.selectbox("Select Gender of passenger",['male','female'])
sibSp=st.slider("Enter the number of Siblings/Spouse the passenger had",min_value=1,max_value=8)
parfCh=st.slider("Enter the number of Parents/Children the passenger had",min_value=0,max_value=6)
fare=st.number_input("Enter the Fare odf the Passenger")
embarked=st.selectbox('Enter the Passenger station from where they started journey',['Southampton','Chebourg','Queenstown'])

data=pd.DataFrame([{
    'Pclass':pClass,
    'Sex':sex,
    'SibSp':sibSp,
    'Parch':parfCh,
    'Fare':fare,
    'Embarked':embarked
}])

model=load_model('model.h5')

with open ('label_encoder.pkl','rb') as file:
    label=pickle.load(file)
with open ('onehot_encoder.pkl','rb') as file:
    onehot=pickle.load(file)
with open ('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

data['Sex']=label.transform(data['Sex'])

embarked=onehot.transform(data[['Embarked']])
embarked=pd.DataFrame(embarked,columns=onehot.get_feature_names_out())
data=pd.concat([data.drop(columns=['Embarked']),embarked],axis=1)

data[['Pclass','SibSp','Parch','Fare']]=scaler.transform(data[['Pclass','SibSp','Parch','Fare']])

y= model.predict(data)
y=y[0][0]

def Chance(y):
    if y>0.5:
        return 'The Passenger will survive the journey'
    else:
        return 'The Passenger will not survive the journey'

if st.button('Predict Survival Chance'):
    st.write("Probability of Passenger Survival Chance",y*100,"%")
    st.write(Chance(y))