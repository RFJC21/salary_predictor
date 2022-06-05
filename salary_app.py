#app
#Exploratory Data Analysis
#Prediuction
#Metrics

#Core Packages
from hashlib import new
from operator import contains
from xml.parsers.expat import model
from nbformat import read
import streamlit as st
import warnings
import os
import sqlite3
import datetime

#EDA Packages
import pandas as pd
import seaborn as sb
import numpy as np
import warnings
warnings.filterwarnings("ignore")

#Data viz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

#ML Packs
import joblib
import sys
sys.modules['sklearn.externals.joblib'] = joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble._forest import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score


#####################################################
#Functions

#get value from dictionary
def get_value(val, my_dict):
    for key, value in my_dict.items():
        if val == key:
            return value

#get keys
def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

# Load models
def load_prediction_models(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

# class to store results
class Monitor(object):
	"""docstring for Monitor"""

	conn = sqlite3.connect('data.db')
	c = conn.cursor()

	def __init__(self,age=None ,workclass=None ,fnlwgt=None ,education=None ,
    education_num=None ,marital_status=None ,occupation=None ,relationship=None ,
    race=None ,sex=None ,capital_gain=None ,capital_loss=None ,hours_per_week=None ,
    native_country=None,predicted_class=None,model_class=None, time_of_prediction=None):
		super(Monitor, self).__init__()
		self.age = age
		self.workclass = workclass
		self.fnlwgt = fnlwgt
		self.education = education
		self.education_num = education_num
		self.marital_status = marital_status
		self.occupation = occupation
		self.relationship = relationship
		self.race = race
		self.sex = sex
		self.capital_gain = capital_gain
		self.capital_loss = capital_loss
		self.hours_per_week = hours_per_week
		self.native_country = native_country
		self.predicted_class = predicted_class
		self.model_class = model_class
		self.time_of_prediction = time_of_prediction
    
	def __repr__(self):
		# return "Monitor(age ={self.age},workclass ={self.workclass},fnlwgt ={self.fnlwgt},education ={self.education},education_num ={self.education_num},marital_status ={self.marital_status},occupation ={self.occupation},relationship ={self.relationship},race ={self.race},sex ={self.sex},capital_gain ={self.capital_gain},capital_loss ={self.capital_loss},hours_per_week ={self.hours_per_week},native_country ={self.native_country},predicted_class ={self.predicted_class},model_class ={self.model_class})".format(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class)
		"Monitor(age = {self.age},workclass = {self.workclass},fnlwgt = {self.fnlwgt},education = {self.education},education_num = {self.education_num},marital_status = {self.marital_status},occupation = {self.occupation},relationship = {self.relationship},race = {self.race},sex = {self.sex},capital_gain = {self.capital_gain},capital_loss = {self.capital_loss},hours_per_week = {self.hours_per_week},native_country = {self.native_country},predicted_class = {self.predicted_class},model_class = {self.model_class})".format(self=self)

	def create_table(self):
		self.c.execute('CREATE TABLE IF NOT EXISTS predictiontable(age NUMERIC,workclass NUMERIC,fnlwgt NUMERIC,education NUMERIC,education_num NUMERIC,marital_status NUMERIC,occupation NUMERIC,relationship NUMERIC,race NUMERIC,sex NUMERIC,capital_gain NUMERIC,capital_loss NUMERIC,hours_per_week NUMERIC,native_country NUMERIC,predicted_class NUMERIC,model_class TEXT)')

	def add_data(self):
		self.c.execute('INSERT INTO predictiontable(age,workclass,fnlwgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,predicted_class,model_class) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(self.age,self.workclass,self.fnlwgt,self.education,self.education_num,self.marital_status,self.occupation,self.relationship,self.race,self.sex,self.capital_gain,self.capital_loss,self.hours_per_week,self.native_country,self.predicted_class,self.model_class))
		self.conn.commit()

	def view_all_data(self):
		self.c.execute('SELECT * FROM predictiontable')
		data = self.c.fetchall()
		# for row in data:
		# 	print(row)
		return data

#####################################################
def main():
    '''Salary Predictor'''

    st.title('Salary Predictor')

    activity = ['EDA', 'Prediction','Metrics', 'Countries'] 
    choice = st.sidebar.selectbox('Choose Activity',activity)

    #load file
    df = pd.read_csv('adult_salary.csv')
    df_not_treated =  pd.read_csv('adult_salary_data.csv')
    
   #To create dummy variables/mappings 
   # Function to Assign Numbers to Unique Values
    #d = {v: k for k, v in enumerate(set(df_not_treated['native-country'].unique()))}
    
    #def generate_dict(data):
        #my_dict = {v: k for k, v in enumerate(set(data))}
        #return my_dict

    #generate_dict(df_not_treated['native-country'].unique())

    #obj_list = ["workclass","education","marital-status ","occupation","relationship","race ","sex ","native-country","class"]
    #obj_names = ["d_workclass","d_education","d_marital-status ","d_occupation","d_relationship","d_race ","d_sex ","d_native-country","d_class"]
    #for i,j in zip(obj_names,obj_list):
        #print('{} = generate_dict(df_not_treated["{}"].unique())'.format(i,j))

    #native = generate_dict(df_not_treated["native-country"].unique())

    #df_not_treated["native-country"] = df_not_treated["native-country"].map(native)
    

##################################################### 
#EDA
    if choice == 'EDA':
        st.subheader('Exploratory Data Analysis')

        #Preview data
        if st.checkbox('Preview data'):
            number_rows = st.number_input('Number of rows', format= '%i', min_value =1, step =1)
            st.dataframe(df.head(int(number_rows)))

        #Show Columns
        if st.checkbox('Show Columns'):
            st.write(df.columns)

        if st.checkbox('Description'):
            st.write(df.describe())  

        if st.checkbox('Select Columns to show:'):
            all_columns = df.columns.to_list()
            selected_columns = st.multiselect('Select Columns', all_columns)
            new_df = df[selected_columns]
            st.dataframe(new_df)  

        #Show rows    
        if st.checkbox('Select rows to show:'):
            selected_index = st.multiselect('Select rows', df.head(10).index)
            selected_rows = df.iloc[selected_index]
            st.dataframe(selected_rows)
        #Value counts
        if st.button('Count Values'):
            st.text('Count by "Class"')
            st.write(df.iloc[:,-1].value_counts()) #-1 -> ultima coluna


        if st.checkbox('Correlation Matrix'):
            mask=np.zeros_like(df.corr())
            triangle_indices=np.triu_indices_from(mask)
            mask[triangle_indices]=True

            plt.figure(figsize=(22,18))
            sns.heatmap(df.corr(), mask=mask, cmap="coolwarm",annot=True, annot_kws={'size':10})
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=12)
            st.pyplot()
            st.set_option('deprecation.showPyplotGlobalUse', False)

#####################################################    
#PREDICTION
#Using the text file
#To be easier to the end user...

    elif choice == 'Prediction':
        st.subheader('Predicitons')
        
        d_workclass = {"Never-worked": 0, "Private": 1, "Federal-gov": 2, "?": 3, "Self-emp-inc": 4, "State-gov": 5, "Local-gov": 6, "Without-pay": 7, "Self-emp-not-inc": 8}
        d_education = {"Some-college": 0, "10th": 1, "Doctorate": 2, "1st-4th": 3, "12th": 4, "Masters": 5, "5th-6th": 6, "9th": 7, "Preschool": 8, "HS-grad": 9, "Assoc-acdm": 10, "Bachelors": 11, "Prof-school": 12, "Assoc-voc": 13, "11th": 14, "7th-8th": 15}
        d_marital_status = {"Separated": 0, "Married-spouse-absent": 1, "Married-AF-spouse": 2, "Married-civ-spouse": 3, "Never-married": 4, "Widowed": 5, "Divorced": 6}
        d_occupation = {"Tech-support": 0, "Farming-fishing": 1, "Prof-specialty": 2, "Sales": 3, "?": 4, "Transport-moving": 5, "Armed-Forces": 6, "Other-service": 7, "Handlers-cleaners": 8, "Exec-managerial": 9, "Adm-clerical": 10, "Craft-repair": 11, "Machine-op-inspct": 12, "Protective-serv": 13, "Priv-house-serv": 14}
        d_relationship = {"Other-relative": 0, "Not-in-family": 1, "Own-child": 2, "Wife": 3, "Husband": 4, "Unmarried": 5}
        d_race = {"Amer-Indian-Eskimo": 0, "Black": 1, "White": 2, "Asian-Pac-Islander": 3, "Other": 4}
        d_sex = {"Female": 0, "Male": 1}
        d_native_country = {"Canada": 0, "Philippines": 1, "Thailand": 2, "Scotland": 3, "Germany": 4, "Portugal": 5, "India": 6, "China": 7, "Japan": 8, "Peru": 9, "France": 10, "Greece": 11, "Taiwan": 12, "Laos": 13, "Hong": 14, "El-Salvador": 15, "Outlying-US(Guam-USVI-etc)": 16, "Yugoslavia": 17, "Cambodia": 18, "Italy": 19, "Honduras": 20, "Puerto-Rico": 21, "Dominican-Republic": 22, "Vietnam": 23, "Poland": 24, "Hungary": 25, "Holand-Netherlands": 26, "Ecuador": 27, "South": 28, "Guatemala": 29, "United-States": 30, "Nicaragua": 31, "Trinadad&Tobago": 32, "Cuba": 33, "Jamaica": 34, "Iran": 35, "?": 36, "Haiti": 37, "Columbia": 38, "Mexico": 39, "England": 40, "Ireland": 41}
        d_class = {">50K": 0, "<=50K": 1}

        # User inputs for ML
        # not considering outliers (use df.describe)
        # using the keys of the dictionary, not the values
    
        st.text('Inputs for Users:')

        age = st.slider('Select Age:', 17, 90)
        workclass = st.selectbox('Select Work Class', tuple(d_workclass.keys()))
        fnlwgt = st.number_input("Enter FNLWGT",1.228500e+04,1.484705e+06)
        education = st.selectbox('Select Work Class', tuple(d_education.keys()))
        education_number = st.slider('Select your level', 1, 16)
        marital_status = st.selectbox("Select Marital-status",tuple(d_marital_status.keys()))
        occupation = st.selectbox("Select Occupation",tuple(d_occupation.keys()))
        relationship = st.selectbox("Select Relationship",tuple(d_relationship.keys()))
        race = st.selectbox("Select Race",tuple(d_race.keys()))
        sex = st.radio("Select Sex",tuple(d_sex.keys()))
        capital_gain = st.number_input("Capital Gain",0,99999)
        capital_loss = st.number_input("Capital Loss",0,4356)
        hours_per_week = st.number_input("Hours Per Week ",0,99)
        native_country = st.selectbox("Select Native Country",tuple(d_native_country.keys()))
    
        # Save user inputs
        # Get values for each input, just the ones not numbers

        k_workclass = get_value(workclass,d_workclass)
        k_education = get_value(education,d_education)
        k_marital_status = get_value(marital_status,d_marital_status)
        k_occupation = get_value(occupation,d_occupation)
        k_relationship = get_value(relationship,d_relationship)
        k_race = get_value(race,d_race)
        k_sex = get_value(sex,d_sex)
        k_native_country = get_value(native_country,d_native_country)

        # Show results to the user
        selected_options = [age ,workclass, fnlwgt, education ,education_number ,marital_status ,
        occupation ,relationship ,race ,sex ,capital_gain ,capital_loss ,hours_per_week ,native_country]
        
        vectorized_result = [age ,k_workclass, fnlwgt, k_education ,education_number ,k_marital_status ,
        k_occupation ,k_relationship ,k_race ,k_sex ,capital_gain ,capital_loss ,hours_per_week ,k_native_country]
        # vectorized are the ones for the ML Model

        # data to predict
        sample_data = np.array(vectorized_result).reshape(1,-1)

        st.subheader('These are the options you choosed:')
        # st.info(selected_options)

        # Results more pritier in a dictionary...
        results_for_user = {
            'age': age,
            "workclass":workclass,
            "fnlwgt":fnlwgt,
            "education":education,
            "education_num":education_number,
            "marital_status":marital_status,
            "occupation":occupation,
            "relationship":relationship,
            "race":race,
            "sex":sex,
            "capital_gain":capital_gain,
            "capital_loss":capital_loss,
            "hours_per_week":hours_per_week,
            "native_country":native_country
        }

        st.json(results_for_user)

        # Results for the model
        st.write(vectorized_result)

        # ML Model
        # Logistic Regression
        df = df.drop('Unnamed: 0', axis = 1)
        target = df.iloc[:,14].name
        # st.info(target)

        X = df.iloc[:,0:14].values
        y = df.iloc[:,14].values

        x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33, random_state=42)

        logit = LogisticRegression()
        logit.fit(x_train,y_train)

        accuracy = logit.score(x_test,y_test)
        #st.success('Accuracy of the model: ' + str(accuracy * 100)+ ' %')

        # Saving the model
        salary_logit_model = open("salary_logit_model.pkl","wb")
        joblib.dump(logit,salary_logit_model)
        salary_logit_model.close()
        
        # Marking actual predictions
        # first create function to load models
        st.subheader('Making Predictions:')
        all_my_list = ['LR']
        #if st.checkbox('Make prediction'):
            #all_my_list = ['LR', 'RFOREST', 'NB']

        model_choice = st.selectbox('Select Model', all_my_list)
        prediction_label = {">50K": 0, "<=50K": 1}
        if st.button('Predict'):

            if model_choice == 'LR':
                model_predictor = load_prediction_models('salary_logit_model.pkl')
                prediction = model_predictor.predict(sample_data)
            
            coment = """elif model_choice == 'RFOREST':
                model_predictor = load_prediction_models("salary_rf_model.pkl")
                prediction = model_predictor.predict(sample_data)
                st.write(prediction)

            elif model_choice == 'NB':
                model_predictor = load_prediction_models("salary_nv_model.pkl")
                prediction = model_predictor.predict(sample_data)
                st.write(prediction)"""

            final_result = get_key(prediction, prediction_label)

            # save predicitons in database
            model_class = model_choice
            time_of_prediction = datetime.datetime.now()
            
            monitor = Monitor(age ,workclass ,fnlwgt ,education ,education_number ,
            marital_status ,occupation ,relationship ,race ,sex ,capital_gain ,
            capital_loss ,hours_per_week ,native_country,final_result,model_choice, time_of_prediction)
            
            monitor.create_table()
            monitor.add_data()

            st.success(f'The predicted salary is: {final_result}')
            st.info('This model has an accuracy of ' + (str(accuracy * 100)+ ' %'))
#####################################################
#METRICS

    elif choice == 'Metrics':
        st.subheader('Metrics')
        # Create your connection.
        cnx = sqlite3.connect('data.db')
        
        mdf = pd.read_sql_query("SELECT * FROM predictiontable", cnx)
        st.dataframe(mdf)


    elif choice == 'Countries':
        st.subheader('Countries Selection')

        # List of Countries
        d_native_country = {"Canada": 0, "Philippines": 1, "Thailand": 2, "Scotland": 3, "Germany": 4, "Portugal": 5, "India": 6, "China": 7, "Japan": 8, "Peru": 9, "France": 10, "Greece": 11, "Taiwan": 12, "Laos": 13, "Hong": 14, "El-Salvador": 15, "Outlying-US(Guam-USVI-etc)": 16, "Yugoslavia": 17, "Cambodia": 18, "Italy": 19, "Honduras": 20, "Puerto-Rico": 21, "Dominican-Republic": 22, "Vietnam": 23, "Poland": 24, "Hungary": 25, "Holand-Netherlands": 26, "Ecuador": 27, "South": 28, "Guatemala": 29, "United-States": 30, "Nicaragua": 31, "Trinadad&Tobago": 32, "Cuba": 33, "Jamaica": 34, "Iran": 35, "?": 36, "Haiti": 37, "Columbia": 38, "Mexico": 39, "England": 40, "Ireland": 41}
        selected_countries = st.selectbox('Choose A Country:', tuple(d_native_country.keys()))

        # Selection Countries
        st.text(selected_countries)
        df2 = pd.read_csv('adult_salary_data.csv')
        
        # Filter dataframe by country
        country_df = df2[df2['native-country'].str.contains(selected_countries)]
        st.dataframe(country_df)

        # Select Columns to show plot
        if st.checkbox("Select Columns To Show"):
            result_df_columns = country_df.columns.tolist()
            selected_columns = st.multiselect('Select',result_df_columns)
            new_df = df2[selected_columns]
            st.dataframe(new_df)

            if st.checkbox("Plot"):
                st.area_chart(df[selected_columns])
                st.pyplot()
                st.set_option('deprecation.showPyplotGlobalUse', False)



if __name__ == '__main__':
    main()

requirements = '''
1- pip freeze > requirements.txt

2- docker file 

FROM python:3.10.4

WORKDIR /app_salary_docker

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8501

COPY . /app_salary_docker

ENTRYPOINT ['streamlit', 'run']

CMD ['salary_app.py']

3- app directory, docker build -t first_docker_app:latest .
4- check docker images: docker images

'''

deploy_streanlit_sharing ='''
1- pip install pipreqs
2- have requiremnts.txt (pipreqs C:\Users\rfjca\OneDrive\Escritorio\python\scripts\ml_apps\projects\salary)

streamlit shared app:
https://share.streamlit.io/rfjc21/first_streamlit_app/main/salary_app.py

'''
