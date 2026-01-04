from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from src.datascience.pipeline.prediction_pipeline import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            person_age =float(request.form['person_age'])
            person_income=float(request.form['person_income'])
            person_emp_length =float(request.form['person_emp_length'])
            loan_amnt =int(request.form['loan_amnt'])
            loan_int_rate =float(request.form['loan_int_rate'])
            loan_status =float(request.form['loan_status'])
            loan_percent_income =float(request.form['loan_percent_income'])
            cb_person_cred_hist_length =float(request.form['cb_person_cred_hist_length'])
            person_home_ownership_OTHER=int(request.form['person_home_ownership_OTHER'])
            person_home_ownership_OWN=int(request.form['person_home_ownership_OWN'])
            person_home_ownership_RENT=int(request.form['person_home_ownership_RENT'])
            loan_intent_EDUCATION=int(request.form['loan_intent_EDUCATION'])
            loan_intent_HOMEIMPROVEMENT=int(request.form['loan_intent_HOMEIMPROVEMENT'])
            loan_intent_MEDICAL=int(request.form['loan_intent_MEDICAL'])
            loan_intent_PERSONAL=int(request.form['loan_intent_PERSONAL'])
            loan_intent_VENTURE=int(request.form['loan_intent_VENTURE'])
            loan_grade_B=int(request.form['loan_grade_B'])
            loan_grade_C=int(request.form['loan_grade_C'])
            loan_grade_D=int(request.form['loan_grade_D'])
            loan_grade_E=int(request.form['loan_grade_E'])
            loan_grade_F=int(request.form['loan_grade_F'])
            loan_grade_G=int(request.form['loan_grade_G'])
       
         
            data = [person_age,person_income,person_emp_length,loan_amnt,loan_int_rate,loan_status,loan_percent_income,cb_person_cred_hist_length,
                    person_home_ownership_OTHER,person_home_ownership_OWN,person_home_ownership_RENT,loan_intent_EDUCATION,loan_intent_HOMEIMPROVEMENT,
                    loan_intent_MEDICAL,loan_intent_PERSONAL,loan_intent_VENTURE,loan_grade_B,loan_grade_C,loan_grade_D,loan_grade_E,loan_grade_F,loan_grade_G]
            data = np.array(data).reshape(1, 22)
            
            obj = PredictionPipeline()
            predict = obj.predict(data)

            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	
	app.run(host="0.0.0.0", port = 8080)