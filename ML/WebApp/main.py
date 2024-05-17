from flask import Flask, render_template, request
import re
import pickle
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import os
dirname = os.path.dirname(__file__)

filename_temp2 = os.path.join(dirname,'./x_train_svm.pkl')
with open(filename_temp2,'rb') as file:
    X_train = pickle.load(file)

enc = DictVectorizer()

filename2 = os.path.join(dirname, './model/SVM.pkl')
class_model = pickle.load(open(filename2,'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/predict',methods=['GET','POST'])
def predict():
    # #! after get value, save to folder and predict
    if request.method == 'POST':
        text1 = request.form.get('class1')
        
        text2 = request.form.get('class2')
        
        text3 = request.form.get('class3')
        text4 = request.form.get('class4')
        text5 = request.form.get('class5')
        text6 = request.form.get('class6')
        
        enc.fit_transform(X_train.to_dict('records'))
        #! 'Title','Theme','score','Duration','Status','Premiered'
        df_temp = pd.DataFrame({'Title': [text1],'Theme': [text2], 'score': [text3], 'Duration':[text4],'Status': [text5],'Premiered': [text6]})
        pred = class_model.predict(enc.transform(df_temp.to_dict("records")))
        
        new_prediction = str(pred[0])
        
        return render_template('classify.html',prediction=new_prediction)
    if request.method == 'GET':
        return render_template('classify.html')






filename3 = os.path.join(dirname, './model/regression.pkl')
reg_model = pickle.load(open(filename3,'rb'))

filename4 = os.path.join(dirname, './csv_file/x_train_score.csv')
x_train_score = pd.read_csv('c:/Users/ASUS/OneDrive/ML_Scientist/MachineLearningCoDiep/MachineLearningFinalProject/ML/WebApp/csv_file/x_train_score.csv')
@app.route('/predictRanked',methods=['GET','POST'])
def regression():
    if request.method == 'POST':
        text1 = request.form.get('mytext_regression1')
        text2 = request.form.get('mytext_regression2')
        text3 = request.form.get('mytext_regression3')
        
        
        enc.fit_transform(x_train_score[['Theme','Genre','Title']].to_dict('records'))
        df_temp = pd.DataFrame({'Theme': [text1], 'Genre': [text2], 'Title': [text3]})
        pred = reg_model.predict(enc.transform(df_temp.to_dict("records")))
        
        new_prediction = str(pred[0])
        
        return render_template('regression.html',prediction=new_prediction)
    if request.method == 'GET':
        return render_template('regression.html')

if __name__ == '__main__':
    app.run(port=8000,debug=True) 