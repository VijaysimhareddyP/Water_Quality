from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('note_modelthree.pkl','rb'))



@app.route('/')
def hello_world():
    return render_template("index3.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    return render_template('index3.html', prediction_text=prediction)

    
    
if __name__ == '__main__':
    app.run(debug=True)

