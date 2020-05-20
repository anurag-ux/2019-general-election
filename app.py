from flask import Flask,render_template,url_for,request,jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    if(prediction==0):
        output='Lost'
    else:
        output='Won'
    return render_template('index.html',prediction_text='The candidate: {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data= request.get_json(force=True)
    prediction=model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)
    
if (__name__ == "__main__"):
    app.run(debug=True)