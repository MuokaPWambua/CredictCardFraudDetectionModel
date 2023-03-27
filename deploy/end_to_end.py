import pickle
import pandas as pd
from flask import Flask, request

model = pickle.load(open('pretrained/model.pkl'))
scaler = pickle.load(open('pretrained/scaler.pkl'))

app = Flask(__name__)

@app.route('/')
def process():
    try:
    
        csv = request.files['csv']

        features = pd.read_csv(csv),
        scaled_features = scaler.transform(features)
        preds = model.predict(scaled_features)

        return {'predictions': preds}
    
    except Exception as e:
        
        print(f"Could not process {str(e)}")
        return {"message" : "Could not process"}

if __name__ == '__main__':
    app.run() 