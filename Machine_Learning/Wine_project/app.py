from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from sklearn import preprocessing

app = Flask(__name__)

# Cargar el modelo previamente entrenados
model_nn = load("Inference_NN.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    df_normalized = preprocessing.normalize(df, norm='l2')
    
    prediction_nn = model_nn.predict(df_normalized)

    return jsonify({
        'nn_predictions': prediction_nn.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)