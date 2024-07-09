from flask import Flask, request, jsonify
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Cargar el modelo previamente entrenado y el escalador
model_nn = load("Inference_NN.joblib"
scaler = load("scaler.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    
    # Escalar los datos usando StandardScaler
    df_scaled = scaler.transform(df)
    
    prediction_nn = model_nn.predict(df_scaled)

    return jsonify({
        'nn_predictions': prediction_nn.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
