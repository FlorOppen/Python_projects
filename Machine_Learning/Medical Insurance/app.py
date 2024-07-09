from flask import Flask, request, jsonify
import pandas as pd
from joblib import load

app = Flask(__name__)

# Load the model pipeline
model_rf = load("rf_model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame(data)
    
    # Use the model pipeline to handle preprocessing and prediction
    prediction_rf = model_rf.predict(df)

    return jsonify({
        'rf_predictions': prediction_rf.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
