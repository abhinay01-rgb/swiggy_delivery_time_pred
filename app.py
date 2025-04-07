from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Load model
model_path = Path(__file__).parent / "models" / "model.pkl"
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        data = {
            'delivery_distance_km': float(request.form['delivery_distance_km']),
            'order_amount': float(request.form['order_amount']),
            'traffic_level': int(request.form['traffic_level']),
            'rain_intensity': int(request.form['rain_intensity']),
            'pickup_time_hour': int(request.form['pickup_time_hour']),
            'pickup_time_minute': int(request.form['pickup_time_minute'])
        }

        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]

        return render_template('index.html', prediction_text=f"Predicted Delivery Time: {round(prediction, 2)} minutes")

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
