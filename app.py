from flask import Flask, render_template, request
import pandas as pd
import joblib
from pathlib import Path

app = Flask(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "models" / "model.pkl")
preprocessor = joblib.load(BASE_DIR / "models" / "preprocessor.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        # Get form data
        form_data = {
            "age": int(request.form["age"]),
            "ratings": float(request.form["ratings"]),
            "pickup_time_minutes": int(request.form["pickup_time_minutes"]),
            "distance": float(request.form["distance"]),
            "weather": request.form["weather"],
            "type_of_order": request.form["type_of_order"],
            "type_of_vehicle": request.form["type_of_vehicle"],
            "festival": request.form["festival"],
            "city_type": request.form["city_type"],
            "is_weekend": request.form["is_weekend"],
            "order_time_of_day": request.form["order_time_of_day"],
            "traffic": request.form["traffic"],
            "distance_type": request.form["distance_type"],
            "multiple_deliveries": int(request.form["multiple_deliveries"]),
            "vehicle_condition": int(request.form["vehicle_condition"])
        }

        # Convert to DataFrame
        df = pd.DataFrame([form_data])

        # Preprocess and Predict
        transformed = preprocessor.transform(df)
        pred = model.predict(transformed)[0]
        prediction = round(pred, 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
