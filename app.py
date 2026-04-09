from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

# ✅ HOME ROUTE (THIS WAS MISSING)
@app.route('/')
def home():
    return "Backend is running 🚀"

# ✅ PREDICT ROUTE
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    age = data['age']
    gender = data['gender']
    height = data['height']
    weight = data['weight']
    ap_hi = data['ap_hi']
    ap_lo = data['ap_lo']
    cholesterol = data['cholesterol']
    gluc = data['gluc']
    smoke = data['smoke']
    alco = data['alco']
    active = data['active']

    pulse_pressure = ap_hi - ap_lo
    bmi = weight / ((height/100) ** 2)
    bp_ratio = ap_hi / ap_lo

    input_data = pd.DataFrame([[ 
        age, gender, height, weight,
        ap_hi, ap_lo, cholesterol,
        gluc, smoke, alco, active,
        pulse_pressure, bmi, bp_ratio
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    return jsonify({
        "prediction": int(prediction),
        "probability": float(probability)
    })

if __name__ == "__main__":
    app.run(debug=True)