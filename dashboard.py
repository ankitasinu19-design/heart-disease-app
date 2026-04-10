import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import pandas as pd
import joblib

# Load model
model = joblib.load("model.pkl")

app = dash.Dash(__name__)

# Layout
app.layout = html.Div(style={
    'backgroundColor': '#0f172a',
    'color': 'white',
    'padding': '20px'
}, children=[

    html.H1("❤️ Heart Disease AI Dashboard"),

    # -------- INPUT SECTION --------
    html.Div([
        dcc.Input(id='age', type='number', placeholder='Age'),
        dcc.Input(id='height', type='number', placeholder='Height'),
        dcc.Input(id='weight', type='number', placeholder='Weight'),
        dcc.Input(id='ap_hi', type='number', placeholder='Systolic BP'),
        dcc.Input(id='ap_lo', type='number', placeholder='Diastolic BP'),
        html.Button('Predict', id='btn')
    ], style={'marginBottom': '20px'}),

    # -------- OUTPUT CARDS --------
    html.Div(id='output'),

    # -------- GRAPH --------
    dcc.Graph(id='feature_graph')

])

# -------- CALLBACK --------
@app.callback(
    [Output('output', 'children'),
     Output('feature_graph', 'figure')],
    Input('btn', 'n_clicks'),
    State('age', 'value'),
    State('height', 'value'),
    State('weight', 'value'),
    State('ap_hi', 'value'),
    State('ap_lo', 'value')
)
def predict(n, age, height, weight, ap_hi, ap_lo):

    if n is None:
        return "", {}

    # Fake rest inputs (for simplicity)
    gender, cholesterol, gluc = 1,1,1
    smoke, alco, active = 0,0,1

    pulse_pressure = ap_hi - ap_lo
    bmi = weight / ((height/100)**2)
    bp_ratio = ap_hi / ap_lo if ap_lo != 0 else 0

    input_data = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "height": height,
        "weight": weight,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "cholesterol": cholesterol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "pulse_pressure": pulse_pressure,
        "bmi": bmi,
        "bp_ratio": bp_ratio
    }])

    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    result = "High Risk" if pred == 1 else "Low Risk"

    # Feature importance (dummy for now)
    df = pd.DataFrame({
        "Feature": ["Age","BP","BMI","Cholesterol"],
        "Importance": [age, ap_hi, bmi, 2]
    })

    fig = px.bar(df, x="Feature", y="Importance", title="Feature Importance")

    return (
        html.Div([
            html.H2(f"Prediction: {result}"),
            html.H3(f"Risk Probability: {round(prob*100,2)}%")
        ]),
        fig
    )

if __name__ == "__main__":
    app.run(debug=True)