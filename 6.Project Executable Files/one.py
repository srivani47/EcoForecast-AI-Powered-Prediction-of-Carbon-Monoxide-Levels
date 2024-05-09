from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load the saved model and scaler
app = Flask(__name__)
model = pickle.load(open('kmodel.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Render HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Retrieve values from UI and make prediction
@app.route('/submit', methods=["POST", "GET"])
def submit():
    if request.method == 'POST':
        input_features = [float(x) for x in request.form.values() if x]
        names = ['Year', 'Month', 'Day', 'Hour', 'temp', 'humidity']
        data = pd.DataFrame([input_features], columns=names)
        
        # Scale the input data
        scaled_data = scaler.transform(data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        rounded_prediction = round(prediction[0], 2)
        output = f"Predicted Carbon Monoxide Level: {rounded_prediction} ppm."

        return render_template("inner-page.html", result=output)
    else:
        return render_template("inner-page.html")

if __name__ == '__main__':
    app.run(debug=False, port=3333)
