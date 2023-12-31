from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
df = pd.read_csv('arrhythmia (1).csv')

# Load the trained model
model = joblib.load("./model.pkl")  # Provide the correct path to your model file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input values from the form
        sex = int(request.form['sex'])
        age = int(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        # Add more input fields as needed

        # Create a DataFrame with the input values
        input_data = pd.DataFrame({
            'Sex': [sex],
            'Age': [age],
            'Height': [height],
            'Weight': [weight],
            # Add more columns as needed
        })

        # Scale the input data (assuming you used StandardScaler during training)
        scaler = StandardScaler()
        input_data_scaled = scaler.fit_transform(input_data)

        # Make a prediction using the loaded model
        prediction = model.predict(input_data_scaled)

        # Display the result
        result = "Heart Patient" if prediction == 1 else "No Heart Disease"

        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
