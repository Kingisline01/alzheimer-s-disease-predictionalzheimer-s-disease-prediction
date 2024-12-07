from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model and preprocessor
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('preprocessor.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    data = request.form.to_dict()
    df = pd.DataFrame([data])

    # Ensure columns are in the correct order
    df = df[['col1', 'col2', 'col3', 'Age', 'Gender', 'Ethnicity', 'EducationLevel']]

    # Apply preprocessing
    preprocessed_data = preprocessor.transform(df)

    # Check if the number of features matches the model's expectation
    if preprocessed_data.shape[1] != model.n_features_in_:
        return f"Error: Mismatch in the number of features. Please check the input. Preprocessed data shape: {preprocessed_data.shape}"

    # Make prediction
    prediction = model.predict(preprocessed_data)
    result = 'Disease Detected' if prediction[0] == 1 else 'No Disease'

    return f'Prediction: {result}'

if __name__ == '__main__':
    app.run(debug=True)
