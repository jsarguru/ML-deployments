from flask import request, jsonify, Flask
import numpy as np
import joblib

# Create an instance of the Flask application
app = Flask(__name__)

# Load the trained Logistic Regression model
model = joblib.load('logistic_regression_model.joblib')
print("Logistic Regression model loaded successfully.")

# Load the fitted StandardScaler
scaler = joblib.load('scaler.joblib')
print("StandardScaler loaded successfully.")
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Convert input data to a numpy array or DataFrame
        # Assuming input data is a list of features (e.g., [feature1, feature2, ...])
        # For a single prediction, it should be reshaped to (1, n_features)
        # For multiple predictions, it should be (n_samples, n_features)
        input_data = np.array(data['input_features']).reshape(1, -1)

        # Preprocess the input data using the loaded scaler
        scaled_input_data = scaler.transform(input_data)

        # Make prediction using the loaded model
        prediction = model.predict(scaled_input_data)

        # Convert prediction to a Python list for JSON serialization
        prediction_list = prediction.tolist()

        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction_list})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

print("Prediction endpoint '/predict' defined.") 

if __name__ == '__main__':
    print("Starting Flask app...")
    # Run the Flask application, making it accessible externally
    # debug=True allows for automatic reloads and provides a debugger
    app.run(debug=True,host='0.0.0.0', port=5000)
