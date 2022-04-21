from flask import Flask, request, jsonify
import joblib
import numpy as np

## Import any other packages that are needed

app = Flask(__name__)

# 1. Load your model here
model = joblib.load("../Milestone_3/model_updated.joblib")

# 2. Define a prediction function
def return_prediction(input_data):

    # format input_data here so that you can pass it to model.predict()
    formatted_data = [[float(x) for x in input_data["data"].split(",")]]

    return model.predict(formatted_data)

# 3. Set up home page using basic html
@app.route("/")
def index():
    # feel free to customize this if you like
    return """
    <h1>Welcome to our rain prediction service</h1>
    To use this service, make a JSON post request to the /predict url with 25 climate model outputs.
    """

# 4. define a new route which will accept POST requests and return model predictions
@app.route('/predict', methods=['POST'])
def rainfall_predict():
    
    content = request.json  # this extracts the JSON content we sent
    prediction = return_prediction(content)

    return jsonify({"prediction":prediction[0]})

if __name__ == "__main__":
    app.run(host='localhost', port=5052, debug=True)