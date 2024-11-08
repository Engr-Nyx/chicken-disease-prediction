from flask import Flask, request, jsonify
from roboflow import Roboflow
import os

app = Flask(__name__)

# Retrieve API key and project configuration from environment variables
api_key = os.getenv("ROBOFLOW_API_KEY")
workspace_name = os.getenv("ROBOFLOW_WORKSPACE")
project_name = os.getenv("ROBOFLOW_PROJECT")
project_version = int(os.getenv("ROBOFLOW_PROJECT_VERSION", "3"))  # Default version to 3 if not set

# Initialize Roboflow with environment variables
rf = Roboflow(api_key=api_key)
project = rf.workspace(workspace_name).project(project_name)
model = project.version(project_version).model

chicken_diseases = [
    {
        "disease": "Salmonella",
        "symptoms": ["Diarrhea", "Fever", "Abdominal cramps"],
        "treatment": "Hydration, antibiotics as prescribed"
    },
    {
        "disease": "Coccidiosis",
        "symptoms": ["Lethargy", "Loss of appetite", "Bloody stools"],
        "treatment": "Coccidiostats and supportive care"
    },
    {
        "disease": "New Castle Disease",
        "symptoms": ["Loss of appetite", "Respiratory distress", "Diarrhea"],
        "treatment": "Vaccination and supportive care"
    },
    {
        "disease": "Healthy",
        "symptoms": ["No visible symptoms"],
        "treatment": "Maintain good hygiene and nutrition"
    }
]

@app.route("/", methods=["GET"])
def home():
    return "Chicken Disease Detection API is up and running!"

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files["image"]
    image_path = "temp.jpg"

    image_file.save(image_path)

    try:
        prediction = model.predict(image_path, confidence=40, overlap=30).json()
        os.remove(image_path)
        return jsonify(prediction)
    except Exception as e:
        os.remove(image_path)
        return jsonify({"error": str(e)}), 500

@app.route("/diseases", methods=["GET"])
def get_diseases():
    return jsonify({"diseases": chicken_diseases})

if __name__ == "__main__":
    # Run the app with a specified host and port
    port = int(os.environ.get("PORT", 5000))  # Use the PORT variable from Render
    app.run(host='0.0.0.0', port=port, debug=True)
