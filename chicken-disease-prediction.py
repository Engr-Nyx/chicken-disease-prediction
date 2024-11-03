from flask import Flask, request, jsonify
from roboflow import Roboflow
import os

app = Flask(__name__)

rf = Roboflow(api_key="Bvcx7gN0KXZU0p57qKWH")
project = rf.workspace("nyxus").project("chicken-disease-detection-djh8g")
model = project.version(3).model

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
    app.run(debug=True)
