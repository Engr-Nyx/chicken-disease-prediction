from flask import Flask, request, jsonify
from roboflow import Roboflow
import os
from PIL import Image, ImageDraw
import uuid
import io
import base64

app = Flask(__name__)

api_key = os.getenv("ROBOFLOW_API_KEY")
workspace_name = os.getenv("ROBOFLOW_WORKSPACE")
project_name = os.getenv("ROBOFLOW_PROJECT")
project_version = int(os.getenv("ROBOFLOW_PROJECT_VERSION", "4"))

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

    try:
        with Image.open(image_file.stream) as img:
            img_path = "temp_image.jpg"
            img.save(img_path)

        prediction = model.predict(img_path, confidence=40, overlap=30).json()

        if not prediction["predictions"]:
            return jsonify({"error": "No predictions made"}), 400

        with Image.open(img_path) as img:
            draw = ImageDraw.Draw(img)

            image_details = {
                "height": img.height,
                "width": img.width
            }

            prediction_details = []

            for pred in prediction["predictions"]:
                class_name = pred["class"]
                confidence = pred["confidence"]
                x_min = pred["x"] - (pred["width"] / 2)
                y_min = pred["y"] - (pred["height"] / 2)
                x_max = pred["x"] + (pred["width"] / 2)
                y_max = pred["y"] + (pred["height"] / 2)

                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

                detection_id = str(uuid.uuid4())

                prediction_details.append({
                    "class": class_name,
                    "class_id": pred["class_id"],
                    "confidence": confidence,
                    "detection_id": detection_id,
                    "height": pred["height"],
                    "image_path": img_path,
                    "prediction_type": "ObjectDetectionModel",
                    "width": pred["width"],
                    "x": pred["x"],
                    "y": pred["y"],
                    "bounding_box": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max
                    }
                })

            prediction_response = {
                "image": image_details,
                "predictions": prediction_details
            }

        output_io = io.BytesIO()
        img.save(output_io, format="JPEG")
        output_io.seek(0)

        os.remove(img_path)

        img_base64 = base64.b64encode(output_io.getvalue()).decode('utf-8')

        response = {
            "image": {
                "base64": img_base64,
                "height": img.height,
                "width": img.width
            },
            "predictions": prediction_details
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/diseases", methods=["GET"])
def get_diseases():
    return jsonify({"diseases": chicken_diseases})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
