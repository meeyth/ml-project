from flask import (
    Flask,
    request,
    jsonify,
)

from src.pipeline.predict_pipeline import CustomData, PredictPipeline


application = Flask(__name__)

app = application


@app.get('/')
def index():
    return "<h1>SUCCESS!</h1>"


@app.post('/predict')
def predict():
    data = CustomData(
        gender=request.json.get('gender'),
        race_ethnicity=request.json.get('ethnicity'),
        parental_level_of_education=request.json.get(
            'parental_level_of_education'),
        lunch=request.json.get('lunch'),
        test_preparation_course=request.json.get(
            'test_preparation_course'),
        reading_score=float(request.json.get('writing_score')),
        writing_score=float(request.json.get('reading_score'))
    )

    pred_df = data.get_data_as_data_frame()
    print(pred_df)
    print("Before Prediction")

    predict_pipeline = PredictPipeline()
    print("Mid Prediction")

    results = predict_pipeline.predict(pred_df)
    print("After Prediction")

    return jsonify({"result": int(results[0])})


if __name__ == "__main__":
    app.run(host="0.0.0.0")
