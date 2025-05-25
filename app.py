from flask import Flask, render_template, request
import numpy as np

from model.model import *
project_root = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)


@app.route("/")
def main():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def home():
    # Get form data
    gender = int(request.form["gender"])
    age = int(request.form["age"])
    academic_pressure = float(request.form["academic_pressure"])
    study_satisfaction = float(request.form["study_satisfaction"])
    sleep_duration = int(request.form["sleep_duration"])
    dietary_habits = int(request.form["dietary_habits"])
    suicidal_thoughts = int(request.form["suicidal_thoughts"])
    study_hours = int(request.form["study_hours"])
    financial_stress = float(request.form["financial_stress"])
    mental_illness = int(request.form["mental_illness"])

    model_input = np.array(
        [
            [
                gender,
                age,
                academic_pressure,
                study_satisfaction,
                sleep_duration,
                dietary_habits,
                suicidal_thoughts,
                study_hours,
                financial_stress,
                mental_illness,
            ]
        ]
    )

    prediction = create_plot(model, model_input)
    pred_rounded = round(prediction[0][0], 2)
    print(pred_rounded)
    if pred_rounded > 0.85:
        message = "You have a high level of depression according to the high level of these indicators below. Please consider seeking professional help."
    elif 0.50 <= pred_rounded <= 0.85:
        message = "You have a moderate level of depression according to these indicators below. It is recommended to consult with someone."
    else:
        message = "Congratulations! You have low level of depression. Keep taking care of your mental health in this way."
        
    return render_template("result.html", pred_rounded=pred_rounded, message=message)


if __name__ == "__main__":
    app.run(debug=True)

