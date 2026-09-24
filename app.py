from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)

MODEL_ACCURACY = 0.9813

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    probability = None
    form = {}

    if request.method == "POST":
        try:
            form = request.form.to_dict()

            data = pd.DataFrame([{
                "no_of_dependents": int(form["no_of_dependents"]),
                "education": form["education"],
                "self_employed": form["self_employed"],
                "income_annum": float(form["income_annum"]),
                "loan_amount": float(form["loan_amount"]),
                "loan_term": int(form["loan_term"]),
                "cibil_score": int(form["cibil_score"])
            }])

            prediction = model.predict(data)[0]
            probabilities = model.predict_proba(data)[0]
            classes = list(model.classes_)
            probability = round(
                float(probabilities[classes.index(prediction)]) * 100, 2
            )

            result = prediction

        except Exception as exc:
            result = f"Error: {exc}"

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        form=form,
        accuracy=round(MODEL_ACCURACY * 100, 2)
    )

if __name__ == "__main__":
    app.run(debug=True)
