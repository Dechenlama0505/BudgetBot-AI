from flask import Flask, request, jsonify
import os

import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoder
model = joblib.load("budget_overrun_model.pkl")
le = joblib.load("category_encoder.pkl")


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "BudgetBot AI service is running"})


@app.route("/categories", methods=["GET"])
def categories():
    return jsonify({"encoder_categories": list(le.classes_)})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        required_fields = [
            "day_of_month",
            "category",
            "spent_so_far",
            "transactions",
            "avg_daily_spend",
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        category_name = data["category"]
        day_of_month = int(data["day_of_month"])
        spent_so_far = float(data["spent_so_far"])
        transactions = int(data["transactions"])
        avg_daily_spend = float(data["avg_daily_spend"])

        encoded_category = le.transform([category_name])[0]

        sample = pd.DataFrame(
            [
                {
                    "day_of_month": day_of_month,
                    "category": encoded_category,
                    "spent_so_far": spent_so_far,
                    "transactions": transactions,
                    "avg_daily_spend": avg_daily_spend,
                }
            ]
        )

        raw_prediction = model.predict(sample)[0]

        # Keep ML prediction as the main result, but never allow
        # the final prediction to be lower than what is already spent.
        prediction = max(float(raw_prediction), spent_so_far)

        print(
            "[AI Forecast]",
            {
                "raw_prediction": round(float(raw_prediction), 2),
                "spent_so_far": round(float(spent_so_far), 2),
                "final_prediction": round(float(prediction), 2),
            },
        )

        return jsonify(
            {
                "category": category_name,
                "predictedFinalSpend": round(float(prediction), 2),
            }
        )

    except ValueError as e:
        return jsonify({"error": f"Value error: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5002))
    app.run(debug=False, host="0.0.0.0", port=port)