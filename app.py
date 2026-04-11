from flask import Flask, request, jsonify
import calendar
from datetime import datetime

import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoder
model = joblib.load("budget_overrun_model.pkl")
le = joblib.load("category_encoder.pkl")


def get_days_in_current_month():
    now = datetime.now()
    return calendar.monthrange(now.year, now.month)[1]


def build_pace_projection(day_of_month, spent_so_far, avg_daily_spend, weight=0.4):
    days_in_month = get_days_in_current_month()
    capped_day_of_month = min(max(int(day_of_month), 1), days_in_month)
    remaining_days = max(days_in_month - capped_day_of_month, 0)

    # Pacing support estimate for remaining month activity.
    pace_projection = spent_so_far + (avg_daily_spend * remaining_days * weight)
    return pace_projection


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

        # ML model is the primary prediction engine.
        raw_prediction = model.predict(sample)[0]

        # Pacing logic is only a realism guardrail and does not replace model output.
        pace_projection = build_pace_projection(
            day_of_month=day_of_month,
            spent_so_far=spent_so_far,
            avg_daily_spend=avg_daily_spend,
            weight=0.4,
        )

        # Controlled lower support: helps avoid flat/illogical month-end forecasts
        # while preserving the model's estimate as the main signal.
        support_floor = spent_so_far + max(0.0, (pace_projection - spent_so_far) * 0.6)

        prediction = max(float(raw_prediction), float(spent_so_far), float(support_floor))

        # Conservative upper guardrail to avoid unrealistic jumps.
        max_guardrail = max(spent_so_far, pace_projection * 1.35)
        prediction = min(prediction, max_guardrail)

        print(
            "[AI Forecast]",
            {
                "raw_prediction": round(float(raw_prediction), 2),
                "spent_so_far": round(float(spent_so_far), 2),
                "pace_projection": round(float(pace_projection), 2),
                "final_prediction": round(float(prediction), 2),
            },
        )

        return jsonify({
            "category": category_name,
            "predictedFinalSpend": round(float(prediction), 2)
        })

    except ValueError as e:
        return jsonify({"error": f"Value error: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5002)
