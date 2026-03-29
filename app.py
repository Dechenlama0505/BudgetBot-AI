# from flask import Flask, request, jsonify
# import pandas as pd
# import joblib

# app = Flask(__name__)

# # Load model and encoder
# model = joblib.load("budget_overrun_model.pkl")
# le = joblib.load("category_encoder.pkl")


# @app.route("/predict", methods=["POST"])
# def predict():
#     try:
#         data = request.get_json()

#         if not data:
#             return jsonify({"error": "No JSON data received"}), 400

#         required_fields = [
#             "day_of_month",
#             "category",
#             "spent_so_far",
#             "transactions",
#             "avg_daily_spend"
#         ]

#         for field in required_fields:
#             if field not in data:
#                 return jsonify({"error": f"Missing field: {field}"}), 400

#         category_name = data["category"]

#         # Encode category
#         encoded_category = le.transform([category_name])[0]

#         # Create input in exact training feature order
#         sample = pd.DataFrame([{
#             "day_of_month": data["day_of_month"],
#             "category": encoded_category,
#             "spent_so_far": data["spent_so_far"],
#             "transactions": data["transactions"],
#             "avg_daily_spend": data["avg_daily_spend"]
#         }])

#         # Predict final spending
#         prediction = model.predict(sample)[0]

#         return jsonify({
#             "category": category_name,
#             "predicted_final_spend": round(float(prediction), 2)
#         })

#     except ValueError as e:
#         return jsonify({"error": f"Value error: {str(e)}"}), 400
#     except Exception as e:
#         return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# @app.route("/", methods=["GET"])
# def home():
#     return jsonify({
#         "message": "BudgetBot AI service is running"
#     })


# if __name__ == "__main__":
#     app.run(debug=True, port=5002)





from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoder
model = joblib.load("budget_overrun_model.pkl")
le = joblib.load("category_encoder.pkl")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "BudgetBot AI service is running"
    })


@app.route("/categories", methods=["GET"])
def categories():
    return jsonify({
        "encoder_categories": list(le.classes_)
    })


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
            "avg_daily_spend"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        category_name = data["category"]

        encoded_category = le.transform([category_name])[0]

        sample = pd.DataFrame([{
            "day_of_month": data["day_of_month"],
            "category": encoded_category,
            "spent_so_far": data["spent_so_far"],
            "transactions": data["transactions"],
            "avg_daily_spend": data["avg_daily_spend"]
        }])

        prediction = model.predict(sample)[0]

        return jsonify({
            "category": category_name,
            "predicted_final_spend": round(float(prediction), 2)
        })

    except ValueError as e:
        return jsonify({"error": f"Value error: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5002)