"""
application.py — Flask web app for Project 3: Car Price Prediction.

Serves a form where a user:
  1. picks a Brand -> the Model dropdown fills in with that brand's listings
  2. picks a specific Model/trim -> engine, power, mileage, seats, fuel,
     and transmission auto-fill from the dataset (a person shopping for a
     used car wouldn't know their exact engine CC off-hand, so the app
     looks it up instead of asking)
  3. fills in the variable details that actually change listing to
     listing: year of purchase, kilometers driven, ownership history,
     seller type
  4. clicks Predict -> gets a price estimate from the Random Forest model
     (R2 ~0.93 on the held-out test set, confirmed stable with 5-fold CV)

Run:
    pip install -r requirements.txt
    python application.py
Then open http://127.0.0.1:5000
"""
import pickle
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

application = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app = application  # alias for WSGI hosts that look for `application`

CURRENT_YEAR = 2026

# ---------- load trained model + form data once at startup ----------
with open(os.path.join(BASE_DIR, "model", "car_price_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "model", "form_options.pkl"), "rb") as f:
    form_options = pickle.load(f)

COMPANIES = form_options["companies"]
NAMES_BY_COMPANY = form_options["names_by_company"]
SPECS_BY_NAME = form_options["specs_by_name"]
FUEL_TYPES = form_options["fuel_types"]
SELLER_TYPES = form_options["seller_types"]
TRANSMISSIONS = form_options["transmissions"]
OWNERS = form_options["owners"]
YEARS = form_options["years"]

_KNOWN_BRAND_CATEGORIES = set(
    model.named_steps["preprocess"].transformers_[0][1].categories_[0]
)


def group_brand(brand: str) -> str:
    """Mirror the 'rare brand -> Other' grouping used at training time."""
    return brand if brand in _KNOWN_BRAND_CATEGORIES else "Other"


@application.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        companies=COMPANIES,
        names_by_company=NAMES_BY_COMPANY,
        specs_by_name=SPECS_BY_NAME,
        fuel_types=FUEL_TYPES,
        seller_types=SELLER_TYPES,
        transmissions=TRANSMISSIONS,
        owners=OWNERS,
        years=YEARS,
    )


@application.route("/specs", methods=["GET"])
def specs():
    """Return the known specs for a specific car name, to auto-fill the form."""
    name = request.args.get("name", "")
    spec = SPECS_BY_NAME.get(name)
    if spec is None:
        return jsonify({"success": False, "error": "Unknown model"}), 404
    return jsonify({"success": True, **spec})


@application.route("/predict", methods=["POST"])
def predict():
    try:
        brand = request.form.get("company")
        year = int(request.form.get("year"))
        km_driven = float(request.form.get("km_driven"))
        fuel = request.form.get("fuel")
        seller_type = request.form.get("seller_type")
        transmission = request.form.get("transmission")
        owner = request.form.get("owner")
        mileage = float(request.form.get("mileage"))
        engine = float(request.form.get("engine"))
        max_power = float(request.form.get("max_power"))
        seats = float(request.form.get("seats"))

        car_age = CURRENT_YEAR - year

        row = pd.DataFrame([{
            "brand_grouped": group_brand(brand),
            "fuel": fuel,
            "seller_type": seller_type,
            "transmission": transmission,
            "owner": owner,
            "car_age": car_age,
            "km_driven": km_driven,
            "mileage_num": mileage,
            "engine_num": engine,
            "max_power_num": max_power,
            "seats": seats,
        }])

        predicted_price = model.predict(row)[0]
        predicted_price = max(0, round(predicted_price))

        return jsonify({
            "success": True,
            "predicted_price": f"₹{predicted_price:,.0f}",
            "predicted_price_raw": predicted_price,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


if __name__ == "__main__":
    application.run(debug=True)