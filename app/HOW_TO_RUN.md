# Running the Car Price Predictor App

This app serves the **Random Forest model (R² ≈ 0.93)** trained in
`Notebook/car_price_prediction_enhanced.ipynb` — the strongest of the two
models in this project.

## 1. Install dependencies
```bash
cd app
pip install -r requirements.txt
```

## 2. (Optional) Retrain the model
A pre-trained model is already included at `model/car_price_model.pkl`.
To retrain it from scratch:
```bash
cd model
python train_model.py
cd ..
```

## 3. Run the app
```bash
python application.py
```
Open **http://127.0.0.1:5000** in your browser.

## How it works
1. Pick a **Brand** — the **Model/Trim** dropdown fills in with that
   brand's actual listings from the dataset.
2. Pick a specific **Model/Trim** — fuel type, transmission, mileage,
   engine size, max power, and seats auto-fill from the dataset. (A buyer
   shopping for a used car wouldn't know the exact engine CC off-hand, so
   the app looks it up instead of asking.)
3. Fill in the details that actually vary listing-to-listing: **year of
   purchase**, **kilometers driven**, **ownership history**, **seller
   type**.
4. Click **Predict Price** — sends a POST to `/predict`, which runs the
   same Random Forest pipeline from the enhanced notebook and returns a
   price estimate.

Brands not seen often enough during training fall back to an `"Other"`
bucket (same logic as the notebook) instead of erroring out.

## Routes
| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Renders the form |
| `/specs?name=<car name>` | GET | Returns that model's specs as JSON (used by the form's auto-fill) |
| `/predict` | POST | Returns a predicted price as JSON |

## Folder structure
```
app/
├── application.py          # Flask app (routes: /, /specs, /predict)
├── requirements.txt
├── model/
│   ├── train_model.py      # trains + pickles the Random Forest pipeline
│   ├── car_price_model.pkl
│   └── form_options.pkl    # brand/model lists + specs lookup table
├── templates/
│   └── index.html
└── static/
    └── style.css
```
