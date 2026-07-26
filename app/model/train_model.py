"""
Trains the best-performing model (Random Forest on the enhanced feature
set) and pickles it for application.py to load. Also builds a
name -> specs lookup table so the web form can auto-fill engine/power/
mileage/seats/fuel/transmission when a user picks a specific car listing,
instead of asking them to type in numbers they wouldn't know off-hand.
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

CURRENT_YEAR = 2026

# ---------- load + clean (same as the enhanced notebook) ----------
df = pd.read_csv("../../Dataset/Car_details_v3.csv")

df = df.dropna(subset=['mileage', 'engine', 'max_power', 'seats']).copy()
df['mileage_num'] = df['mileage'].str.extract(r'([\d.]+)').astype(float)
df['engine_num'] = df['engine'].str.extract(r'([\d.]+)').astype(float)
df['max_power_num'] = pd.to_numeric(df['max_power'].str.extract(r'([\d.]+)')[0], errors='coerce')
df = df.dropna(subset=['max_power_num'])

df['brand'] = df['name'].str.split().str[0]
df = df.drop_duplicates().reset_index(drop=True)
df['car_age'] = CURRENT_YEAR - df['year']

brand_counts = df['brand'].value_counts()
rare_brands = brand_counts[brand_counts < 20].index
df['brand_grouped'] = df['brand'].apply(lambda x: 'Other' if x in rare_brands else x)

# ---------- build pipeline ----------
cat_cols = ['brand_grouped', 'fuel', 'seller_type', 'transmission', 'owner']
num_cols = ['car_age', 'km_driven', 'mileage_num', 'engine_num', 'max_power_num', 'seats']

X = df[cat_cols + num_cols]
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
    remainder='passthrough'
)

pipe = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', RandomForestRegressor(n_estimators=300, max_depth=14, random_state=42))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"R2={r2:.4f}  MAE={mae:,.0f}  RMSE={rmse:,.0f}")

with open("car_price_model.pkl", "wb") as f:
    pickle.dump(pipe, f)

# ---------- dropdown + specs lookup for the web form ----------
companies = sorted(df['brand'].unique().tolist())

names_by_company = (
    df.groupby('brand')['name']
    .apply(lambda s: sorted(s.unique().tolist()))
    .to_dict()
)

# one representative spec row per unique car name (most common values)
specs_by_name = {}
for name, group in df.groupby('name'):
    row = group.iloc[0]
    specs_by_name[name] = {
        "fuel": row['fuel'],
        "transmission": row['transmission'],
        "mileage": round(float(group['mileage_num'].median()), 1),
        "engine": round(float(group['engine_num'].median())),
        "max_power": round(float(group['max_power_num'].median()), 1),
        "seats": int(group['seats'].median()),
    }

fuel_types = sorted(df['fuel'].unique().tolist())
seller_types = sorted(df['seller_type'].unique().tolist())
transmissions = sorted(df['transmission'].unique().tolist())
owners = sorted(df['owner'].unique().tolist())
years = sorted(df['year'].unique().tolist(), reverse=True)

with open("form_options.pkl", "wb") as f:
    pickle.dump({
        "companies": companies,
        "names_by_company": names_by_company,
        "specs_by_name": specs_by_name,
        "fuel_types": fuel_types,
        "seller_types": seller_types,
        "transmissions": transmissions,
        "owners": owners,
        "years": years,
    }, f)

print("Saved car_price_model.pkl and form_options.pkl")
print("Companies:", len(companies), " Unique models:", len(specs_by_name))
