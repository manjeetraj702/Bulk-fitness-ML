from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import uvicorn
import os

app = FastAPI(title="Bulky Fitness Dynamic AI Prediction Engine")

# =========================================================
# CORS
# =========================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# GLOBALS
# =========================================================
model_accuracy_score = 0.0
feature_importance_vector = [0.0, 0.0, 0.0, 0.0]

# =========================================================
# HEALTH CHECK
# =========================================================
@app.get("/")
def health():
    return {
        "status": "running",
        "service": "Bulk Fitness ML Engine"
    }

# =========================================================
# TRAIN MODEL
# =========================================================
def train_fit_model():
    global model_accuracy_score, feature_importance_vector

    print("🔄 Training AI model...")

    np.random.seed(42)

    # REDUCED FOR RENDER FREE TIER
    n_samples = 800

    age = np.random.randint(16, 70, n_samples)
    weight = np.random.uniform(40, 140, n_samples)
    height = np.random.uniform(140, 210, n_samples)

    # 0 = cut
    # 1 = maintain
    # 2 = bulk
    goal_encoded = np.random.randint(0, 3, n_samples)

    base_bmr = (
        (10.0 * weight)
        + (6.25 * height)
        - (5.0 * age)
    )

    calories = np.where(
        goal_encoded == 0,
        base_bmr * 0.85,
        np.where(
            goal_encoded == 2,
            base_bmr * 1.20,
            base_bmr * 1.05
        )
    )

    calories += np.random.normal(0, 10, n_samples)

    protein = np.where(
        goal_encoded == 0,
        weight * 2.2,
        np.where(
            goal_encoded == 2,
            weight * 2.0,
            weight * 1.7
        )
    )

    fats = (calories * 0.25) / 9.0
    carbs = (
        calories
        - (protein * 4.0)
        - (fats * 9.0)
    ) / 4.0

    X = pd.DataFrame({
        "age": age,
        "weight": weight,
        "height": height,
        "goal": goal_encoded
    })

    y = pd.DataFrame({
        "calories": calories,
        "protein": protein,
        "carbs": carbs,
        "fats": fats
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # REDUCED MODEL SIZE
    model = RandomForestRegressor(
        n_estimators=30,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    model_accuracy_score = round(
        r2_score(y_test, predictions) * 100,
        2
    )

    feature_importance_vector = [
        round(num, 3)
        for num in model.feature_importances_
    ]

    print(f"✅ Model Accuracy: {model_accuracy_score}%")

    return model


# =========================================================
# STARTUP MODEL CACHE
# =========================================================
ml_model = train_fit_model()

# =========================================================
# REQUEST MODEL
# =========================================================
class PredictRequest(BaseModel):
    age: int
    weightKg: float
    heightCm: float
    optimizationGoal: str

# =========================================================
# GENERATE PLAN
# =========================================================
@app.post("/generate-plan")
def predict_metrics(req: PredictRequest):

    try:

        goal_mapping = {
            "cut": 0,
            "maintain": 1,
            "bulk": 2,
            "Cut": 0,
            "Maintain": 1,
            "Bulk": 2
        }

        goal_val = goal_mapping.get(
            req.optimizationGoal,
            1
        )

        features = pd.DataFrame([{
            "age": req.age,
            "weight": req.weightKg,
            "height": req.heightCm,
            "goal": goal_val
        }])

        pred = ml_model.predict(features)[0]

        return {
            "calories": int(pred[0]),
            "protein": int(pred[1]),
            "carbs": int(pred[2]),
            "fats": int(pred[3])
        }

    except Exception as e:

        print("❌ Prediction Error")
        print(str(e))

        return {
            "error": str(e)
        }

# =========================================================
# CALORIE COUNTER
# =========================================================
class CalorieCalculatorRequest(BaseModel):
    foodKey: str
    quantityAmount: float

INDIAN_FOOD_METRICS = {
    "paneer": {
        "protein": 18.0,
        "carbs": 1.2,
        "fats": 20.0,
        "calories": 257.0,
        "unit": "grams"
    },
    "rice": {
        "protein": 2.7,
        "carbs": 28.0,
        "fats": 0.3,
        "calories": 130.0,
        "unit": "grams"
    },
    "roti": {
        "protein": 3.5,
        "carbs": 18.0,
        "fats": 0.5,
        "calories": 85.0,
        "unit": "per piece"
    },
    "egg": {
        "protein": 6.0,
        "carbs": 0.6,
        "fats": 5.0,
        "calories": 78.0,
        "unit": "per whole egg"
    }
}

@app.post("/calculate-calories")
def calculate_calories(req: CalorieCalculatorRequest):

    key = req.foodKey.lower().strip()

    if key not in INDIAN_FOOD_METRICS:
        return {
            "error": "Food not found"
        }

    amount = req.quantityAmount

    base = INDIAN_FOOD_METRICS[key]

    scalar = (
        amount
        if base["unit"] in ["per piece", "per whole egg"]
        else amount / 100.0
    )

    return {
        "itemName": key.capitalize(),
        "servingLogged": f"{amount} {base['unit']}",
        "computedMetrics": {
            "calories": round(base["calories"] * scalar, 1),
            "protein": round(base["protein"] * scalar, 1),
            "carbs": round(base["carbs"] * scalar, 1),
            "fats": round(base["fats"] * scalar, 1)
        }
    }

# =========================================================
# DIAGNOSTICS
# =========================================================
@app.get("/model-diagnostics")
def diagnostics():
    return {
        "accuracy": model_accuracy_score,
        "importance": feature_importance_vector
    }

# =========================================================
# RENDER ENTRYPOINT
# =========================================================
if __name__ == "__main__":

    assigned_port = int(
        os.environ.get("PORT", 5001)
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=assigned_port
    )