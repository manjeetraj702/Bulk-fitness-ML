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

# Enable CORS for clean cross-service backend routing handshakes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global nodes to cache mathematical metrics for your diagnostics dashboard
model_accuracy_score = 0.0
feature_importance_vector = [0.0, 0.0, 0.0, 0.0]

# -------------------------------------------------------------
# 1. BIOLOGICALLY SENSITIVE ML MODEL TRAINING PIPELINE
# -------------------------------------------------------------
def train_fit_model():
    global model_accuracy_score, feature_importance_vector
    print("🔄 Training highly sensitive Random Forest Regressor Matrix...")
    
    np.random.seed(42)
    n_samples = 3000  # Increased density for smoother continuous feature splits
    
    # Generate human biological feature distributions
    age = np.random.randint(16, 70, n_samples)
    weight = np.random.uniform(40, 140, n_samples)
    height = np.random.uniform(140, 210, n_samples)
    goal_encoded = np.random.randint(0, 3, n_samples)  # 0: Cut, 1: Maintain, 2: Bulk
    
    # Core Basal Metabolic Rate (BMR) Calculation Matrix
    # Derived from established exercise science equations where weight and height act as primary drivers
    base_bmr = (10.0 * weight) + (6.25 * height) - (5.0 * age)
    
    # Apply Goal Modifiers as Dynamic Scalars instead of static flat numbers
    # Cut = -15% Deficit, Maintain = Balanced TDEE baseline, Bulk = +20% Surplus
    calories = np.where(goal_encoded == 0, base_bmr * 0.85, 
               np.where(goal_encoded == 2, base_bmr * 1.20, base_bmr * 1.05))
    
    # Inject minimal background variance noise (reduced from 50 to 10 to protect feature weights)
    calories += np.random.normal(0, 10, n_samples)

    # Calculate Macronutrient Metrics directly proportional to Body Weight
    protein = np.where(goal_encoded == 0, weight * 2.2,
              np.where(goal_encoded == 2, weight * 2.0, weight * 1.7))
    
    fats = (calories * 0.25) / 9.0
    carbs = (calories - (protein * 4.0) - (fats * 9.0)) / 4.0

    X = pd.DataFrame({'age': age, 'weight': weight, 'height': height, 'goal': goal_encoded})
    y = pd.DataFrame({'calories': calories, 'protein': protein, 'carbs': carbs, 'fats': fats})
    
    # Split dataset for validation metrics auditing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    # Verify performance integrity
    predictions = model.predict(X_test)
    model_accuracy_score = round(r2_score(y_test, predictions) * 100, 2)
    feature_importance_vector = [round(num, 3) for num in model.feature_importances_]
    
    print(f"✅ AI Node Ready! Model R² Accuracy Score: {model_accuracy_score}%")
    print(f"📊 Feature Split Weights [Age, Weight, Height, Goal]: {feature_importance_vector}")
    return model

# Train and cache the model context on application startup
ml_model = train_fit_model()


# -------------------------------------------------------------
# 2. DATA TRANSFER SCHEMAS (Pydantic Models)
# -------------------------------------------------------------
class PredictRequest(BaseModel):
    age: int
    weightKg: float
    heightCm: float
    optimizationGoal: str  # Matches Java's UserProfileDocument field exactly


# -------------------------------------------------------------
# 3. ENDPOINT 1: DYNAMIC INFERENCE PROCESSING HANDSHAKE
# -------------------------------------------------------------
@app.post("/generate-plan")
def predict_metrics(req: PredictRequest):
    # Map the raw string goal coming from Java to the numerical encoding model expects
    goal_mapping = {
        "cut": 0, "maintain": 1, "bulk": 2,
        "Cut": 0, "Maintain": 1, "Bulk": 2
    }
    goal_val = goal_mapping.get(req.optimizationGoal, 1)

    # Match the exact column sorting layout schema context used during training phase
    features = pd.DataFrame([{
        'age': req.age,
        'weight': req.weightKg,
        'height': req.heightCm,
        'goal': goal_val
    }])
    
    # Execute prediction matrix inference
    pred = ml_model.predict(features)[0]
    
    return {
        "calories": int(pred[0]),
        "protein": int(pred[1]),
        "carbs": int(pred[2]),
        "fats": int(pred[3])
    }


# -------------------------------------------------------------
# 4. ENDPOINT 2: INDIAN CALORIE COUNTER CONTROLLER
# -------------------------------------------------------------
class CalorieCalculatorRequest(BaseModel):
    foodKey: str
    quantityAmount: float

INDIAN_FOOD_METRICS = {
    "paneer": {"protein": 18.0, "carbs": 1.2, "fats": 20.0, "calories": 257.0, "unit": "grams"},
    "rice": {"protein": 2.7, "carbs": 28.0, "fats": 0.3, "calories": 130.0, "unit": "grams"},
    "roti": {"protein": 3.5, "carbs": 18.0, "fats": 0.5, "calories": 85.0, "unit": "per piece"},
    "egg": {"protein": 6.0, "carbs": 0.6, "fats": 5.0, "calories": 78.0, "unit": "per whole egg"},
    "soybean": {"protein": 52.0, "carbs": 33.0, "fats": 0.5, "calories": 345.0, "unit": "grams (raw chunks)"},
    "oats": {"protein": 13.0, "carbs": 66.0, "fats": 6.5, "calories": 389.0, "unit": "grams"}
}

@app.post("/calculate-calories")
def calculate_calories(req: CalorieCalculatorRequest):
    key = req.foodKey.lower().strip()
    amount = req.quantityAmount

    if key not in INDIAN_FOOD_METRICS:
        return {"error": "Item not indexed in system portfolio."}

    base_stats = INDIAN_FOOD_METRICS[key]
    scalar = amount if base_stats["unit"] in ["per piece", "per whole egg"] else (amount / 100.0)

    return {
        "itemName": key.capitalize(),
        "servingLogged": f"{amount} {base_stats['unit']}",
        "computedMetrics": {
            "calories": round(base_stats["calories"] * scalar, 1),
            "protein": round(base_stats["protein"] * scalar, 1),
            "carbs": round(base_stats["carbs"] * scalar, 1),
            "fats": round(base_stats["fats"] * scalar, 1)
        },
        "calculationEngine": "Linear Macro Scalar Coefficient Matrix"
    }


# -------------------------------------------------------------
# 5. ENDPOINT 3: LIVE METRICS AUDITING FOR DIAGNOSTICS TAB
# -------------------------------------------------------------
@app.get("/model-diagnostics")
def get_diagnostics():
    return {
        "r2AccuracyScore": model_accuracy_score,
        "features": ["Age Factor", "Body Weight", "Height Spectrum", "Fitness Goal"],
        "importanceWeights": feature_importance_vector
    }


# -------------------------------------------------------------
# 6. SERVER RUNTIME ENTRYPOINT (OPTIMIZED FOR CLOUD PORT BINDING)
# -------------------------------------------------------------
if __name__ == "__main__":
    # 🎯 Render injects an environment variable named 'PORT' at startup.
    # This block intercepts it dynamically, falling back to 5001 if testing locally.
    assigned_port = int(os.environ.get("PORT", 5001))
    uvicorn.run(app, host="0.0.0.0", port=assigned_port) 