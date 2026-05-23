from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import os
import uvicorn
import asyncio
import httpx
from contextlib import asynccontextmanager

# ============================================================
# 💓 LIGHTWEIGHT NON-BLOCKING KEEP-ALIVE HEARBEAT THREAD
# ============================================================
async def keep_alive_scheduler():
    """Defensive standalone thread loop that keeps the Render microservice awake."""
    # 🎯 FIX: Wait a safe 30 seconds for Uvicorn to fully bind the public networking ports first!
    await asyncio.sleep(30)
    
    async with httpx.AsyncClient() as client:
        while True:
            try:
                assigned_port = os.environ.get("PORT", "5001")
                # Loopback check directly against local machine port context arrays
                target_url = f"http://127.0.0.1:{assigned_port}/"
                response = await client.get(target_url, timeout=5.0)
                if response.status_code == 200:
                    print("💓 ML Keep-Alive Heartbeat: Container Warmth Verified.")
            except Exception as e:
                print(f"⚠️ Keep-Alive heartbeat skipped: {str(e)}")
            
            # Sleep the worker thread for exactly 11 minutes (660 seconds)
            await asyncio.sleep(660)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 🎯 FIX: Use create_task to launch the loop completely decoupled from the main thread stack
    loop_task = asyncio.create_task(keep_alive_scheduler())
    yield
    # Cleanup task references gracefully on platform shutdown cycles
    loop_task.cancel()

# Initialize the engine utilizing the robust non-blocking lifespan framework
app = FastAPI(title="Bulky Fitness Dynamic AI Prediction Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Telemetry Trackers
model_accuracy_score = 0.0
feature_importance_vector = [0.0, 0.0, 0.0, 0.0]

INDIAN_FOOD_METRICS = {
    "paneer": {"protein": 18.0, "carbs": 1.2, "fats": 20.0, "calories": 257.0, "unit": "grams"},
    "rice": {"protein": 2.7, "carbs": 28.0, "fats": 0.3, "calories": 130.0, "unit": "grams"},
    "roti": {"protein": 3.5, "carbs": 18.0, "fats": 0.5, "calories": 85.0, "unit": "per piece"},
    "egg": {"protein": 6.0, "carbs": 0.6, "fats": 5.0, "calories": 78.0, "unit": "per whole egg"},
    "soybean": {"protein": 52.0, "carbs": 33.0, "fats": 0.5, "calories": 345.0, "unit": "grams"},
    "oats": {"protein": 16.9, "carbs": 66.3, "fats": 6.9, "calories": 389.0, "unit": "grams"}
}

def train_fit_model():
    global model_accuracy_score, feature_importance_vector
    print("🔄 Training AI Model Matrix...")
    np.random.seed(42)
    n_samples = 800

    age = np.random.randint(16, 70, n_samples)
    weight = np.random.uniform(40, 140, n_samples)
    height = np.random.uniform(140, 210, n_samples)
    goal_encoded = np.random.randint(0, 3, n_samples)

    base_bmr = (10.0 * weight) + (6.25 * height) - (5.0 * age)
    calories = np.where(goal_encoded == 0, base_bmr * 0.85, np.where(goal_encoded == 2, base_bmr * 1.20, base_bmr * 1.05))
    calories += np.random.normal(0, 10, n_samples)
    protein = np.where(goal_encoded == 0, weight * 2.2, np.where(goal_encoded == 2, weight * 2.0, weight * 1.7))
    fats = (calories * 0.25) / 9.0
    carbs = (calories - (protein * 4.0) - (fats * 9.0)) / 4.0

    X = pd.DataFrame({"age": age, "weight": weight, "height": height, "goal": goal_encoded})
    y = pd.DataFrame({"calories": calories, "protein": protein, "carbs": carbs, "fats": fats})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=30, max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    
    model_accuracy_score = round(r2_score(y_test, model.predict(X_test)) * 100, 2)
    feature_importance_vector = [round(num, 3) for num in model.feature_importances_]
    print(f"✅ Model Setup Complete. Accuracy Profile: {model_accuracy_score}%")
    return model

ml_model = train_fit_model()

# Lightweight root checkpoint target endpoint for self-pings
@app.get("/")
def health_checkpoint():
    return {"status": "warm", "service": "Bulky Fitness ML Engine Node"}

class PredictRequest(BaseModel):
    age: int
    weightKg: float
    heightCm: float
    optimizationGoal: str

@app.post("/generate-plan")
def predict_metrics(req: PredictRequest):
    try:
        raw_goal = req.optimizationGoal.lower()
        goal_val = 0 if "cut" in raw_goal else (2 if "bulk" in raw_goal else 1)

        features = pd.DataFrame([{
            "age": int(req.age),
            "weight": float(req.weightKg),
            "height": float(req.heightCm),
            "goal": int(goal_val)
        }])

        pred = ml_model.predict(features)[0]

        return {
            "calories": int(pred[0]),
            "protein": int(pred[1]),
            "carbs": int(pred[2]),
            "fats": int(pred[3])
        }
    except Exception as e:
        print(f"❌ Prediction Pipeline Interception Error: {str(e)}")
        return {"error": str(e)}

class CalorieCalculatorRequest(BaseModel):
    foodKey: str
    quantityAmount: float

@app.post("/calculate-calories")
def calculate_calories(req: CalorieCalculatorRequest):
    key = req.foodKey.lower().strip()
    if key not in INDIAN_FOOD_METRICS:
        return {"error": "Food registry signature missing."}
    
    try:
        amount = float(req.quantityAmount)
    except (ValueError, TypeError):
        return {"error": "Invalid density configuration value."}

    base = INDIAN_FOOD_METRICS[key]
    scalar = amount if base["unit"] in ["per piece", "per whole egg"] else amount / 100.0

    return {
        "itemName": key.capitalize(),
        "servingLogged": f"{amount} {base['unit']}",
        "computedMetrics": {
            "calories": round(float(base["calories"]) * scalar, 1),
            "protein": round(float(base["protein"]) * scalar, 1),
            "carbs": round(float(base["carbs"]) * scalar, 1),
            "fats": round(float(base["fats"]) * scalar, 1)
        }
    }

@app.get("/model-diagnostics")
def diagnostics():
    return {"accuracy": model_accuracy_score, "importance": feature_importance_vector}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))