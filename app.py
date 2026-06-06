from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import os
import uvicorn
import asyncio
import httpx
from contextlib import asynccontextmanager

# ============================================================
# KEEP ALIVE
# ============================================================

async def keep_alive_scheduler():
    await asyncio.sleep(30)

    async with httpx.AsyncClient() as client:
        while True:
            try:
                assigned_port = os.environ.get("PORT", "5001")
                target_url = f"http://127.0.0.1:{assigned_port}/"

                response = await client.get(
                    target_url,
                    timeout=5.0
                )

                if response.status_code == 200:
                    print("💓 Keep Alive Success")

            except Exception as e:
                print(f"⚠️ Keep Alive Error: {e}")

            await asyncio.sleep(660)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(keep_alive_scheduler())
    yield
    task.cancel()


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Bulky Fitness AI Engine",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GLOBALS
# ============================================================

model_accuracy_score = 0.0
feature_importance_vector = []
INDIAN_FOOD_METRICS = {}

# ============================================================
# LOAD FOOD DATA
# ============================================================

def load_food_database():
    global INDIAN_FOOD_METRICS

    try:
        food_df = pd.read_csv("food_data.csv")

        INDIAN_FOOD_METRICS = {
            str(row["food"]).lower().strip(): {
                "protein": float(row["protein"]),
                "carbs": float(row["carbs"]),
                "fats": float(row["fats"]),
                "calories": float(row["calories"]),
                "unit": str(row["unit"])
            }
            for _, row in food_df.iterrows()
        }

        print(
            f"✅ Loaded {len(INDIAN_FOOD_METRICS)} foods"
        )

    except Exception as e:
        print(f"❌ Food Data Error: {e}")
        INDIAN_FOOD_METRICS = {}


# ============================================================
# TRAIN MODEL FROM CSV
# ============================================================

def train_fit_model():
    global model_accuracy_score
    global feature_importance_vector

    try:
        print("🔄 Loading fitness dataset...")

        df = pd.read_csv("fitness_data.csv")

        required_columns = [
            "age",
            "weight",
            "height",
            "goal",
            "calories",
            "protein",
            "carbs",
            "fats"
        ]

        for col in required_columns:
            if col not in df.columns:
                raise Exception(
                    f"Missing column: {col}"
                )

        X = df[
            [
                "age",
                "weight",
                "height",
                "goal"
            ]
        ]

        y = df[
            [
                "calories",
                "protein",
                "carbs",
                "fats"
            ]
        ]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        model_accuracy_score = round(
            r2_score(y_test, predictions) * 100,
            2
        )

        feature_importance_vector = [
            round(x, 4)
            for x in model.feature_importances_
        ]

        print(
            f"✅ Model Accuracy: {model_accuracy_score}%"
        )

        return model

    except Exception as e:
        print(f"❌ Model Training Error: {e}")

        return RandomForestRegressor()


# ============================================================
# STARTUP
# ============================================================

load_food_database()
ml_model = train_fit_model()

# ============================================================
# REQUEST MODELS
# ============================================================

class PredictRequest(BaseModel):
    age: int
    weightKg: float
    heightCm: float
    optimizationGoal: str


class CalorieCalculatorRequest(BaseModel):
    foodKey: str
    quantityAmount: float


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def home():
    return {
        "status": "running",
        "service": "Bulky Fitness AI Engine"
    }


@app.post("/generate-plan")
def predict_metrics(req: PredictRequest):
    try:

        goal = req.optimizationGoal.lower()

        if "cut" in goal:
            goal_val = 0
        elif "bulk" in goal:
            goal_val = 2
        else:
            goal_val = 1

        features = pd.DataFrame([
            {
                "age": req.age,
                "weight": req.weightKg,
                "height": req.heightCm,
                "goal": goal_val
            }
        ])

        prediction = ml_model.predict(features)[0]

        return {
            "calories": int(prediction[0]),
            "protein": int(prediction[1]),
            "carbs": int(prediction[2]),
            "fats": int(prediction[3])
        }

    except Exception as e:
        return {
            "error": str(e)
        }


@app.post("/calculate-calories")
def calculate_calories(
    req: CalorieCalculatorRequest
):
    try:

        food_name = (
            req.foodKey
            .lower()
            .strip()
        )

        if food_name not in INDIAN_FOOD_METRICS:
            return {
                "error": "Food not found"
            }

        food = INDIAN_FOOD_METRICS[food_name]

        amount = float(req.quantityAmount)

        if food["unit"] in [
            "per piece",
            "per whole egg"
        ]:
            multiplier = amount
        else:
            multiplier = amount / 100.0

        return {
            "itemName": food_name.title(),
            "servingLogged": f"{amount} {food['unit']}",
            "computedMetrics": {
                "calories": round(
                    food["calories"] * multiplier,
                    1
                ),
                "protein": round(
                    food["protein"] * multiplier,
                    1
                ),
                "carbs": round(
                    food["carbs"] * multiplier,
                    1
                ),
                "fats": round(
                    food["fats"] * multiplier,
                    1
                )
            }
        }

    except Exception as e:
        return {
            "error": str(e)
        }


@app.get("/foods")
def get_foods():
    return {
        "count": len(
            INDIAN_FOOD_METRICS
        ),
        "foods": list(
            INDIAN_FOOD_METRICS.keys()
        )
    }


@app.get("/model-diagnostics")
def diagnostics():
    return {
        "accuracy": model_accuracy_score,
        "feature_importance":
            feature_importance_vector
    }


@app.get("/reload-data")
def reload_data():
    global ml_model

    load_food_database()
    ml_model = train_fit_model()

    return {
        "message":
            "Data reloaded successfully"
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5001
            )
        )
    )