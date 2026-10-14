from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, json
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("bitfinder_model.joblib")   # <- match your model file name exactly
columns = json.load(open("columns.json"))

# Fill this in once you know which course each cluster means
model = joblib.load("model.joblib")
    # 0: "Web Development",
    # 1: "Data Analysis",
}

class Inquiry(BaseModel):
    exp: float
    budget: float
    age_range: str          # Under 18, 18-24, 25-34, 35+
    education_level: str    # SSCE, OND/HND, Undergraduate, Graduate
    current_status: str     # Student, Employed, Self-employed, Unemployed
    learning_style: str     # Theory, Practical, Mix of both
    main_goal: str          # Get a job, Freelance, Start a business, School project
    available_time: str     # Weekdays, Weekends, Flexible
    interest_areas: str     # Coding, Content/Video, Data, Design, Networking
    career_path: str        # Tech employee, Freelancer, Entrepeneur, Further studies

CATEGORY_FIELDS = [
    "age_range", "education_level", "current_status", "learning_style",
    "main_goal", "available_time", "interest_areas", "career_path",
]

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(data: Inquiry):
    row = {c: 0 for c in columns}
    row["exp"] = data.exp
    row["budget"] = data.budget

    for field in CATEGORY_FIELDS:
        col = f"{field}_{getattr(data, field)}"
        if col not in row:
            raise HTTPException(status_code=400, detail=f"Unknown value for {field}: {getattr(data, field)}")
        row[col] = 1

    df = pd.DataFrame([row], columns=columns)
    cluster = int(model.predict(df)[0])
    return {"cluster": cluster, "course": CLUSTER_TO_COURSE.get(cluster, "unmapped")}
