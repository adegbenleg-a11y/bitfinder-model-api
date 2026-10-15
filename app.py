from collections import Counter
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load the saved bundle (tfidf, km, cols, X, courses) ----
bundle = joblib.load("model.joblib")
tfidf = bundle["tfidf"]
km = bundle["km"]
cols = bundle["cols"]
X_train = bundle["X"]
train_courses = bundle["courses"]
train_labels = km.predict(X_train)

# ---- How the training features were built ----
TEXT_WEIGHT = 1.0                       # change only if the check cell says so
EXP_LEVELS = [0.0, 0.5, 1.0]            # exp_level 0, 1, 2
BUDGET_LEVELS = [0.0, 0.33, 0.66, 1.0]  # budget_level 0, 1, 2, 3
DOUBLE_WEIGHT_PREFIXES = ("interest_areas_", "career_path_")

SINGLE_FIELDS = [
    "age_range", "education_level", "current_status",
    "learning_style", "main_goal", "available_time",
]
MULTI_FIELDS = ["interest_areas", "career_path"]


class Inquiry(BaseModel):
    exp_level: int            # position of the chosen experience option, starting at 0
    budget_level: int         # position of the chosen budget option, starting at 0
    age_range: str            # 18-24, 25-34, 35+, Under 18
    education_level: str      # Graduate, OND/HND, SSCE, Undergraduate
    current_status: str       # Employed, Self-employed, Student, Unemployed
    learning_style: str       # Mix of both, Practical, Theory
    main_goal: str            # Freelance, Get a job, School project, Start a business
    available_time: str       # Flexible, Weekdays, Weekends
    interest_areas: List[str]  # Coding, Content/Video, Data, Design, Networking
    career_path: List[str]     # Entrepreneur, Freelancer, Further studies, Tech employee
    inquiry: str = ""         # the student's typed message (optional)


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/predict")
def predict(data: Inquiry):
    if not 0 <= data.exp_level < len(EXP_LEVELS):
        raise HTTPException(status_code=400, detail="exp_level must be 0 to 2")
    if not 0 <= data.budget_level < len(BUDGET_LEVELS):
        raise HTTPException(status_code=400, detail="budget_level must be 0 to 3")

    row = {c: 0.0 for c in cols}
    row["exp"] = EXP_LEVELS[data.exp_level]
    row["budget"] = BUDGET_LEVELS[data.budget_level]

    chosen = []
    for field in SINGLE_FIELDS:
        chosen.append((field, getattr(data, field)))
    for field in MULTI_FIELDS:
        for value in getattr(data, field):
            chosen.append((field, value))

    for field, value in chosen:
        col = f"{field}_{value}"
        if col not in row:
            raise HTTPException(status_code=400, detail=f"Unknown value for {field}: {value}")
        row[col] = 2.0 if col.startswith(DOUBLE_WEIGHT_PREFIXES) else 1.0

    survey = np.array([row[c] for c in cols], dtype=float).reshape(1, -1)
    text = tfidf.transform([data.inquiry]).toarray() * TEXT_WEIGHT
    x = np.hstack([survey, text])

    cluster = int(km.predict(x)[0])

    # Most common courses among the training students in this cluster
    in_cluster = [c for c, label in zip(train_courses, train_labels) if label == cluster]
    ranked = [name for name, _ in Counter(in_cluster).most_common()]

    # If the cluster has fewer than 2 courses, fill from the nearest students
    distances = np.linalg.norm(X_train - x, axis=1)
    for i in np.argsort(distances):
        if len(ranked) >= 2:
            break
        name = train_courses[i]
        if name not in ranked:
            ranked.append(name)

    courses = ranked[:2]
    return {
        "cluster": cluster,
        "courses": courses,
        "course": " or ".join(courses),
    }
