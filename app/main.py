from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.recommender import predict_next_categories   # <-- FIXED

app = FastAPI(title="GRU4Rec Recommendation API")


class PredictRequest(BaseModel):
    product_ids: List[str]
    category_ids: List[str]
    brand_ids: List[str]
    event_types: List[str]


@app.get("/")
def home():
    return {"status": "running", "message": "GRU4Rec API Alive"}


@app.post("/predict")
def predict(req: PredictRequest):
    predictions = predict_next_categories(
        req.product_ids,
        req.category_ids,
        req.brand_ids,
        req.event_types
    )
    return {"recommended_categories": predictions}
