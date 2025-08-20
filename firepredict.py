from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import random
# import pickle
import joblib
import numpy as np
from shapely.geometry import shape, Polygon

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Try to load model if available
try:
    with open(r"py_files\best_random_forest_model.pkl", "rb") as f:
        model = joblib.load(f)
        print("model succ loaded")
except:
    model = None
    print("Warning: Model not found. Using dummy confidences.")

# ========== Request & Response Schemas ==========
class Geometry(BaseModel):
    type: str
    coordinates: List[List[List[float]]]

class FireMapRequest(BaseModel):
    geometry: Geometry
    zoomLevel: int
    bbox: List[float]

class FireTile(BaseModel):
    lat: float
    lon: float
    conf: float
    size: float

class FireMapResponse(BaseModel):
    results: List[FireTile]
# python -m uvicorn firepredict:app --reload
# uvicorn py_files.firepredict:app --reload

# ========== Endpoint ==========
@app.post("/predict_fire_map", response_model=FireMapResponse)

class PointRequest(BaseModel):
    lat: float
    lon: float
    month: int

@app.post("/predict_point")
def predict_point(req: PointRequest):
    # Load extracted environmental features from CSV
    import pandas as pd
    try:
        df = pd.read_csv("point_data_output.csv")
    except:
        return {"error": "CSV file with environmental data not found."}

    # Model expects same order of columns
    feature_order = [
        "elevation", "lat", "lon", "ndvi", "lst", "slope", "aspect",
        "land_cover_type", "month", "relative_humidity", "wind_speed"
    ]
    
    if model is not None:
        proba = model.predict_proba(df[feature_order])[0][1]
    else:
        print("using fake inputs")
        proba = round(random.uniform(0, 1), 2)

    # Assign label based on proba ranges
    if proba < 0.2:
        label = "🟢 Very low chances"
    elif proba < 0.4:
        label = "🟡 Low chances"
    elif proba < 0.6:
        label = "🟣 Moderate chances"
    elif proba < 0.8:
        label = "🟠 Elevated risk"
    else:
        label = "🔴 Possible fire hazard"


    return {
        "label": label,
        "confidence": round(proba, 2)
    }

def predict_fire_map(req: FireMapRequest):
    polygon = shape(req.geometry.dict())

    # Approx area in square meters
    area_m2 = polygon.area * (111000 ** 2)  # rough conversion for lat/lon → meters
    target_num_cells = 1000  # ~100 tiles max
    cell_area = area_m2 / target_num_cells
    cell_size = (cell_area) ** 0.5 / 111000  # Convert meters → degrees (roughly)

    minx, miny, maxx, maxy = polygon.bounds
    results = []

    lat = miny
    while lat < maxy:
        lon = minx
        while lon < maxx:
            center_lat = lat + cell_size / 2
            center_lon = lon + cell_size / 2
            point = Polygon([
                (lon, lat),
                (lon + cell_size, lat),
                (lon + cell_size, lat + cell_size),
                (lon, lat + cell_size)
            ])

            if polygon.intersects(point):
                # Dummy features OR use real model later
                features = np.random.rand(10).reshape(1, -1)

                if model is not None:
                    conf = model.predict_proba(features)[0][1]
                else:
                    conf = round(random.uniform(0, 1), 2)

                results.append(FireTile(
                    lat=round(lat, 6),
                    lon=round(lon, 6),
                    conf=round(conf, 2),
                    size=round(cell_size, 6)
                ))
            lon += cell_size
        lat += cell_size

    return {"results": results}
