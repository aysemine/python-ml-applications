from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import uvicorn

app = FastAPI()

if __name__ == "__app__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PostBase(BaseModel):
    observed_length: float
    observed_weight: float
    scientific_name: str
    genus: str
    age_class: str
    sex: str
    habitat_simple: str


with open('rf_model_pca.pkl', 'rb') as f:
    saved = joblib.load(f)
model = saved["model"]
model_columns = saved["columns"]


def preprocess_for_model(post: PostBase, model_columns):
    df = pd.DataFrame([{
        "Observed Length (m)": post.observed_length,
        "Observed Weight (kg)": post.observed_weight,
        "Scientific Name": post.scientific_name,
        "Genus": post.genus,
        "Age Class": post.age_class,
        "Sex": post.sex,
        "Habitat Simple": post.habitat_simple
    }])

    df_encoded = pd.get_dummies(df)

    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)
    return df_encoded


@app.post("/", tags=["Predict"])
def post(postBase: PostBase):
    df_model = preprocess_for_model(postBase, model_columns)
    prediction = model.predict(df_model)
    predicted_class = prediction[0]
    return {"prediction": predicted_class}





