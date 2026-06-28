from fastapi import FastAPI,Request,Form
from fastapi.templating import Jinja2Templates
from typing import Annotated
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app=FastAPI()
templates=Jinja2Templates(directory='templates')

@app.get('/')
def root(request : Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})


@app.get('/predict')
def home(request: Request):
    return templates.TemplateResponse(request=request,name='predict.html',context={'request': request})

@app.post('/predict')
def predict(
    request: Request,
    gender: Annotated[str, Form(...)],
    ethnicity: Annotated[str, Form(...)],
    parental_level_of_education: Annotated[str, Form(...)],
    lunch: Annotated[str, Form(...)],
    test_preparation_course: Annotated[str, Form(...)],
    reading_score: Annotated[int, Form(...)],
    writing_score: Annotated[int, Form(...)]
):
    data=CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
    )
    pred_df=data.get_data_as_data_frame()
    print(pred_df)

    predict_pipeline=PredictPipeline()
    results=predict_pipeline.predict(pred_df)

    return templates.TemplateResponse(request=request,name='predict.html',context={'request':request,'results':round(results[0],3)})