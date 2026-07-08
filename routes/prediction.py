from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

router = APIRouter()

templates = Jinja2Templates(directory="templates")


@router.get("/")
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


@router.post("/predict")
async def predict(
    request: Request,
    name: str = Form(...),
    year: int = Form(...),
    km_driven: int = Form(...),
    fuel: str = Form(...),
    seller_type: str = Form(...),
    transmission: str = Form(...),
    owner: str = Form(...)
):

    data = CustomData(
        name=name,
        year=year,
        km_driven=km_driven,
        fuel=fuel,
        seller_type=seller_type,
        transmission=transmission,
        owner=owner
    )

    pred_df = data.get_data_as_dataframe()

    predict_pipeline = PredictPipeline()

    prediction = predict_pipeline.predict(pred_df)

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": round(float(prediction[0]), 2)
        }
    )