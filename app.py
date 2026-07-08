from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = FastAPI(
    title="🚗 AutoValueAI",
    description="Used Car Price Prediction",
    version="1.0.0"
)

# Templates folder
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "prediction": None
        }
    )


@app.post("/predict", response_class=HTMLResponse)
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
    try:
        # Create input object
        data = CustomData(
            name=name,
            year=year,
            km_driven=km_driven,
            fuel=fuel,
            seller_type=seller_type,
            transmission=transmission,
            owner=owner
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_dataframe()

        # Predict
        predictor = PredictPipeline()
        prediction = predictor.predict(pred_df)

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": f"₹ {prediction[0]:,.2f}"
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={
                "prediction": f"Error: {str(e)}"
            }
        )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "message": "AutoValueAI API is running"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )