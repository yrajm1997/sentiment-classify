import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
#print(sys.path)

from fastapi import FastAPI, Request, APIRouter, Form
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app import __version__, schemas

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sentiment_model import __version__ as _version
from sentiment_model.config.core import config
from sentiment_model.processing.data_manager import load_tokenizer
from sentiment_model.predict import make_prediction


tokenizer_file_name = f"{config.app_config.tokenizer_save_file}{_version}.json"
tokenizer = load_tokenizer(file_name = tokenizer_file_name)


app = FastAPI(
    title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

templates = Jinja2Templates(directory="app/templates")


def preprocess_text(text):
    text = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text, 
                             maxlen = config.model_config.max_review_length, 
                             padding= config.model_config.padding_type, 
                             truncating= config.model_config.truncating_type)
    
    return text_pad


@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {'request': request,})


@app.post("/predict/")
async def prediction(request: Request, review_text: str = Form(...)):

    input_text = review_text
    data_in = preprocess_text(review_text)
    
    results = make_prediction(input_data = data_in)
    y_pred = results['predictions'][0]
    
    return templates.TemplateResponse("predict.html", {"request": request,
                                                       "result": y_pred,
                                                       "input_text": input_text,})


@app.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=_version
    )

    return health.dict()


# Set all CORS enabled origins
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
