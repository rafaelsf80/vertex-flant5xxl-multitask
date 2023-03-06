""" 
    FastAPI app with the Uvicorn server
"""
from fastapi import FastAPI, Request
from fastapi.logger import logger

from typing import Dict, List, Any
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import json
import logging
import numpy as np
import os
import torch

from transformers import pipeline

app = FastAPI()

gunicorn_logger = logging.getLogger('gunicorn.error')
logger.handlers = gunicorn_logger.handlers

if __name__ != "main":
    logger.setLevel(gunicorn_logger.level)
else:
    logger.setLevel(logging.INFO)

logger.info(f"Is CUDA available: {torch.cuda.is_available()}")
logger.info(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

model =  AutoModelForSeq2SeqLM.from_pretrained('../flan-t5-xxl-sharded-fp16', device_map="auto")#, load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained('../flan-t5-xxl-sharded-fp16')

@app.get(os.environ['AIP_HEALTH_ROUTE'], status_code=200)
def health():
    return {"status": "healthy"}

@app.post(os.environ['AIP_PREDICT_ROUTE'])
async def predict(request: Request):
    body = await request.json()

    instances = body["instances"]

    outputs = []
    for instance in instances:
        input_ids = tokenizer(instance, return_tensors="pt").input_ids
        output = model.generate(input_ids)
        prediction = tokenizer.decode(output[0], skip_special_tokens=True)
        outputs.append(prediction)


    return {"predictions": [outputs]}

