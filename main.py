print("Hello, FastAPI with SAM2 API! This is the main.py file for the chatbot application.")

from fastapi import FastAPI, Form, Request

import os
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello World"}



# uvicorn main:app --reload
