# app/main.py
import uvicorn, os
from fastapi import FastAPI
from app.api.routes import router

app = FastAPI()
app.include_router(router)

if __name__ == "__main__": 
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host=host, port=port)