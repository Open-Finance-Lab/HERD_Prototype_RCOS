import uvicorn
from fastapi import FastAPI
from app.api import routes

app = FastAPI(title="Expert Inference Service")

app.include_router(routes.router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)
