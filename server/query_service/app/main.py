import uvicorn, os
from fastapi import FastAPI
from app.api.routes import router
from app.startup import preload_all_models

async def lifespan(app: FastAPI):
    preload_all_models()
    yield
    #Shutdown Logic Eventually

app = FastAPI(lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__": 
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host=host, port=port)