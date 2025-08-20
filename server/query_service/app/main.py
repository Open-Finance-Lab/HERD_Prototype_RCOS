import uvicorn, os, subprocess
from fastapi import FastAPI
from app.api.routes import router

async def lifespan(app: FastAPI):
    try:
        subprocess.run(
            ["helm", "upgrade", "--install", "experts", "../charts/experts-chart"],
            check=False
        )
    except Exception:
        pass

    yield
    # Cleanup or shutdown tasks can be added here if needed

app = FastAPI(lifespan=lifespan)
app.include_router(router)

if __name__ == "__main__": 
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host=host, port=port)