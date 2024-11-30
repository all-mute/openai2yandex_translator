from fastapi import FastAPI
from app.yc_log_handler import ycLogHandler
from app.app import app
import os, sys, json
from dotenv import load_dotenv

load_dotenv()

GITHUB_SHA = os.getenv("GITHUB_SHA", "unknown_version")
GITHUB_REF = os.getenv("GITHUB_REF", "unknown_branch")

main_app = FastAPI(
    title="OpenAI SDK Adapter",
    description="Adapter from OpenAI SDK to Yandex Cloud FoMo API",
    version=f"{GITHUB_SHA=} - {GITHUB_REF=}"
)

main_app.include_router(app)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run("main:main_app", host="0.0.0.0", port=9041, reload=True)
