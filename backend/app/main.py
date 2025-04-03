from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from app.api.routes import router

app = FastAPI()

database_url = "sqlite:///./.test.db"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.include_router(router, prefix="/api")
