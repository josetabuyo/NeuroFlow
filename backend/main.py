"""NeuroFlow Backend — FastAPI entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.websocket import ws_router

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="NeuroFlow",
    description="Connectionist neural automata framework",
    version="0.1.0",
)

# CORS — allow frontend origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://neuroflow.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)
app.include_router(ws_router)
