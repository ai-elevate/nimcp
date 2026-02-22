"""Authentication endpoints — login and registration."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import nimcp_logger
import user_store

_log = nimcp_logger.get("routers.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])


class AuthRequest(BaseModel):
    username: str
    password: str


@router.post("/login")
async def login(req: AuthRequest):
    if not user_store.verify(req.username, req.password):
        raise HTTPException(401, "Invalid username or password")
    return {"ok": True, "username": req.username}


@router.post("/register")
async def register(req: AuthRequest):
    try:
        user_store.register(req.username, req.password)
        return {"ok": True, "username": req.username}
    except ValueError as exc:
        raise HTTPException(400, str(exc))
