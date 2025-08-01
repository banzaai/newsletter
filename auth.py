# auth.py
import os
from fastapi import Depends, HTTPException, Header

AUTHORIZED_TOKEN = os.getenv("AUTHORIZED_TOKEN")
AUTHORIZED_EMAIL = os.getenv("AUTHORIZED_EMAIL")

def verify_token(authorization: str = Header(...)):
    if authorization != f"Bearer {AUTHORIZED_TOKEN}":
        raise HTTPException(status_code=403, detail="Unauthorized")
    return AUTHORIZED_EMAIL
