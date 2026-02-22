"""JSON file-based user store with bcrypt passwords."""
import json
import os
import threading

from passlib.hash import bcrypt as _bcrypt

import nimcp_logger
from config import AUTH_USER, AUTH_PASS_HASH

_log = nimcp_logger.get("user_store")
_STORE_PATH = os.path.join(os.path.dirname(__file__), "users.json")
_lock = threading.Lock()


def _load() -> dict[str, dict]:
    if not os.path.isfile(_STORE_PATH):
        return {}
    with open(_STORE_PATH) as f:
        return json.load(f)


def _save(users: dict[str, dict]) -> None:
    with open(_STORE_PATH, "w") as f:
        json.dump(users, f, indent=2)


def _ensure_seed_user(users: dict[str, dict]) -> bool:
    """Seed the admin user from .env if not already in the store."""
    if AUTH_USER in users or not AUTH_PASS_HASH:
        return False
    users[AUTH_USER] = {"password_hash": AUTH_PASS_HASH}
    return True


def verify(username: str, password: str) -> bool:
    """Verify credentials against the user store (and .env seed user)."""
    with _lock:
        users = _load()
        changed = _ensure_seed_user(users)
        if changed:
            _save(users)

    entry = users.get(username)
    if not entry:
        return False
    try:
        return _bcrypt.verify(password, entry["password_hash"])
    except Exception as exc:
        _log.warning("Password verify error for %s: %s", username, exc)
        return False


def register(username: str, password: str) -> None:
    """Register a new user. Raises ValueError if username taken or invalid."""
    username = username.strip()
    if not username or len(username) < 2:
        raise ValueError("Username must be at least 2 characters")
    if len(username) > 64:
        raise ValueError("Username must be 64 characters or fewer")
    if not username.isalnum() and not all(c.isalnum() or c in "_-" for c in username):
        raise ValueError("Username may only contain letters, numbers, hyphens, and underscores")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters")

    with _lock:
        users = _load()
        _ensure_seed_user(users)
        if username in users:
            raise ValueError("Username already taken")
        users[username] = {"password_hash": _bcrypt.hash(password)}
        _save(users)
    _log.info("Registered new user: %s", username)


def user_exists(username: str) -> bool:
    with _lock:
        users = _load()
        _ensure_seed_user(users)
        return username in users
