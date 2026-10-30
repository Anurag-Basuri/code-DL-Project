"""
=============================================================================
Hinglish Code-Switching Token-Level Language Identification
Interactive App Entrypoint (FastAPI + Modern Web Dashboard)
=============================================================================
Usage:
    python app.py
    # or
    python server.py
"""

import os
import socket
import uvicorn
from server import app


def get_available_port(preferred_port: int = 8000) -> int:
    """Finds preferred port or the next free port to prevent Errno 10048."""
    for p in range(preferred_port, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            # If connect_ex returns non-zero, port is not currently listening (free)
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return preferred_port


if __name__ == "__main__":
    preferred = int(os.environ.get("PORT", 8000))
    port = get_available_port(preferred)

    print("=" * 70)
    print(">> Hinglish Language ID -- Production Server")
    print(f">> Web Dashboard: http://127.0.0.1:{port}")
    print(f">> API Docs:      http://127.0.0.1:{port}/docs")
    if port != preferred:
        print(f">> Note: Port {preferred} was in use, automatically switched to {port}")
    print("=" * 70)

    uvicorn.run(app, host="0.0.0.0", port=port)
