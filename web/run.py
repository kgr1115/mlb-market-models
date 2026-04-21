"""
Launcher: starts the FastAPI backend + serves the React frontend from one port.

Usage:
    pip install -r web/requirements.txt
    python web/run.py                 # LAN-exposed (default: binds 0.0.0.0)
    python web/run.py --local         # bind 127.0.0.1 only (no LAN access)
    python web/run.py --port 9000     # custom port

Or with env vars: HOST=0.0.0.0 PORT=8000 OPEN_BROWSER=0 python web/run.py

On Windows, the first LAN launch may trigger a Windows Defender Firewall prompt
asking whether to allow Python through the firewall — allow it on "Private
networks" only (not public) so other devices on your LAN can reach the app.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web" / "backend"))

import uvicorn  # noqa: E402

from web.backend.api import app  # noqa: E402, F401


def detect_lan_ip() -> str | None:
    """Best-effort: find this machine's primary LAN IPv4 address.

    Uses the classic "connect to a non-routable address" trick — no packets
    are actually sent, but the OS picks the outbound interface for us.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("10.255.255.255", 1))
            ip = s.getsockname()[0]
        finally:
            s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    # Fallback: look through hostname resolutions
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127."):
                return ip
    except Exception:
        pass
    return None


def main():
    ap = argparse.ArgumentParser(description="MLB Betting Predictor — web launcher")
    ap.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"),
                    help="Interface to bind (default 0.0.0.0 = all interfaces).")
    ap.add_argument("--local", action="store_true",
                    help="Shortcut: bind 127.0.0.1 only (no LAN access).")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")),
                    help="Port (default 8000).")
    ap.add_argument("--no-open", action="store_true",
                    help="Don't auto-open the browser.")
    args = ap.parse_args()

    host = "127.0.0.1" if args.local else args.host
    port = args.port
    lan_ip = detect_lan_ip()

    print()
    print("  MLB Betting Predictor")
    print("  " + "─" * 44)
    print(f"  Local      →  http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        if lan_ip:
            print(f"  Network    →  http://{lan_ip}:{port}")
        else:
            print("  Network    →  (could not auto-detect LAN IP — check `ipconfig`)")
    elif host not in ("127.0.0.1", "localhost"):
        print(f"  Bound to   →  http://{host}:{port}")
    print()
    if host == "0.0.0.0" and os.name == "nt":
        print("  First run on Windows may prompt Windows Defender Firewall —")
        print("  allow Python on Private networks so other devices can connect.")
        print()

    if not args.no_open and os.environ.get("OPEN_BROWSER", "1") != "0":
        try:
            webbrowser.open(f"http://127.0.0.1:{port}")
        except Exception:
            pass

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
