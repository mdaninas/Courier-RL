from __future__ import annotations

import argparse

import _bootstrap

from src.visualization.interactive_app import serve


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the interactive courier demo.")
    ap.add_argument("--config", default="config/jakarta_menteng.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    serve(args.config, host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
