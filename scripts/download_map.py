from __future__ import annotations

import argparse

import _bootstrap

from src.map import osm_loader
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("download_map")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download an OSM road network.")
    ap.add_argument("--config", default="config/jakarta_menteng.yaml")
    ap.add_argument("--place", default=None, help="OSM place name, e.g. 'Menteng, Jakarta, Indonesia'")
    ap.add_argument("--bbox", nargs=4, type=float, default=None,
                    metavar=("NORTH", "SOUTH", "EAST", "WEST"))
    ap.add_argument("--network-type", default=None, help="drive/walk/bike")
    ap.add_argument("--output", default=None, help="GraphML cache path")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.place:
        config["map"]["place_name"] = args.place
        config["map"]["bbox"] = None
    if args.bbox:
        config["map"]["bbox"] = args.bbox
        config["map"]["place_name"] = None
    if args.network_type:
        config["map"]["network_type"] = args.network_type
    if args.output:
        config["map"]["cache_path"] = args.output

    graph = osm_loader.load_map(config, force_download=args.force)
    cache = resolve_path(config["map"]["cache_path"])
    logger.info("Done. nodes=%d edges=%d synthetic=%s cache=%s",
                graph.number_of_nodes(), graph.number_of_edges(),
                graph.graph.get("synthetic", False), cache)


if __name__ == "__main__":
    main()
