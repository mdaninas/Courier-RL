from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from src.map import graph_utils as gu
from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

PRIORITY_COLOR = {"normal": "#3186cc", "express": "#9b59b6", "urgent": "#e74c3c"}


def _cumulative_minutes(graph, route_nodes: list[int]) -> list[float]:
    times = [0.0]
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        data = graph.get_edge_data(u, v)
        if data:
            t = min(
                d.get("current_travel_time", d.get("base_travel_time", 0.5))
                for d in data.values()
            )
        else:
            t = 0.5
        times.append(times[-1] + float(t))
    return times


def _iso_times(cum_minutes: list[float], start_hour: int) -> list[str]:
    base = datetime(2024, 1, 1, int(start_hour) % 24, 0, 0)
    return [(base + timedelta(minutes=m)).isoformat() for m in cum_minutes]


def _sw(color: str) -> str:
    return (
        f'<span style="display:inline-block;width:11px;height:11px;background:{color};'
        f'border-radius:50%;vertical-align:-1px;"></span>'
    )


def _legend_html(env, label: str, metrics: dict[str, Any] | None) -> str:
    delivered = sum(1 for p in env.packages if p.delivered)
    late = sum(1 for p in env.packages if p.is_late())
    total = len(env.packages)
    extra = ""
    if metrics:
        extra = (
            f"<tr><td>Total time</td><td><b>{metrics.get('total_travel_minutes', 0):.0f} min</b></td></tr>"
            f"<tr><td>Distance</td><td><b>{metrics.get('total_distance_km', 0):.1f} km</b></td></tr>"
            f"<tr><td>Reward</td><td><b>{metrics.get('total_reward', 0):.0f}</b></td></tr>"
        )
    return f"""
    <div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999;
                background: rgba(255,255,255,0.95); padding: 12px 14px; border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.25); font-family: -apple-system, Segoe UI, sans-serif;
                font-size: 13px; color: #222; min-width: 210px;">
      <div style="font-weight:700; font-size:14px; margin-bottom:6px;">{label}</div>
      <table style="border-collapse:collapse;width:100%;">
        <tr><td>Delivered</td><td><b>{delivered}/{total}</b></td></tr>
        <tr><td>Late</td><td><b>{late}</b></td></tr>
        {extra}
      </table>
      <hr style="border:none;border-top:1px solid #eee;margin:8px 0;">
      <div style="font-size:12px;line-height:1.9;">
        {_sw('#3186cc')} normal &nbsp; {_sw('#9b59b6')} express &nbsp; {_sw('#e74c3c')} urgent<br>
        <span style="display:inline-block;width:11px;height:11px;border:2px solid #000;border-radius:50%;vertical-align:-1px;"></span> late &nbsp;
        {_sw('#999999')} not delivered<br>
        <span style="display:inline-block;width:11px;height:11px;background:#111;border-radius:2px;vertical-align:-1px;"></span> depot &nbsp;
        <span style="display:inline-block;width:11px;height:11px;background:#d9534f;border-radius:50%;vertical-align:-1px;"></span> traffic hotspot
      </div>
    </div>
    """


def render_episode_map(
    env,
    route_nodes: list[int] | None = None,
    save_path: str = "data/results/route_map.html",
    title: str = "Courier route",
    metrics: dict[str, Any] | None = None,
    animate: bool = True,
) -> str:
    import folium
    from folium.plugins import AntPath, Fullscreen, MiniMap

    graph = env.graph
    bounds = gu.graph_bounds(graph)
    fmap = folium.Map(
        location=[bounds["center_lat"], bounds["center_lon"]],
        zoom_start=15,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer("cartodbpositron", name="Light map").add_to(fmap)
    folium.TileLayer("OpenStreetMap", name="Street map").add_to(fmap)

    g_hotspots = folium.FeatureGroup(name="Traffic hotspots", show=True)
    g_packages = folium.FeatureGroup(name="Packages", show=True)
    g_order = folium.FeatureGroup(name="Delivery order", show=True)
    g_route = folium.FeatureGroup(name="Route", show=True)

    if env.traffic is not None:
        for hs in env.traffic.hotspots:
            folium.Circle(
                location=[hs["lat"], hs["lon"]],
                radius=hs["radius_m"],
                color="#d9534f",
                fill=True,
                fill_opacity=0.10,
                weight=1,
                popup=f"Traffic hotspot ×{hs['multiplier']:.1f}",
            ).add_to(g_hotspots)

    dlat, dlon = gu.node_latlon(graph, env.depot_node)
    folium.Marker(
        [dlat, dlon],
        tooltip="Depot",
        popup="Depot (start)",
        icon=folium.Icon(color="black", icon="home", prefix="fa"),
    ).add_to(fmap)

    for p in env.packages:
        lat, lon = gu.node_latlon(graph, p.destination_node)
        base_color = PRIORITY_COLOR.get(p.priority, "#3186cc")
        fill = base_color if p.delivered else "#bbbbbb"
        border = "#000000" if p.is_late() else fill
        status = (
            f"delivered @ {p.delivery_time:.0f} min" if p.delivered else "not delivered"
        )
        on_time = "LATE" if p.is_late() else ("on time" if p.delivered else "—")
        folium.CircleMarker(
            [lat, lon],
            radius=7,
            color=border,
            weight=3 if p.is_late() else 1,
            fill=True,
            fill_color=fill,
            fill_opacity=0.9,
            tooltip=f"Pkg {p.package_id} ({p.priority})",
            popup=folium.Popup(
                f"<b>Package {p.package_id}</b><br>"
                f"priority: {p.priority}<br>"
                f"deadline: {p.deadline:.0f} min<br>"
                f"status: {status}<br>"
                f"{on_time}",
                max_width=220,
            ),
        ).add_to(g_packages)

    id_to_pkg = {p.package_id: p for p in env.packages}
    order = env.courier.delivered_packages if env.courier else []
    for seq, pkg_id in enumerate(order, start=1):
        p = id_to_pkg.get(pkg_id)
        if p is None:
            continue
        lat, lon = gu.node_latlon(graph, p.destination_node)
        color = PRIORITY_COLOR.get(p.priority, "#3186cc")
        folium.map.Marker(
            [lat, lon],
            icon=folium.DivIcon(
                icon_size=(22, 22),
                icon_anchor=(11, 11),
                html=(
                    f'<div style="background:{color};color:#fff;border:2px solid #fff;'
                    f'border-radius:50%;width:20px;height:20px;line-height:20px;'
                    f'text-align:center;font-size:11px;font-weight:700;'
                    f'box-shadow:0 1px 3px rgba(0,0,0,0.4);">{seq}</div>'
                ),
            ),
        ).add_to(g_order)

    route_nodes = route_nodes if route_nodes is not None else env.route_history()
    coords: list[tuple[float, float]] = []
    if route_nodes and len(route_nodes) > 1:
        coords = [gu.node_latlon(graph, n) for n in route_nodes]
        AntPath(
            coords,
            color="#0275d8",
            weight=4,
            opacity=0.85,
            delay=600,
            dash_array=[12, 24],
        ).add_to(g_route)

    for grp in (g_hotspots, g_route, g_packages, g_order):
        grp.add_to(fmap)

    if animate and coords and len(coords) > 1:
        from folium.plugins import TimestampedGeoJson

        cum = _cumulative_minutes(graph, route_nodes)
        times = _iso_times(cum, getattr(env, "start_hour", 8))
        geojson = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[lon, lat] for (lat, lon) in coords],
                    },
                    "properties": {
                        "times": times,
                        "style": {"color": "#0275d8", "weight": 5, "opacity": 0.9},
                        "icon": "circle",
                        "iconstyle": {
                            "fillColor": "#e74c3c",
                            "fillOpacity": 1.0,
                            "stroke": True,
                            "color": "#fff",
                            "weight": 2,
                            "radius": 8,
                        },
                    },
                }
            ],
        }
        TimestampedGeoJson(
            geojson,
            period="PT2M",
            add_last_point=True,
            auto_play=False,
            loop=False,
            max_speed=10,
            loop_button=True,
            date_options="HH:mm",
            time_slider_drag_update=True,
            transition_time=120,
        ).add_to(fmap)

    Fullscreen(position="topright").add_to(fmap)
    MiniMap(toggle_display=True, position="bottomright").add_to(fmap)
    folium.LayerControl(collapsed=False, position="topleft").add_to(fmap)
    fmap.get_root().html.add_child(folium.Element(_legend_html(env, title, metrics)))

    out = resolve_path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out))
    logger.info("Saved interactive route map -> %s", out)
    return str(out)
