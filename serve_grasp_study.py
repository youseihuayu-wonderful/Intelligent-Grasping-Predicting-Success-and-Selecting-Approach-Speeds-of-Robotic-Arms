"""Live continuous five-object physical storage and contact-driven drawer closure."""

import argparse
import asyncio
import base64
from contextlib import suppress
from io import BytesIO
import json
import logging
from pathlib import Path
import time

from aiohttp import WSMsgType, web
import mujoco
from PIL import Image

from tabletop_storage import StorageSession
from server import local_only

ROOT = Path(__file__).resolve().parent
PERIOD_S = 0.05
LOGGER = logging.getLogger(__name__)
VIEWER = web.AppKey("study_viewer")
CLIENTS = web.AppKey("study_clients", set)


class StudyViewer:
    def __init__(self):
        self.session = None
        self.serial = 0
        self.frame_id = 0
        self.view = "workspace"
        self.paused = False
        self.error = None
        self.restart()

    @property
    def sim(self):
        return self.session.sim

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    def restart(self):
        self.close()
        self.session = StorageSession()
        self.serial += 1
        self.paused = False

    def advance(self):
        if not self.paused and not self.error:
            # Display advances at 2x wall time; physical controller settings and
            # the 2 ms integration step are unchanged.
            self.session.advance(2 * PERIOD_S)

    def state(self):
        if self.error:
            return {"runtime_error": self.error, "paused": True}
        return {
            **self.session.state(), "runtime_error": None,
            "run_serial": self.serial, "frame_id": self.frame_id,
            "execution_s": float(self.sim.data.time) - self.session.started_at,
            "paused": self.paused, "view": self.view,
        }

    def render(self):
        sim = self.sim
        if self.view == "workspace":
            sim.camera.azimuth, sim.camera.elevation, sim.camera.distance = 135, -38, 1.95
            sim.camera.lookat[:] = [.33, -.03, .86]
        else:
            sim.camera.azimuth, sim.camera.elevation, sim.camera.distance = 135, -28, .65
            sim.camera.lookat[:] = sim.data.xpos[sim.skill.target_body]
            sim.camera.lookat[2] += .07
        if sim.renderer is None:
            sim.renderer = mujoco.Renderer(sim.model, height=600, width=800)
        sim.renderer.update_scene(sim.data, camera=sim.camera, scene_option=sim.options)
        pixels = sim.renderer.render()
        with BytesIO() as buffer:
            Image.fromarray(pixels).save(buffer, format="JPEG", quality=85)
            result = base64.b64encode(buffer.getvalue()).decode("ascii")
        self.frame_id += 1
        return result

    def command(self, message):
        if not isinstance(message, dict):
            return "Command must be an object"
        action = message.get("action")
        if action == "restart":
            self.restart()
        elif action == "pause":
            self.paused = not self.paused
        elif action == "view":
            if message.get("view") not in ("workspace", "close"):
                return "Unknown camera view"
            self.view = message["view"]
        else:
            return "Unknown command"
        return None


async def broadcast(app, packet):
    for client in tuple(app[CLIENTS]):
        try:
            await asyncio.wait_for(client.send_json(packet), timeout=1)
        except (ConnectionError, asyncio.TimeoutError):
            app[CLIENTS].discard(client)
            await client.close()


async def physics_loop(app):
    viewer = app[VIEWER]
    try:
        while True:
            started = time.monotonic()
            if app[CLIENTS] and not viewer.error:
                viewer.advance()
                image = viewer.render()
                await broadcast(app, {"state": viewer.state(), "image": image})
            await asyncio.sleep(max(0, PERIOD_S - (time.monotonic() - started)))
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        LOGGER.exception("Live study physics/rendering stopped")
        viewer.error = f"{type(exc).__name__}: {exc}"
        await broadcast(app, {"state": viewer.state()})


async def lifecycle(app):
    task = asyncio.create_task(physics_loop(app), name="study-physics")
    try:
        yield
    finally:
        task.cancel()
        try:
            with suppress(asyncio.CancelledError):
                await task
        finally:
            for client in tuple(app[CLIENTS]):
                await client.close(code=1001, message=b"Server shutdown")
            app[VIEWER].close()


async def websocket(request):
    if request.headers.get("Origin") not in (None, f"{request.scheme}://{request.host}"):
        raise web.HTTPForbidden(text="Cross-origin control is not allowed")
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    app = request.app
    app[CLIENTS].add(ws)
    try:
        async for message in ws:
            if message.type == WSMsgType.TEXT:
                try:
                    payload = message.json()
                except json.JSONDecodeError:
                    await ws.send_json({"error": "Malformed JSON command"})
                    continue
                if app[VIEWER].error:
                    await ws.send_json({"error": "Physics stopped; restart the server. See its log."})
                    continue
                try:
                    error = app[VIEWER].command(payload)
                except Exception as exc:
                    LOGGER.exception("Live study command failed")
                    app[VIEWER].error = f"{type(exc).__name__}: {exc}"
                    await broadcast(app, {"state": app[VIEWER].state()})
                else:
                    if error:
                        await ws.send_json({"error": error})
    finally:
        app[CLIENTS].discard(ws)
    return ws


async def state(request):
    return web.json_response(request.app[VIEWER].state(), headers={"Cache-Control": "no-store"})


async def asset(request):
    name = request.match_info.get("name", "study.html")
    if name not in ("study.html", "study.js", "study.css"):
        raise web.HTTPNotFound()
    return web.FileResponse(ROOT / "web" / name, headers={"Cache-Control": "no-cache"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8877)
    args = parser.parse_args()
    app = web.Application(middlewares=[local_only])
    app[VIEWER] = StudyViewer()
    app[CLIENTS] = set()
    app.cleanup_ctx.append(lifecycle)
    app.router.add_get("/ws", websocket)
    app.router.add_get("/api/state", state)
    app.router.add_get("/", asset)
    app.router.add_get("/{name}", asset)
    web.run_app(app, host="127.0.0.1", port=args.port, access_log=None)


if __name__ == "__main__":
    main()
