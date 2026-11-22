"""Live five-object tabletop physics; frozen single-object study data is untouched."""

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

from grasp import GraspParameters
from grasp_study_objects import STUDY_OBJECTS
from grasp_study_physics import PHYSICAL
from grasp_tabletop import FiveObjectSimulation
from server import local_only

ROOT = Path(__file__).resolve().parent
PERIOD_S = 0.05
LOGGER = logging.getLogger(__name__)
VIEWER = web.AppKey("study_viewer")
CLIENTS = web.AppKey("study_clients", set)


class StudyViewer:
    def __init__(self):
        self.targets = tuple(STUDY_OBJECTS)
        self.sim = None
        self.serial = 0
        self.frame_id = 0
        self.view = "workspace"
        self.paused = False
        self.auto_advance = True
        self.error = None
        self.select(self.targets[0])

    def close(self):
        if self.sim is not None:
            self.sim.close()
            self.sim = None

    def select(self, target):
        if target not in self.targets:
            raise ValueError(f"Unknown grasp target: {target}")
        self.close()
        self.sim = FiveObjectSimulation(target)
        self.target = target
        self.serial += 1
        self.started_at = None
        self.hold_started = None
        self.verified_hold_s = 0.0
        self.finished_at = None
        self.outcome = None
        self.initial_z = None
        self.invalid_reason = None
        self.paused = False

    def next(self):
        index = (self.targets.index(self.target) + 1) % len(self.targets)
        self.select(self.targets[index])

    def advance(self):
        if self.paused or self.error:
            return
        if self.finished_at is not None:
            if self.auto_advance and time.monotonic() - self.finished_at >= 3:
                self.next()
            return
        dt = float(self.sim.model.opt.timestep)
        for _ in range(round(PERIOD_S / dt)):
            self.sim.step(dt)
            now = float(self.sim.data.time)
            if self.started_at is None:
                if now < 1.0 - 1e-12:
                    continue
                self.invalid_reason = self.sim.skill.precondition_error("grasp_and_hold")
                if self.invalid_reason is not None:
                    self.finished_at = time.monotonic()
                    break
                settings = self.sim.settings
                self.initial_z = float(self.sim.data.xpos[self.sim.skill.target_body, 2])
                self.started_at = now
                self.sim.start_task("grasp", GraspParameters(
                    initial_opening_m=settings["opening_m"],
                    descent_speed_m_s=settings["descent_speed_m_s"],
                    close_trigger_height_m=settings["close_trigger_m"],
                ))
            predicates = self.sim.skill.grasp_postconditions()
            eligible = (self.sim.skill.status in ("running", "succeeded")
                        and self.sim.skill.phase in ("hold", "done")
                        and all(predicates[key] for key in PHYSICAL))
            if eligible:
                if self.hold_started is None:
                    self.hold_started = now
                self.verified_hold_s = now - self.hold_started
            else:
                self.hold_started, self.verified_hold_s = None, 0.0
            if self.sim.skill.status == "failed":
                self.outcome = False
                self.finished_at = time.monotonic()
                break
            if self.sim.skill.status == "succeeded" and self.verified_hold_s >= 2.0:
                self.outcome = True
                self.finished_at = time.monotonic()
                break
            if now - self.started_at > 22.1:
                raise RuntimeError("Independent observer watchdog exceeded in live rerun")

    def state(self):
        sim = self.sim
        if self.error:
            return {"runtime_error": self.error, "paused": True}
        phase = "settling" if self.started_at is None else sim.skill.phase
        return {
            "runtime_error": None, "scene_mode": "five_objects_together",
            "target": self.target, "targets": list(self.targets),
            "objects": sim.object_states(),
            "settings": sim.settings, "run_serial": self.serial,
            "frame_id": self.frame_id, "time_s": float(sim.data.time),
            "execution_s": None if self.started_at is None else float(sim.data.time) - self.started_at,
            "phase": phase, "controller_status": sim.skill.status,
            "outcome": self.outcome, "finished": self.finished_at is not None,
            "invalid_reason": self.invalid_reason, "failure_reason": sim.skill.failure_reason,
            "verified_hold_s": self.verified_hold_s,
            "predicates": sim.skill.grasp_postconditions(),
            "lift_m": None if self.initial_z is None else float(sim.data.xpos[sim.skill.target_body, 2]) - self.initial_z,
            "joint_positions_rad": sim.data.qpos[sim.arm_qpos].tolist(),
            "gripper_opening_m": float(sim.data.qpos[sim.finger_qpos].sum()),
            "target_position_m": sim.data.xpos[sim.skill.target_body].tolist(),
            "warnings": sim.warning_names(), "paused": self.paused,
            "auto_advance": self.auto_advance, "view": self.view,
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
        if action == "select":
            target = message.get("target")
            if target not in self.targets:
                return "Unknown grasp target"
            self.select(target)
        elif action == "restart":
            self.select(self.target)
        elif action == "next":
            self.next()
        elif action == "pause":
            self.paused = not self.paused
        elif action == "view":
            if message.get("view") not in ("workspace", "close"):
                return "Unknown camera view"
            self.view = message["view"]
        elif action == "auto":
            if not isinstance(message.get("enabled"), bool):
                return "Auto-advance must be a boolean"
            self.auto_advance = message["enabled"]
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
