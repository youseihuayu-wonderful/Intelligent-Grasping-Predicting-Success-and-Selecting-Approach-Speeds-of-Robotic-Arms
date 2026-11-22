# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["av==19.0.2", "mujoco==3.3.7", "numpy==2.2.6", "pillow>=11,<13"]
# ///
"""Render every recorded Experiment 02 trajectory sample, without control replay.

uv run render_tabletop_storage_video_en.py --run-dir results/experiment_02_five_object_storage/final \
    --output media/experiment-02-five-object-storage-en.mp4 \
    --preview media/experiment-02-five-object-storage.webp

The optional preview requires ffmpeg. Existing media or provenance are rejected;
choose new output paths rather than replacing an earlier experimental artifact.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from fractions import Fraction
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile
import zlib

import av
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version

from grasp_study_objects import STUDY_OBJECTS
from render_storage_video_en import default_font, portable_path, sha256


ROOT = Path(__file__).resolve().parent
TARGETS = frozenset(("spoon", "pen", "bowl", "bottle", "glass"))
PREDICATES = ("inside", "supported", "detached", "stable")
PLAYBACK_RATE = 2
SOURCE_HZ = 10
OUTPUT_FPS = 20
TIME_BASE = Fraction(1, 1_000_000)
FRAME_TICKS = 50_000
CAMERA = {"azimuth": 135, "elevation": -38, "distance": 1.95,
          "lookat": [.33, -.03, .86], "viewport": [800, 600]}
SIZE = (960, 840)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value, label: str) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label} must be a recorded finite number")
    require(math.isfinite(value), f"{label} must be finite")
    return float(value)


def validate_evidence(evidence: dict) -> list[dict]:
    require(evidence.get("outcome") is True and evidence.get("controller_status") == "succeeded",
            "Only an explicitly successful recorded run can be rendered")
    require(evidence.get("scene_mode") == "continuous_drawer_storage",
            "Evidence is not a continuous five-object storage run")
    for name in ("all_stored", "drawer_closed", "hand_empty"):
        require(evidence.get(name) is True, f"Final evidence does not establish {name}")
    require(evidence.get("scene_resets_during_mission") == 0
            and type(evidence.get("scene_resets_during_mission")) is int,
            "The recorded mission must have zero scene resets")
    require(evidence.get("warnings") == [] and evidence.get("failure_reason") is None,
            "Successful evidence must contain no warnings or failure reason")
    require(finite_number(evidence.get("verified_hold_s"), "verified_hold_s") >= 3.,
            "Strict final verification must have lasted at least three simulation seconds")
    objects = evidence.get("objects")
    require(isinstance(objects, list) and len(objects) == 5,
            "Final evidence must contain exactly five objects")
    require(all(isinstance(item, dict) for item in objects), "Malformed final object evidence")
    require({item.get("target") for item in objects} == TARGETS,
            "Final evidence must cover each of the five canonical objects once")
    for item in objects:
        for name in (*PREDICATES, "stored"):
            require(item.get(name) is True, f"{item['target']}: final {name} is not true")
        require(finite_number(item.get("support_force_n"), "support_force_n") > 1e-6,
                f"{item['target']}: final floor support force is absent")
    require(evidence.get("stored_count") == 5, "Final stored_count must be five")
    events = evidence.get("events")
    require(isinstance(events, list) and bool(events), "Recorded phase events are required")
    previous = -math.inf
    for event in events:
        require(isinstance(event, dict), "Malformed phase event")
        time_s = finite_number(event.get("time_s"), "event time_s")
        require(time_s >= previous, "Phase events must be in recorded chronological order")
        previous = time_s
        require(isinstance(event.get("phase"), str) and bool(event["phase"]),
                "Every phase event must have an actual logged phase")
        require(event.get("target") in TARGETS, "Unknown target in phase event")
    return events


def validate_sources(evidence: dict) -> dict[str, str]:
    recorded = evidence.get("source_sha256")
    require(isinstance(recorded, dict) and bool(recorded), "Recorded source hashes are required")
    required = {
        "tabletop_storage.py", "drawer_storage_checks.py", "grasp_tabletop.py",
        "grasp.py", "simulation.py", "grasp_study_physics.py", "grasp_study_objects.py",
        "scene.xml", "vendor/panda/panda.xml",
        "assets/grasp_study/tabletop_cabinet.xml",
        "vendor/libero/turbosquid_objects/wine_rack/wine_rack.xml",
        *(spec.asset_path for spec in STUDY_OBJECTS.values()),
    }
    require(required <= recorded.keys(),
            f"Recorded hashes omit required scene/controller sources: {sorted(required - recorded.keys())}")
    for relative, expected in recorded.items():
        require(isinstance(relative, str) and isinstance(expected, str)
                and len(expected) == 64 and all(c in "0123456789abcdef" for c in expected),
                "Malformed recorded source SHA-256 entry")
        path = (ROOT / relative).resolve()
        require(not Path(relative).is_absolute() and path.is_relative_to(ROOT),
                f"Source hash path leaves the repository: {relative}")
        require(path.is_file() and sha256(path) == expected,
                f"Current repository source does not match recorded SHA-256: {relative}")
    require(evidence.get("mujoco_version") == mujoco.__version__,
            "The MuJoCo version must match the recorded model runtime")
    return recorded


@contextmanager
def trajectory_arrays(path: Path):
    """Extract NPY members to bounded-memory, read-only memory maps."""
    names = ("time_s", "qpos", "qvel", "ctrl", "object_order", "predicate_names",
             "predicates", "support_force_n")
    with tempfile.TemporaryDirectory(prefix="experiment-02-trajectory-") as directory:
        arrays = {}
        try:
            with zipfile.ZipFile(path) as archive:
                require(len(archive.namelist()) == len(set(archive.namelist())),
                        "Trajectory archive contains duplicate members")
                for name in names:
                    member = name + ".npy"
                    if member not in archive.namelist():
                        continue
                    destination = Path(directory) / member
                    with archive.open(member) as source, destination.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                    arrays[name] = np.load(destination, mmap_mode="r", allow_pickle=False)
            yield arrays
        finally:
            for array in arrays.values():
                if isinstance(array, np.memmap):
                    array._mmap.close()


def validate_trajectory(arrays: dict, evidence: dict, events: list[dict], model) -> dict:
    required = {"time_s", "qpos", "object_order", "predicate_names", "predicates", "support_force_n"}
    require(required <= arrays.keys(), f"Trajectory lacks final-run evidence arrays: {sorted(required - arrays.keys())}")
    times, positions = arrays["time_s"], arrays["qpos"]
    require(times.ndim == 1 and len(times) >= 2, "At least two recorded timestamps are required")
    count = len(times)
    require(positions.shape == (count, model.nq), "Recorded qpos does not match the current model")
    for name, width in (("qvel", model.nv), ("ctrl", model.nu)):
        if name in arrays:
            require(arrays[name].shape == (count, width), f"Recorded {name} does not match the current model")
    for name in ("time_s", "qpos", "qvel", "ctrl", "support_force_n"):
        if name in arrays:
            require(np.issubdtype(arrays[name].dtype, np.number), f"{name} must be numeric")
            for start in range(0, count, 1024):
                require(bool(np.isfinite(arrays[name][start:start + 1024]).all()),
                        f"Recorded {name} contains non-finite values")
    order = arrays["object_order"].tolist()
    names = arrays["predicate_names"].tolist()
    require(isinstance(order, list) and len(order) == 5 and set(order) == TARGETS,
            "Recorded predicate object order is invalid")
    require(isinstance(names, list) and len(names) == 4 and set(names) == set(PREDICATES),
            "Recorded predicate names are invalid")
    require(arrays["predicates"].shape == (count, 5, 4)
            and arrays["predicates"].dtype == np.dtype(bool), "Recorded predicate array must be bool[frame,5,4]")
    require(arrays["support_force_n"].shape == (count, 5), "Recorded support forces must be [frame,5]")
    first, last = float(times[0]), float(times[-1])
    max_quantization_error = 0.
    for index in range(count):
        time_s = float(times[index])
        if index:
            require(abs(time_s - float(times[index - 1]) - 1 / SOURCE_HZ) <= 1e-7,
                    f"Frame {index}: expected uninterrupted 10 Hz source timestamps; refusing retiming")
        scaled = (time_s - first) / PLAYBACK_RATE
        max_quantization_error = max(max_quantization_error, abs(round(scaled / float(TIME_BASE)) * float(TIME_BASE) - scaled))
        require(round(scaled / float(TIME_BASE)) == index * FRAME_TICKS,
                "Source timestamps cannot be represented at 20 fps without retiming")
    require(events[0]["time_s"] <= first + 1e-9,
            "Initial frames have no logged phase; record the initial phase rather than guessing it")
    require(events[-1]["time_s"] <= last + 1e-9, "Phase log extends beyond the recorded trajectory")
    require(abs(finite_number(evidence.get("time_s"), "final time_s") - last) <= 1e-7,
            "Final verification timestamp does not match the last trajectory frame")
    hold_s = float(evidence["verified_hold_s"])
    require(last - first >= hold_s - 1 / SOURCE_HZ,
            "Trajectory is shorter than its claimed final verification hold")
    # These are logged labels only: replay contact dynamics never supply labels.
    for index in range(count - 1, -1, -1):
        if float(times[index]) < last - hold_s + 1 / SOURCE_HZ - 1e-7:
            break
        require(bool(arrays["predicates"][index].all()),
                f"Frame {index}: logged final hold contains a failed storage predicate")
        require(bool((arrays["support_force_n"][index] > 1e-6).all()),
                f"Frame {index}: logged final hold lacks physical floor support")
    final_objects = {item["target"]: item for item in evidence["objects"]}
    for index, target in enumerate(order):
        require(math.isclose(float(arrays["support_force_n"][-1, index]),
                             final_objects[target]["support_force_n"], rel_tol=1e-7, abs_tol=1e-9),
                f"Final logged support force differs from verification for {target}")
    require(math.isclose(float(model.opt.timestep),
                         finite_number(evidence.get("physics_timestep_s"), "physics_timestep_s"),
                         abs_tol=1e-12, rel_tol=0.), "Current and recorded physics timesteps differ")
    return {"frame_count": count, "first_time_s": first, "last_time_s": last,
            "source_time_span_s": last - first, "source_sample_rate_hz": SOURCE_HZ,
            "playback_rate": PLAYBACK_RATE, "output_fps": OUTPUT_FPS,
            "output_duration_s": count / OUTPUT_FPS,
            "last_sample_display_duration_s": 1 / OUTPUT_FPS,
            "timestamp_quantization_max_error_s": max_quantization_error,
            "time_base": str(TIME_BASE), "state_arrays": sorted(arrays),
            "object_order": order, "predicate_names": names}


def render_video(path: Path, arrays: dict, events: list[dict], model, data, renderer,
                 font: Path, bold: Path) -> int:
    camera = mujoco.MjvCamera()
    camera.azimuth = CAMERA["azimuth"]
    camera.elevation = CAMERA["elevation"]
    camera.distance = CAMERA["distance"]
    camera.lookat[:] = CAMERA["lookat"]
    options = mujoco.MjvOption()
    options.geomgroup[0] = 0
    options.geomgroup[3] = 0
    regular = ImageFont.truetype(str(font), 21)
    title = ImageFont.truetype(str(bold), 32)
    label = ImageFont.truetype(str(bold), 22)
    template = Image.new("RGB", SIZE, "#101722")
    draw = ImageDraw.Draw(template)
    draw.text((24, 16), "Experiment 02 | Five-object storage", font=title, fill="#eef3f8")
    draw.text((24, 64), "Recorded physical execution | 2x playback", font=label, fill="#8ddac7")
    draw.text((24, 783), "Simulation clock, not wall time | Logged phases, not inferred labels", font=regular, fill="#adbdcd")
    first = float(arrays["time_s"][0])
    event_index = 0
    count = 0
    packet_count = 0
    with av.open(str(path), "w", format="mp4", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream("libx264", rate=OUTPUT_FPS)
        stream.width, stream.height = SIZE
        stream.pix_fmt = "yuv420p"
        stream.time_base = TIME_BASE
        stream.codec_context.time_base = TIME_BASE
        stream.options = {"preset": "slow", "crf": "23", "profile": "high", "bf": "0"}

        def mux(packet):
            nonlocal packet_count
            packet.duration = round(FRAME_TICKS * TIME_BASE / packet.time_base)
            output.mux(packet)
            packet_count += 1

        for index, time_value in enumerate(arrays["time_s"]):
            time_s = float(time_value)
            # Clear initialization/previous-frame state; never integrate or run a skill.
            mujoco.mj_resetData(model, data)
            data.qpos[:] = arrays["qpos"][index]
            for name in ("qvel", "ctrl"):
                if name in arrays:
                    getattr(data, name)[:] = arrays[name][index]
            data.time = time_s
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=options)
            canvas = template.copy()
            canvas.paste(Image.fromarray(renderer.render()), (80, 132))
            while event_index + 1 < len(events) and events[event_index + 1]["time_s"] <= time_s + 1e-9:
                event_index += 1
            event = events[event_index]
            draw = ImageDraw.Draw(canvas)
            draw.text((24, 102), f"Simulation time: {time_s:.3f} s  |  Target: {event['target']}  |  Phase: {event['phase']}",
                      font=regular, fill="#eef3f8")
            phase_note = "Video from recorded physical state; no controller re-execution."
            if event["phase"] == "done":
                phase_note = (
                    "Mission complete (recorded successful final evidence)"
                    if event_index == len(events) - 1
                    else "Grasp hold verified; continuing"
                )
            draw.text((24, 748), phase_note, font=regular, fill="#adbdcd")
            frame = av.VideoFrame.from_image(canvas)
            frame.pts = round((time_s - first) / PLAYBACK_RATE / float(TIME_BASE))
            frame.time_base = TIME_BASE
            frame.duration = FRAME_TICKS
            for packet in stream.encode(frame):
                mux(packet)
            count += 1
        for packet in stream.encode():
            mux(packet)
    require(packet_count == count, "Encoder did not produce one video packet per recorded frame")
    return count


def animated_preview(video: Path, destination: Path, ffmpeg: str) -> list[str]:
    # Native APNG plus Pillow avoids requiring an optional FFmpeg WebP encoder.
    # Pillow reads APNG frames sequentially instead of retaining every RGB frame.
    intermediate = destination.with_suffix(".apng")
    command = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
               "-i", str(video), "-an", "-vf", "fps=5:round=down:eof_action=pass,scale=480:-2:flags=lanczos",
               "-c:v", "apng", "-plays", "0", str(intermediate)]
    with tempfile.TemporaryFile() as errors:
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=errors, check=False)
        if result.returncode:
            errors.seek(0, os.SEEK_END)
            length = errors.tell()
            errors.seek(max(0, length - 16_384))
            detail = errors.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg APNG conversion failed (exit {result.returncode}):\n{detail}")
    with av.open(str(video)) as clip:
        require(clip.duration is not None and clip.duration > 0, "MP4 has no recorded duration")
        duration_ms = round(clip.duration * 1000 / av.time_base)
    with Image.open(intermediate) as animation:
        require(getattr(animation, "is_animated", False) and animation.n_frames > 1,
                "FFmpeg produced no animated preview timeline")
        durations = [200] * animation.n_frames
        durations[-1] = duration_ms - 200 * (animation.n_frames - 1)
        require(0 < durations[-1] <= 200, "Preview resampling did not preserve the MP4 timeline")
        animation.save(destination, format="WEBP", save_all=True, duration=durations,
                       loop=0, quality=70, method=5, lossless=False)
    require(destination.is_file() and destination.stat().st_size > 0,
            "Pillow produced no animated WebP preview")
    with Image.open(destination) as image:
        require(getattr(image, "is_animated", False) and image.n_frames > 1,
                "Preview is not a full animated WebP; refusing a static substitute")
    return command


def file_facts(path: Path) -> dict:
    return {"path": portable_path(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def recorded_model(run_dir: Path, snapshot: dict, private_dir: Path) -> tuple[Path, dict]:
    """Resolve identity or explicitly declared lossless storage, never rebuild."""
    expected = snapshot.get("sha256")
    require(isinstance(expected, str) and len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected),
            "Original verification must contain a valid compiled-model SHA-256")
    raw = run_dir / "model.mjb"
    if raw.exists():
        require(raw.is_file(), "Recorded model.mjb is not a file")
        facts = file_facts(raw)
        require(facts["sha256"] == expected,
                "Raw compiled model does not match original verification SHA-256")
        return raw, {"encoding": "identity", "logical": facts, "stored": facts}

    descriptor_path = run_dir / "model_storage.json"
    require(descriptor_path.is_file(),
            "No exact model.mjb or explicit model_storage.json declaration exists")
    descriptor_facts = file_facts(descriptor_path)
    descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    fields = {"logical_path", "logical_sha256", "encoding", "stored_path",
              "stored_sha256", "uncompressed_bytes", "stored_bytes"}
    require(isinstance(descriptor, dict) and descriptor.keys() == fields,
            "model_storage.json must contain exactly the declared storage schema")
    require(descriptor["logical_path"] == "model.mjb"
            and descriptor["encoding"] == "gzip"
            and descriptor["stored_path"] == "model.mjb.gz",
            "Only explicitly declared model.mjb.gz storage of model.mjb is accepted")
    require(descriptor["logical_sha256"] == expected,
            "Stored model logical SHA-256 differs from original verification")
    stored_hash = descriptor["stored_sha256"]
    require(isinstance(stored_hash, str) and len(stored_hash) == 64
            and all(character in "0123456789abcdef" for character in stored_hash),
            "Stored model SHA-256 is invalid")
    for name in ("uncompressed_bytes", "stored_bytes"):
        require(type(descriptor[name]) is int and descriptor[name] > 0,
                f"Model storage {name} must be a positive integer")
    require(descriptor["stored_bytes"] < 100 * 1024 * 1024,
            "Compressed model is at or above the 100 MiB publication limit")
    stored = run_dir / descriptor["stored_path"]
    require(stored.is_file(), "Declared model.mjb.gz is missing")
    stored_facts = file_facts(stored)
    require(stored_facts["bytes"] == descriptor["stored_bytes"]
            and stored_facts["sha256"] == stored_hash,
            "Stored gzip model byte size or SHA-256 differs from its declaration")

    decoded = private_dir / "model.mjb"
    digest = hashlib.sha256()
    uncompressed_bytes = 0
    try:
        with gzip.open(stored, "rb") as source, decoded.open("xb") as destination:
            while chunk := source.read(1024 * 1024):
                uncompressed_bytes += len(chunk)
                require(uncompressed_bytes <= descriptor["uncompressed_bytes"],
                        "Decoded model exceeds its declared uncompressed byte size")
                digest.update(chunk)
                destination.write(chunk)
    except (OSError, EOFError, zlib.error) as error:
        raise ValueError(f"Cannot losslessly decode declared model.mjb.gz: {error}") from error
    logical_hash = digest.hexdigest()
    require(uncompressed_bytes == descriptor["uncompressed_bytes"],
            "Decoded model byte size differs from its declaration")
    require(logical_hash == descriptor["logical_sha256"] == expected,
            "Decoded model SHA-256 differs from descriptor or original verification")
    require(sha256(descriptor_path) == descriptor_facts["sha256"],
            "Model storage declaration changed while decoding")
    return decoded, {
        "encoding": "gzip",
        "logical": {"path": portable_path(raw), "sha256": logical_hash,
                    "bytes": uncompressed_bytes},
        "stored": stored_facts, "storage_descriptor": descriptor_facts,
        "declaration": descriptor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    for requested in (args.output, args.preview):
        if requested is not None and requested.is_symlink():
            raise FileExistsError(f"Refusing to replace an existing output symlink: {requested}")
    run_dir = args.run_dir.resolve()
    output = args.output.resolve()
    preview = args.preview.resolve() if args.preview else None
    provenance_path = output.with_suffix(".provenance.json")
    require(output.suffix.lower() == ".mp4", "--output must be an MP4 path")
    require(preview is None or preview.suffix.lower() == ".webp", "--preview must be a WebP path")
    destinations = [output, provenance_path] + ([preview] if preview else [])
    require(len(set(destinations)) == len(destinations), "Output paths must be distinct")
    for path in destinations:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to overwrite existing experiment output: {path}")
    verification = run_dir / "verification.json"
    trajectory = run_dir / "trajectory.npz"
    evidence = json.loads(verification.read_text(encoding="utf-8"))
    require(isinstance(evidence, dict), "verification.json must be a JSON object")
    events = validate_evidence(evidence)
    sources = validate_sources(evidence)
    snapshot = evidence.get("compiled_model")
    require(isinstance(snapshot, dict) and snapshot.get("path") == "model.mjb",
            "Final evidence must identify its exact compiled model.mjb snapshot")
    inputs = {"verification": file_facts(verification), "trajectory": file_facts(trajectory)}
    font, bold = default_font(False), default_font(True)
    ffmpeg = shutil.which("ffmpeg") if preview else None
    if preview and not ffmpeg:
        raise FileNotFoundError("--preview requires ffmpeg with its native APNG encoder on PATH")
    renderer = {"path": portable_path(Path(__file__)), "sha256": sha256(Path(__file__)),
                "presentation_helpers": file_facts(ROOT / "render_storage_video_en.py"),
                "dependencies": {"av": av.__version__, "mujoco": mujoco.__version__,
                                 "numpy": np.__version__, "pillow": pillow_version},
                "font": file_facts(font), "bold_font": file_facts(bold)}
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="experiment-02-render-", dir=output.parent) as directory:
        directory = Path(directory)
        pending_video = directory / output.name
        model_path, model_facts = recorded_model(run_dir, snapshot, directory)
        inputs["compiled_model"] = model_facts
        # Load the actual recorded FiveObjectSimulation model, including baked assets
        # and runtime model overrides. Rebuilding the scene is not equivalent evidence.
        model = mujoco.MjModel.from_binary_path(str(model_path))
        data = mujoco.MjData(model)
        scene_renderer = mujoco.Renderer(model, height=600, width=800)
        try:
            with trajectory_arrays(trajectory) as arrays:
                timing = validate_trajectory(arrays, evidence, events, model)
                count = render_video(pending_video, arrays, events, model, data, scene_renderer, font, bold)
        finally:
            scene_renderer.close()
        require(count == timing["frame_count"], "Not every recorded trajectory frame was encoded")
        # Detect source/input changes during rendering before publishing any output.
        validate_sources(evidence)
        require(all(sha256(run_dir / filename) == inputs[key]["sha256"]
                    for key, filename in (("verification", "verification.json"), ("trajectory", "trajectory.npz"))),
                "Recorded inputs changed during rendering")
        for name in ("stored", "storage_descriptor"):
            if name in model_facts:
                fact = model_facts[name]
                path = ROOT / fact["path"]
                require(path.is_file() and path.stat().st_size == fact["bytes"]
                        and sha256(path) == fact["sha256"],
                        f"Recorded compiled-model {name} changed during rendering")
        require(sha256(Path(__file__)) == renderer["sha256"], "Renderer changed during rendering")
        facts = file_facts(pending_video)
        facts["path"] = portable_path(output)
        facts["frame_count"] = count
        provenance = {
            "kind": "experiment_02_recorded_physical_trajectory_render", "inputs": inputs,
            "source_sha256": sources, "renderer": renderer, "timing": timing,
            "camera": CAMERA, "canvas": list(SIZE), "events": events,
            "recorded_final_evidence": evidence,
            "policy": {"one_mp4_frame_per_recorded_sample": True, "cuts": [],
                       "interpolation": False, "controller_reexecution": False,
                       "new_acceptance_labels": False, "object_attachments": False,
                       "added_dwell": False,
                       "missing_qvel_or_ctrl": "MuJoCo reset defaults, never inferred; present arrays are restored",
                       "phase": "Latest logged event at or before each recorded simulation timestamp",
                       "done_caption": "Non-final done: Grasp hold verified; continuing. Final done: mission complete, supported by successful final evidence. Raw phase/event fields remain unchanged.",
                       "last_frame_duration": "One 10 Hz sample period at 2x; no extra terminal dwell",
                       "preview": "Full MP4 timeline converted to 480 px / 5 fps animated WebP at unchanged 2x timing"},
            "encoding": {"codec": "libx264", "pixel_format": "yuv420p", "fps": OUTPUT_FPS,
                         "preset": "slow", "crf": 23, "profile": "high", "b_frames": 0,
                         "faststart": True, "audio": False},
            "output": facts,
        }
        pending = [(pending_video, output)]
        if preview:
            pending_preview = directory / preview.name
            command = animated_preview(pending_video, pending_preview, ffmpeg)
            preview_facts = file_facts(pending_preview)
            preview_facts["path"] = portable_path(preview)
            provenance["preview"] = {
                **preview_facts, "apng_conversion_command": command,
                "webp_encoding": {"library": "Pillow", "version": pillow_version,
                                  "duration_policy": "200 ms frames; final frame trimmed to the MP4 duration",
                                  "loop": 0,
                                  "quality": 70, "method": 5, "lossless": False},
                "playback_rate": PLAYBACK_RATE, "fps": 5, "width": 480,
            }
            pending.append((pending_preview, preview))
        pending_provenance = directory / provenance_path.name
        pending_provenance.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pending.append((pending_provenance, provenance_path))
        published = []
        try:
            for source, destination in pending:
                # Exclusive creation also protects against another writer racing us.
                with destination.open("xb") as handle:
                    published.append(destination)
                    with source.open("rb") as incoming:
                        shutil.copyfileobj(incoming, handle, length=1024 * 1024)
        except BaseException:
            for path in published:
                path.unlink()
            raise
    print(json.dumps({"output": portable_path(output), "frames": count,
                      "provenance": portable_path(provenance_path),
                      "preview": portable_path(preview) if preview else None}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, zipfile.BadZipFile, RuntimeError) as error:
        raise SystemExit(f"Experiment 02 renderer: {error}") from error
