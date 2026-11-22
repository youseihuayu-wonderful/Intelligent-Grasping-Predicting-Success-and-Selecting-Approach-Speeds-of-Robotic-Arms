# Two different simulation experiments

Simulation code, runtime assets, recorded evidence and videos belong in this original repository. The [lightweight course ML repository](https://github.com/youseihuayu-wonderful/intelligent-grasping-course-project) links here rather than duplicating this runtime. Neither experiment is driven by that repository's classifier baseline.

## Tasks and historical evidence

### Experiment01: independent five-object grasp and hold

Bowl, bottle, spoon, glass and pen coexist. Each target attempt starts with a reset of the complete layout; the robot grasps and continuously holds that target while the other four remain on the table. This is **five independent attempts**, not one continuous storage mission.

Existing, immutable records at the Experiment02 release commit:

- [First five-object verification](https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/results/five_object_tabletop/verification.json).
- [Earlier initial-layout verification, including the failed bottle attempt](https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/results/five_object_tabletop/initial_layout_verification.json). The corrected record does not erase this failure.
- [Subsequent open-cabinet/independent-grasp verification](https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/results/five_object_cabinet_20261122/verification.json). Its scope explicitly excludes placement and drawer closure.

The [original six source files](../archives/experiment_01_sources/) have been recovered from stored source-write payloads and successful recorded edits. **Every file exactly matches `viewer_source_sha256` in the first verification**, including the tabletop model, server, three frontend files and tests. The [recovery manifest](../archives/experiment_01_sources/manifest.json) records their identities. No source was inferred from animation.

Use `restore_experiment01.py` below to assemble these original files over the verified shared runtime in a new directory. This restores the original interface/source, not an unavailable original compiled-model or operating-system snapshot. The current-layout `run_tabletop_grasps.py` is a separate additional reproduction: its cabinet is 400 mm wide with 240 mm travel versus the later archived open-cabinet baseline's 300/160 mm, and some poses changed. Do not confuse that current rerun with the original source mode. Historical raw records remain unchanged.

### Experiment02: continuous five-object drawer storage

One physical mission stores all five objects, releases them onto the drawer floor, closes the drawer by verified robot contact, returns home, and verifies the final state for at least three simulation seconds. There are no inter-item scene resets. Success requires every object's inside/supported/detached/stable/stored predicates, plus the mission's terminal checks; it is not equivalent to a successful grasp label.

The existing release tag is [`experiment-02-five-object-storage`](https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/tree/experiment-02-five-object-storage), commit `7a2c5642113d0d3b9c5b5d5205feda843b2c043a`:

- [Final physical verification](https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/results/experiment_02_five_object_storage/final/verification.json).
- [Trajectory/model archive and manifest](https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/tree/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/results/experiment_02_five_object_storage).
- [English MP4](https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/media/experiment-02-five-object-storage-en.mp4), [direct video download](https://raw.githubusercontent.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/media/experiment-02-five-object-storage-en.mp4), and [render provenance](https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/media/experiment-02-five-object-storage-en.provenance.json).

`tabletop_storage.py --output ...` is the actual storage CLI. Its physical source is unchanged by this delivery. The old `source_snapshot/` is explicitly partial: source/video publication alone did not guarantee that every ignored vendor mesh was available in a fresh checkout.

## Source capsule: restore before running

The generated [runtime capsule directory](../assets/simulation_delivery_v1/) contains `runtime.tar.gz` and `manifest.json`. The capsule is a deterministic source distribution, not a compiled-model snapshot or new experimental result. It packages:

- Inputs listed in the recorded final Experiment02 `source_sha256` and study `environment.json` `input_sha256`, requiring exact recorded hashes.
- Transitive local Python imports and XML mesh/texture/include references, with independently recorded supplement hashes. Imported legacy ML/planning modules are included because the current runtime imports them; fixed grasp/storage execution requires no API key or external planner call.
- The independent CLI, existing storage/live-viewer CLI, three study web files, unchanged `pyproject.toml`/`uv.lock`, and original provenance records.
- Applicable bundled `vendor/libero/LICENSE` (MIT) and `vendor/panda/LICENSE` (Apache-2.0), preserved verbatim. Asset provenance remains LIBERO and MuJoCo Menagerie Panda; this delivery does not relicense them.

No `.env`, virtual environment, private data, unrelated vendor checkout or historical result directory is recursively swept into the capsule. Every member and the compressed archive must be smaller than 100 MiB. The manifest preserves logical source SHA256 values; normalized tar metadata and gzip headers do not alter those source bytes. The original archive's logical compiled-model hashes still refer to its original model, not this source capsule.

XML dependency expansion starts from the task model roots listed in `model_dependency_roots` (including the legacy cabinet needed by the base scene), not every unused variant in the historical input inventory. For example, the preserved unused `wine_bottle_red_main.xml` refers to an absent `template.png`; the loaded `wine_bottle.xml` instead uses the available label/cork textures. The unused variant is provenance, not a supported alternative scene. All references of the selected runtime roots must exist; missing required assets abort the build.

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). From a fresh checkout of this repository's delivery revision (not only the older release tag):

```sh
git clone https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms.git
cd Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms
ORIGINAL=$(pwd -P)
python3 simulation_delivery.py verify --capsule "$ORIGINAL/assets/simulation_delivery_v1"
RESTORED=$(mktemp -d)
# Canonicalize the temporary root: restoration deliberately rejects symlink ancestors.
RESTORED=$(cd "$RESTORED" && pwd -P)
python3 simulation_delivery.py restore \
  --capsule "$ORIGINAL/assets/simulation_delivery_v1" --destination "$RESTORED"
python3 simulation_delivery.py verify \
  --capsule "$ORIGINAL/assets/simulation_delivery_v1" --destination "$RESTORED"
cd "$RESTORED"
uv sync --locked
```

Verification checks the compressed archive and every source member. Restore rejects absolute/traversing paths, links, special members, duplicates and unlisted files. It preflights existing destinations and refuses differing files; identical existing files are harmless. Use a new canonical temporary directory for isolation. Hashes establish consistency with the checked-out manifest, not authenticity independent of the GitHub revision you trust. No workstation vendor paths are required after restoration.

### Isolated current-scene smoke (actual physics)

Run from the restored directory, with no `PYTHONPATH` pointing at the original checkout:

```sh
env -u PYTHONPATH uv run --locked python -c 'from grasp_tabletop import FiveObjectSimulation; s = FiveObjectSimulation("bowl"); s.step(.02); objects = s.object_states(); assert len(objects) == 5; assert not s.warning_names(); print({"time_s": float(s.data.time), "objects": objects}); s.close()'
```

This compiles the restored XML/assets and advances real dynamics; it is not a success test for either mission.

### Restore the original Experiment01 source and viewer

From the original repository checkout, restore into a **different new or empty directory**:

```sh
FIRST=$(mktemp -d)
FIRST=$(cd "$FIRST" && pwd -P)
cd "$ORIGINAL"
python3 restore_experiment01.py --destination "$FIRST"
cd "$FIRST"
uv sync --locked
uv run --locked python -m unittest discover -s tests -p test_grasp_tabletop.py
uv run --locked python serve_grasp_study.py --port 8878
```

Open `http://127.0.0.1:8878/` for the recovered original independent-grasp interface. It is distinct from the continuous-storage viewer on port 8877. The restorer verifies all six old hashes against the original verification, checks all other shared runtime files, and writes `EXPERIMENT01_RESTORED.json`. It refuses a populated destination rather than replacing an existing working tree.


### Current-layout independent-grasp reproduction

```sh
cd "$RESTORED"
uv run --locked python run_tabletop_grasps.py --output results/current_independent_grasps
# Optional single target:
uv run --locked python run_tabletop_grasps.py --targets bottle --output results/current_bottle_grasp
```

Both commands require a new output directory. The five-target run creates per-target JSON, sampled traces and `verification.json` with explicit current-scene scope, settings, source hashes, predicate hold duration and observed outcomes. Each target gets a fresh complete scene; no reset or pose command occurs during an attempt. These outputs are new task reproductions, never edits to the published historical records.

For a real live native viewer on Linux/Windows:

```sh
uv run --locked python run_tabletop_grasps.py --viewer --output results/current_independent_viewer
```

On macOS, MuJoCo's passive viewer requires its Python launcher:

```sh
uv run --locked mjpython run_tabletop_grasps.py --viewer --output results/current_independent_viewer
```

The native viewer follows actual running dynamics. Closing it early fails the attempt; it does not synthesize completion.

### Continuous storage reproduction and browser viewer

```sh
cd "$RESTORED"
uv run --locked python tabletop_storage.py --output results/current_continuous_storage
uv run --locked python serve_grasp_study.py --port 8877
```

Open `http://127.0.0.1:8877/` for the current **continuous storage** viewer. It is not the earlier independent-grasp viewer. It advances while a browser client is connected, exposes actual telemetry at `/api/state`, and supports pause/restart/camera controls. The viewer targets 2× playback; rendering load can reduce actual throughput. This does not change the physics timestep or controller commands. Rendering needs a working graphics backend (on headless Linux, a suitable EGL installation and `MUJOCO_GL=egl`); pure physics CLI commands do not require a browser or renderer. The storage CLI requires a new output directory and records its actual trajectory, compiled model and physical verification; a failed mission exits nonzero.

## Maintainer-only capsule rebuild

A fresh public checkout should **restore the published capsule**, not attempt to rebuild it from absent ignored assets. To rebuild on a complete source checkout with all recorded vendor inputs:

```sh
cd "$ORIGINAL"
python3 simulation_delivery.py build --output assets/simulation_delivery_v1_local
python3 simulation_delivery.py verify --capsule assets/simulation_delivery_v1_local
```

The output directory must not exist. A missing or mismatched recorded source aborts the build; there is no download substitution or permissive fallback. Identical input bytes produce identical archive and manifest bytes. Building verifies archive consistency, not physics; keep source packaging and physical outcomes distinct.

## Keep older data and demos separate

The [five-object classifier batch](https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/tree/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/results/grasp_classifier_five_objects_20261121) contains the existing 3,840-row CSV and its recorded environment. Its labels are independent grasp-and-hold observations under that batch's protocol, not drawer-storage completion. [Historical batch comparison](https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/results/experiment_02_five_object_storage/dataset_batch_comparison.json) identifies earlier batches separately. Experiment02 is the recorded engineering release above, not a new training-data batch.

Earlier [black-bowl grasp demo](https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/media/grasp-demo-en.mp4) and [sorting/contact-feedback demo](https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms/blob/7a2c5642113d0d3b9c5b5d5205feda843b2c043a/media/gpt-sorting-contact-feedback-demo-en.mp4) are separate public demonstrations, not substitutes for Experiment01 evidence. No labels are inferred from videos. Nominal demonstrations provide no population success-rate estimate, no classifier improvement claim and no sim-to-real validation; objects remain rigid simulation approximations.

## Observed delivery checks

- [Source build](../results/simulation_delivery_validation_v1/capsule_build.log) and [isolated restoration](../results/simulation_delivery_validation_v1/capsule_restore.log): 168 runtime files packaged and hash-verified.
- [Actual restored-scene smoke](../results/simulation_delivery_validation_v1/restored_scene_smoke.json): all five dynamic objects loaded from the new temporary runtime, with no MuJoCo warnings.
- [Current independent-grasp verification](../results/simulation_delivery_validation_v1/independent_grasp_validation/verification.json): five real attempts passed the two-second hold and undisturbed-neighbor checks. This is a nominal task reproduction, not a population success-rate estimate.
- [Original Experiment01 source tests](../results/simulation_delivery_validation_v1/original_experiment01_tests.log): five tests passed after restoring the six exact original sources. [Restoration guard evidence](../results/simulation_delivery_validation_v1/original_experiment01_restoration_check.json) also verifies refusal to overwrite a populated runtime.
- Experiment02's complete physical mission, model snapshot, trajectory and movie remain the immutable recorded release linked above. Packaging does not rerun, overwrite or relabel that experiment.
