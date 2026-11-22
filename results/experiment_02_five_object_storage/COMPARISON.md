# Experiment 02 — continuous five-object storage

Prior demonstrations independently grasped and held five objects with resets between targets; the cabinet demonstration did not place objects or close the drawer. Experiment 02 continuously stores all five, physically closes the drawer, returns home and holds the final state without inter-item resets.

## Protocol differences

| Fact | Previous demonstration | Experiment 02 |
| --- | --- | --- |
| Task | Five independent grasp-and-hold attempts; reset between targets | Five continuous storage placements, drawer closure and home; no inter-item resets |
| Housing width (configured mm) | 300 | 400 |
| Conservative cavity width (configured mm) | 254 | 354 |
| Drawer travel (configured mm) | 160 | 240 |
| Drawer joint position (baseline initial / Experiment 02 final), m | -0.16 | 3.604782546323508e-08 |
| Final drawer joint velocity (measured m/s) | Not recorded in cabinet baseline | -8.878313744643389e-09 |
| Verified all-five closed-storage hold (simulation s) | Not performed | 3.001999999999891 |
| Scene resets during continuous storage mission | No continuous storage mission | 0 |

Dimensions are configuration specifications, not measured object poses. Previous joint position is its recorded initially open position, not a storage end state. Null means unavailable.

## Recorded result
The explicitly supplied final verification passes all five strict physical predicates for every object: inside, floor-supported, detached, stable and stored. The drawer is closed, the hand is empty, the verified hold is at least 3 simulation seconds, there are zero inter-item scene resets and no numerical warnings.

Development runs remain under `runs/`; the explicit successful run is under `final/`. Original files are byte-for-byte except `model.mjb`, stored losslessly as deterministic gzip with a `model_storage.json` descriptor. Original verification JSON and its logical model hash remain unchanged. File sizes, stored/logical SHA256 checks, NPZ fields/shapes and simulation sample spans are in `manifest.json`. Missing metadata is identified there, never rewritten.

## Evidence and limits
Prior records are copied under `baselines/` and compared in `comparison.json`. Independent grasp-and-hold demonstrations are not equivalent to continuous storage. No success-rate or speed improvement is inferred, and simulation clocks are not compared to wall clocks.

Existing classifier datasets are unchanged by this archive and were not newly collected or trained here. Historical frozen artifacts below match the prior recorded hashes; this is not a new classifier evaluation.

Historical classifier batch comparison is documented separately in `dataset_batch_comparison.json`; those batches were not collected by this nominal storage task.

Optional check logs are copied verbatim under `check_logs/`; their actual content, including any failures, is authoritative. No pass status or missing log is invented.

Partial source snapshot only. Reproduce from the repository checkout with uv.lock; this snapshot is not standalone.

## Detailed records

- [Initialization and controller changes](protocol_changes.json).
- [Observed checks and retained failures](check_history.json).
- [Read-only live capture metadata](runs/experiment02_live_02/capture.json) and [actual final telemetry](runs/experiment02_live_02/final_state.json).
- [Full English MP4](../../media/experiment-02-five-object-storage-en.mp4) and [render provenance](../../media/experiment-02-five-object-storage-en.provenance.json).
- Run and replay commands are in the [project README](../../README.md).
