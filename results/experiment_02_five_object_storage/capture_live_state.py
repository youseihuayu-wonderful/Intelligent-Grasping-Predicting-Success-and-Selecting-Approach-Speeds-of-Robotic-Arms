"""Read-only capture of the actual local live viewer; never sends control commands."""

import argparse
import asyncio
import base64
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time

import aiohttp


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8877')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    count = 0
    first = last = None
    before_close = False
    final_image = None
    terminal = False
    started = time.monotonic()
    try:
        async with asyncio.timeout(540):
            async with aiohttp.ClientSession() as client:
                async with client.ws_connect(args.url.rstrip('/') + '/ws', heartbeat=20) as socket:
                    with gzip.open(args.output/'states.jsonl.gz', 'xt', encoding='utf-8') as stream:
                        async for message in socket:
                            if message.type != aiohttp.WSMsgType.TEXT:
                                raise RuntimeError(f'Unexpected WebSocket message: {message.type}')
                            packet = message.json()
                            if packet.get('error'):
                                raise RuntimeError(packet['error'])
                            state = packet.get('state')
                            if state is None:
                                continue
                            count += 1
                            first = state if first is None else first
                            last = state
                            stream.write(json.dumps({'received_utc': datetime.now(timezone.utc).isoformat(),
                                                     'received_elapsed_wall_s': time.monotonic()-started,
                                                     'state': state}, ensure_ascii=False, allow_nan=False)+'\n')
                            if packet.get('image'):
                                final_image = base64.b64decode(packet['image'], validate=True)
                            if not before_close and state.get('all_stored') and not state.get('drawer_closed'):
                                before_close = True
                                save(args.output/'before_closure_state.json', state)
                                if final_image:
                                    (args.output/'before_closure_frame.jpg').write_bytes(final_image)
                                print('All five measured stored in actual live scene; closure still pending', flush=True)
                            if state.get('finished') or state.get('runtime_error'):
                                terminal = True
                                break
                    if not terminal:
                        raise RuntimeError('WebSocket ended before an actual terminal state')
    finally:
        if last is not None:
            save(args.output/'final_state.json', last)
        if final_image is not None:
            (args.output/'last_real_frame.jpg').write_bytes(final_image)
        save(args.output/'capture.json', {
            'experiment_label': 'Experiment 02', 'scope': 'Read-only actual live WebSocket telemetry',
            'control_commands_sent': 0, 'packets_recorded': count, 'terminal_observed': terminal,
            'first_simulation_time_s': None if first is None else first.get('time_s'),
            'last_simulation_time_s': None if last is None else last.get('time_s'),
            'wall_capture_s': time.monotonic()-started,
            'limits': 'Every received state is retained; full physics states are in the separate headless NPZ. JPEGs are only selected actual frames, not a complete live video. Complete published video is explicitly rendered from the separate recorded physical trajectory.'})
    print(json.dumps(last, indent=2, ensure_ascii=False), flush=True)
    if not last or last.get('outcome') is not True:
        raise SystemExit(1)


if __name__ == '__main__':
    asyncio.run(main())
