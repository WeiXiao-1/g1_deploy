#!/usr/bin/env python3
"""Decode GetUpMimic controller traces without MuJoCo, NumPy, SDK or robot access."""
import argparse
from contextlib import ExitStack
import json
import math
from pathlib import Path
import statistics
import struct
import sys


def records(directory, warnings):
    metadata = json.loads((directory / 'manifest.json').read_text())
    if metadata.get('version') != 1 or metadata.get('byte_order') not in ('little', 'big'):
        raise ValueError('Unsupported controller trace format')
    endian = '<' if metadata['byte_order'] == 'little' else '>'
    schemas = {}
    for key, definition in metadata['records'].items():
        fields = []
        for field in definition['fields']:
            if not field['count']:
                continue
            dtype = {'u8': 'Q', 'u4': 'I', 'f4': 'f'}[field['type']]
            unpack = struct.Struct(endian + str(field['count']) + dtype)
            if field['offset'] < 0 or field['offset'] + unpack.size > definition['size']:
                raise ValueError('Invalid trace field bounds')
            fields.append((field, unpack))
        schemas[int(key)] = (definition, fields)
    with (directory / 'records.bin').open('rb') as source:
        if source.read(8) != b'G1TRACE1':
            raise ValueError('Invalid controller trace file header')
        while True:
            header = source.read(8)
            if not header:
                return
            if len(header) != 8:
                warnings.append('Incomplete final record header; preceding records recovered')
                return
            kind, size = struct.unpack(endian + 'II', header)
            if kind not in schemas or size != schemas[kind][0]['size'] or size > 1_000_000:
                raise ValueError('Unknown or inconsistent controller trace record')
            payload = source.read(size)
            if len(payload) != size:
                warnings.append('Incomplete final record payload; preceding records recovered')
                return
            definition, fields = schemas[kind]
            row = {'type': definition['name']}
            for field, unpack in fields:
                values = unpack.unpack_from(payload, field['offset'])
                values = [None if isinstance(v, float) and not math.isfinite(v) else v for v in values]
                row[field['name']] = values[0] if field['count'] == 1 else values
            if kind == 1:
                n = row['observation_size']
                if n > len(row['observation']):
                    raise ValueError('Invalid observation length')
                row['observation'] = row['observation'][:n]
            yield row


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    return dict(count=len(values), mean_ms=statistics.fmean(values),
                p95_ms=values[math.ceil(.95*(len(values)-1))],
                p99_ms=values[math.ceil(.99*(len(values)-1))], maximum_ms=values[-1])


def decode(directory, output):
    directory, output = Path(directory), Path(output)
    metadata = json.loads((directory/'manifest.json').read_text())
    output.mkdir(parents=True, exist_ok=True)
    warnings = []
    counts = {'inference': 0, 'control': 0}
    phase_counts = {}
    timing = {k: [] for k in ('policy_step', 'inference_callback_to_enqueue', 'producer_bookkeeping',
                              'active_control_work', 'active_control_interval', 'input_submit_to_publish', 'result_to_publish')}
    inference_times = {}
    controls = []
    previous = None
    with ExitStack() as stack:
        sinks = {name: stack.enter_context((output/(name+'.jsonl')).open('w')) for name in counts}
        for row in records(directory, warnings):
            kind = row['type']
            counts[kind] += 1
            if kind == 'inference':
                row['relative_ms'] = (row['inference_start_ns']-metadata['monotonic_start_ns'])/1e6
                if row['flags'] & 1:
                    step = (row['inference_end_ns']-row['inference_start_ns'])/1e6
                    gross = (row['enqueue_ready_ns']-row['producer_begin_ns'])/1e6
                    row['policy_step_ms'], row['callback_to_enqueue_ms'] = step, gross
                    timing['policy_step'].append(step)
                    timing['inference_callback_to_enqueue'].append(gross)
                    timing['producer_bookkeeping'].append(max(0, gross-step))
                inference_times[row['id']] = row['state_received_ns']
            else:
                phase = metadata['phase_names'][str(row['phase'])]
                row['phase_name'] = phase
                row['relative_ms'] = (row['loop_start_ns']-metadata['monotonic_start_ns'])/1e6
                row['published'] = bool(row['flags'] & 8)
                row['state_valid'] = bool(row['flags'] & 1)
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                if row['phase'] in (4, 5):
                    timing['active_control_work'].append((row['loop_end_ns']-row['loop_start_ns'])/1e6)
                    if previous and previous['activation_id'] == row['activation_id'] and previous['id']+1 == row['id']:
                        timing['active_control_interval'].append((row['loop_start_ns']-previous['loop_start_ns'])/1e6)
                    previous = {k: row[k] for k in ('activation_id', 'id', 'loop_start_ns')}
                else:
                    previous = None
                if row['published'] and row['inference_id']:
                    timing['input_submit_to_publish'].append((row['write_start_ns']-row['result_input_ns'])/1e6)
                    timing['result_to_publish'].append((row['write_start_ns']-row['result_completed_ns'])/1e6)
                    controls.append((row['inference_id'], row['write_start_ns']))
            sinks[kind].write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
    stats_path = directory/'stats.json'
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else None
    if stats is None:
        warnings.append('No final stats.json: run may still be active or ended abruptly; drop totals unavailable')
    else:
        for kind in counts:
            if counts[kind] != stats[kind+'_written']:
                warnings.append(kind+' record count differs from writer stats (buffered write failure or truncated file)')
    missing = sum(identifier not in inference_times for identifier, _ in controls)
    if missing:
        warnings.append(str(missing)+' published control samples reference missing inference records; inspect drop counters')
    timing['state_receive_to_publish'] = [(published-inference_times[i])/1e6 for i, published in controls if i in inference_times]
    summary = dict(source=str(directory.resolve()), decoded_records=counts, phases=phase_counts,
                   final_recorder_stats=stats, timing={k: distribution(v) for k,v in timing.items()}, warnings=warnings,
                   notes='All timings use host monotonic clocks. policy_step includes observation assembly, ONNX and target processing (plus trace copies when enabled); producer_bookkeeping excludes trace copies inside policy_step. Publication success does not prove drive receipt or execution. Samples are not time-sorted across producer queues; align by IDs and timestamps.')
    (output/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False)+'\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='run_* directory containing manifest.json and records.bin')
    parser.add_argument('--output', type=Path, help='Decoded output directory (default: TRACE_DIRECTORY/decoded)')
    args = parser.parse_args()
    try:
        result = decode(args.directory, args.output or args.directory/'decoded')
    except (OSError, ValueError, KeyError, struct.error) as error:
        parser.exit(1, 'Trace decode failed: '+str(error)+'\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
