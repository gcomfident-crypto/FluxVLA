# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Summarize official ``trtexec --exportProfile`` JSON artifacts.

The raw CSV retains one row per exported layer record and records an optional
leading profile-count header as source provenance.  The top-10 CSV sums
``timeMs`` only for records with the exact same layer name in the exact same
source artifact; it does not infer, merge, or rename graph nodes.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

_PROVENANCE_FIELDS = (
    'source_json',
    'source_json_sha256',
    'source_profile_count',
    'source_record_index_0based',
)
_PREFERRED_PROFILE_FIELDS = (
    'name',
    'count',
    'timeMs',
    'averageMs',
    'medianMs',
    'percentage',
)
_TOP_FIELDS = (
    'rank',
    'source_json',
    'source_json_sha256',
    'source_profile_count',
    'layer_name',
    'source_record_count',
    'source_record_indices_0based',
    'total_time_ms',
)


class ProfileError(ValueError):
    """An invalid or ambiguous profile artifact."""


def _number(value: Any, field: str, source: Path, index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(
            f'{source}: record {index} field {field!r} must be a number')
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ProfileError(
            f'{source}: record {index} field {field!r} must be finite and '
            'non-negative')
    return number


def _load_profile(
        path: Path
) -> tuple[str, int | None, list[tuple[int, dict[str, Any]]]]:
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise ProfileError(f'cannot read profile JSON {path}: {exc}') from exc
    digest = hashlib.sha256(encoded).hexdigest()
    try:
        value = json.loads(encoded.decode('utf-8-sig'))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f'invalid profile JSON {path}: {exc}') from exc
    if not isinstance(value, list) or not value:
        raise ProfileError(
            f'{path}: trtexec exportProfile JSON must be a non-empty array')

    profile_count = None
    records = []
    for index, record in enumerate(value):
        if not isinstance(record, Mapping):
            raise ProfileError(f'{path}: record {index} must be an object')
        if index == 0 and set(record) == {'count'}:
            count = record['count']
            if (isinstance(count, bool) or not isinstance(count, int)
                    or count <= 0):
                raise ProfileError(
                    f'{path}: leading profile count must be a positive '
                    'integer')
            profile_count = count
            continue
        conflicting = set(record).intersection(_PROVENANCE_FIELDS)
        if conflicting:
            names = ', '.join(sorted(conflicting))
            raise ProfileError(
                f'{path}: record {index} uses reserved field(s): {names}')
        name = record.get('name')
        if not isinstance(name, str) or not name:
            raise ProfileError(
                f'{path}: record {index} field "name" must be a non-empty '
                'string')
        _number(record.get('timeMs'), 'timeMs', path, index)
        records.append((index, dict(record)))
    if not records:
        raise ProfileError(f'{path}: profile contains no layer records')
    return digest, profile_count, records


def _csv_value(value: Any) -> Any:
    if value is None:
        return ''
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProfileError(f'cannot write non-finite CSV value: {value}')
        return format(value, '.17g')
    if isinstance(value, (dict, list)):
        return json.dumps(
            value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return value


def _write_csv(path: Path, fields: Sequence[str],
               rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fields,
                extrasaction='raise',
                lineterminator='\n')
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: _csv_value(row.get(field))
                     for field in fields})
    except OSError as exc:
        raise ProfileError(f'cannot write CSV {path}: {exc}') from exc


def summarize(profile_paths: Sequence[Path],
              output_dir: Path) -> tuple[Path, Path]:
    """Write raw and exact-name aggregate CSVs for profile artifacts."""
    if not profile_paths:
        raise ProfileError('at least one profile JSON path is required')
    seen = set()
    all_profile_fields = set()
    raw_rows = []
    aggregates: dict[tuple[str, str, str], dict[str, Any]] = {}

    for requested_path in profile_paths:
        source = requested_path.expanduser().resolve()
        if source in seen:
            raise ProfileError(f'duplicate profile JSON path: {source}')
        seen.add(source)
        digest, profile_count, records = _load_profile(source)
        for index, record in records:
            all_profile_fields.update(record)
            raw_rows.append({
                'source_json': str(source),
                'source_json_sha256': digest,
                'source_profile_count': profile_count,
                'source_record_index_0based': index,
                **record,
            })
            key = (str(source), digest, record['name'])
            aggregate = aggregates.setdefault(
                key, {
                    'source_json': str(source),
                    'source_json_sha256': digest,
                    'source_profile_count': profile_count,
                    'layer_name': record['name'],
                    'indices': [],
                    'total_time_ms': 0.0,
                })
            aggregate['indices'].append(index)
            aggregate['total_time_ms'] += _number(record['timeMs'], 'timeMs',
                                                  source, index)

    preferred = [
        field for field in _PREFERRED_PROFILE_FIELDS
        if field in all_profile_fields
    ]
    remaining = sorted(all_profile_fields.difference(preferred))
    raw_fields = [*_PROVENANCE_FIELDS, *preferred, *remaining]

    top_rows = []
    source_paths = sorted({row['source_json'] for row in aggregates.values()})
    for source_path in source_paths:
        ranked = sorted((row for row in aggregates.values()
                         if row['source_json'] == source_path),
                        key=lambda row:
                        (-row['total_time_ms'], row['layer_name']))[:10]
        for rank, aggregate in enumerate(ranked, start=1):
            indices = aggregate['indices']
            top_rows.append({
                **{
                    key: value
                    for key, value in aggregate.items() if key != 'indices'
                },
                'rank':
                rank,
                'source_record_count':
                len(indices),
                'source_record_indices_0based':
                ';'.join(map(str, indices)),
            })

    destination = output_dir.expanduser().resolve()
    try:
        destination.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ProfileError(
            f'cannot create output directory {destination}: {exc}') from exc
    raw_path = destination / 'trtexec_layer_profile_raw.csv'
    top_path = destination / 'trtexec_layer_profile_top10.csv'
    _write_csv(raw_path, raw_fields, raw_rows)
    _write_csv(top_path, _TOP_FIELDS, top_rows)
    return raw_path, top_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'profile_json',
        nargs='+',
        type=Path,
        help='One or more official trtexec --exportProfile JSON files.')
    parser.add_argument(
        '--output-dir',
        required=True,
        type=Path,
        help='Directory for the raw-layer and top-10 CSV files.')
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        raw_path, top_path = summarize(args.profile_json, args.output_dir)
    except ProfileError as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                'raw_layer_csv': str(raw_path),
                'top_10_by_total_time_csv': str(top_path),
            },
            indent=2))


if __name__ == '__main__':
    main()
