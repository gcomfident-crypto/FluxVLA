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
"""Strictly identify the TensorRT version reported by ``trtexec --help``."""

from __future__ import annotations

import argparse
import json
import re
import subprocess

_BANNER_PATTERN = re.compile(
    r'^&&&& RUNNING TensorRT\.trtexec '
    r'\[TensorRT v(?P<encoded>[1-9][0-9]{3,7})\]'
    r'(?: \[[^\]\r\n]+\])? # (?P<command>[^\r\n]+)$', re.MULTILINE)


def parse_trtexec_help(output: str) -> dict:
    matches = list(_BANNER_PATTERN.finditer(output))
    if len(matches) != 1:
        raise ValueError(
            'Expected exactly one trtexec TensorRT banner in --help output')
    match = matches[0]
    command = match.group('command').strip()
    if re.fullmatch(r'\S+ --help', command) is None:
        raise ValueError('trtexec banner does not describe a --help probe')
    encoded = match.group('encoded')
    number = int(encoded)
    major = number // 10000
    minor = number // 100 % 100
    patch = number % 100
    if major <= 0:
        raise ValueError(f'Invalid encoded TensorRT version: {encoded}')
    return {
        'banner': match.group(0),
        'encoded': encoded,
        'label': f'TensorRT v{encoded}',
        'version': f'{major}.{minor}.{patch}',
    }


def probe_trtexec(executable: str) -> dict:
    process = subprocess.run([executable, '--help'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             text=True,
                             encoding='utf-8',
                             check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f'trtexec --help exited with status {process.returncode}')
    return parse_trtexec_help(process.stdout)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('executable')
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(json.dumps(probe_trtexec(args.executable), sort_keys=True))


if __name__ == '__main__':
    main()
