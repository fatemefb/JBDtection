#!/usr/bin/env python3
"""Run focused Graphify commands with a visible, bounded context window."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("query", "explain", "path", "affected"))
    parser.add_argument("symbols", nargs="+")
    parser.add_argument("--budget", type=int, default=1200,
                        help="Graphify's approximate token budget for queries")
    parser.add_argument("--max-chars", type=int, default=4800,
                        help="Strict character limit for stdout, including truncation notice")
    parser.add_argument("--offset", type=int, default=0,
                        help="Read a later character window of a large result")
    args = parser.parse_args()
    expected = 2 if args.operation == "path" else 1
    if len(args.symbols) != expected:
        parser.error(f"{args.operation} needs {expected} quoted argument(s)")
    if args.budget < 1 or args.max_chars < 256 or args.offset < 0:
        parser.error("budget must be positive, max-chars >= 256, offset >= 0")
    executable = shutil.which("graphify")
    if not executable:
        parser.exit(1, "graphify is not on PATH\n")
    command = [executable, args.operation, *args.symbols]
    if args.operation == "query":
        command += ["--budget", str(args.budget)]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    output = result.stdout
    if result.returncode:
        output += result.stderr
    remaining = output[args.offset:]
    if len(remaining) > args.max_chars:
        # Reserve room for the notice, so it is part of the strict limit.
        window = args.max_chars - 200
        chunk = remaining[:window]
        boundary = chunk.rfind("\n")
        if boundary > 0:
            chunk = chunk[:boundary + 1]
        next_offset = args.offset + len(chunk)
        output = (chunk + f"\n[Context truncated at {next_offset}/{len(output)} characters. "
                  f"Narrow the symbol or repeat with --offset {next_offset}. "
                  "A graph relationship is not proof of runtime behavior.]\n")
    else:
        output = remaining
    sys.stdout.write(output)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
