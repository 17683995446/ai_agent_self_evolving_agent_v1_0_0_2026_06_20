#!/bin/bash
set -e

cd "$(dirname "$0")/.."

python -m src.cli.main stop "$@"
