#!/usr/bin/env bash
set -euo pipefail

project="${NIRI_GLASS_LOCK_PROJECT:-$HOME/projects/niri-glass-lock}"
exec nix run --no-write-lock-file "path:$project" -- "$@"
