#!/bin/sh
# Compiles a Qt Quick ShaderEffect fragment shader to a .qsb package.
#
# Qt 6 only loads shaders that the Qt Shader Tools' qsb has preprocessed, so the shell runs
# this before pointing ShaderEffect.fragmentShader at the result. qsb usually reaches us
# through PATH (quickshell wrappers built with qtshadertools in PATH); otherwise, on NixOS,
# fall back to the newest qtshadertools in the store, since a shell spawned by niri inherits
# the session PATH rather than a quickshell wrapper's.
set -eu

src=$1
out=$2

# Nothing to do when the compiled shader is newer than its source.
if [ -f "$out" ] && [ "$out" -nt "$src" ]; then
    exit 0
fi

qsb=$(command -v qsb || true)
if [ -z "$qsb" ]; then
    qsb=$(ls -d /nix/store/*qtshadertools-*/bin/qsb 2>/dev/null \
        | while read -r candidate; do
            version=${candidate#*-qtshadertools-}
            printf '%s %s\n' "${version%%/*}" "$candidate"
        done \
        | sort -V \
        | tail -n 1 \
        | cut -d' ' -f2-)
fi
if [ -z "$qsb" ]; then
    echo "no qsb found; install Qt Shader Tools or put qsb on PATH" >&2
    exit 1
fi

mkdir -p "$(dirname "$out")"
exec "$qsb" --qt6 -o "$out" "$src"
