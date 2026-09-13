# Bundled engine

This folder receives the self-contained Python + OpenCASCADE engine that makes the
Mac app work offline with no Python installed on the user's machine.

Produce it with:

    ./scripts/bundle-engine.sh            # arm64 (Apple silicon) from PyPI
    ./scripts/bundle-engine.sh --arch x86_64
    ./scripts/bundle-engine.sh --source ..   # install the repo's own source tree

The result lands in `Engine/engine/` (gitignored) and is copied into
`CADVERT.app/Contents/Resources/engine` by the macOS build. If the folder is absent
the app still builds; the Mac app then looks for an engine at `~/.cadvert/engine`,
a custom Python path from Settings, or falls back to connecting to a remote server.
