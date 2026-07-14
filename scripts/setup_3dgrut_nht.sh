#!/usr/bin/env bash
set -Eeuo pipefail

# Backward-compatible entrypoint. All setup logic now lives in setup_all.sh so
# there is only one installation path to maintain and verify.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "setup_3dgrut_nht.sh has been merged into setup_all.sh; forwarding..."
exec "${SCRIPT_DIR}/setup_all.sh" "$@"
