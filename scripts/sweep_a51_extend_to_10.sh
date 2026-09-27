#!/usr/bin/env bash
# -----------------------------------------------------------------------
# sweep_a51_extend_to_10.sh - DEPRECATED shim (2026-05-27).
#
# This script used to wait for the warmup a51 sweep to finish, then
# launch a51 on seeds {3, 7, 13, 99, 123}. Both behaviours are now in
# sweep_a51_vs_a45.sh via STAGE / WAIT_FOR_INFLIGHT flags.
#
# Equivalent invocation:
#   WAIT_FOR_INFLIGHT=1 STAGE=extend ./scripts/sweep_a51_vs_a45.sh
#
# This shim forwards to that call for backwards compatibility.
# -----------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."

echo "[deprecated] sweep_a51_extend_to_10.sh -> forwarding to:" >&2
echo "  WAIT_FOR_INFLIGHT=1 STAGE=extend ./scripts/sweep_a51_vs_a45.sh" >&2

exec env \
  WAIT_FOR_INFLIGHT=1 \
  STAGE=extend \
  ./scripts/sweep_a51_vs_a45.sh "$@"
