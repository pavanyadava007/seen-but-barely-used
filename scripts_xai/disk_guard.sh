#!/usr/bin/env bash
# Abort a phase if the root filesystem has less than MIN_FREE_GB free (default 40).
# Sourced by every scripts_xai/p*.sh before any heavy output is written.
MIN_FREE_GB=${MIN_FREE_GB:-40}
free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
  echo "DISK GUARD: only ${free_gb} GB free (< ${MIN_FREE_GB} GB); aborting." >&2
  exit 3
fi
echo "disk guard: ${free_gb} GB free"
