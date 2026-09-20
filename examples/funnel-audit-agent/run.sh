#!/usr/bin/env bash
set -euo pipefail
echo "Dry run (no paid call):"
bruv demo funnel-audit --backend simple-jev
echo
echo "Execute the real evaluation:"
bruv demo funnel-audit --backend simple-jev --execute