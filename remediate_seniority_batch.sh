#!/bin/bash
# Kill any stray N2 local-seniority batch processes and relaunch cleanly.
# Only invoked by the monitoring routine when it detects a broken state
# (oversubscription, memory exhaustion risk, a crashed process, or log errors).
set -e
pkill -f run_seniority_oo_n2_minimal.py 2>/dev/null || true
pkill -f optimize_symmetries.py 2>/dev/null || true
sleep 2
cd /workspace
nohup python3 run_seniority_oo_n2_minimal.py > hamiltonians/N2/minimal_examples/seniority_oo/driver.log 2>&1 &
echo "relaunched pid $!"
