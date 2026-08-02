#!/bin/bash
# Read-only health check for the N2 local-seniority orbital-optimization batch.
OUT_DIR=/workspace/hamiltonians/N2/minimal_examples/seniority_oo

echo "=== real optimize_symmetries.py processes ==="
ps aux | awk '$0 ~ /python3 (-u )?\/workspace\/optimize_symmetries\.py/ && $0 !~ /awk|grep|bash -c/'
echo
echo "=== real driver process ==="
ps aux | awk '$0 ~ /run_seniority_oo_n2_minimal\.py/ && $0 !~ /awk|grep|bash -c/'
echo
echo "=== thread/rss for real processes ==="
total_threads=0; total_rss_kb=0; nprocs=0
for p in $(ps aux | awk '$0 ~ /python3 (-u )?\/workspace\/optimize_symmetries\.py/ && $0 !~ /awk|grep|bash -c/ {print $2}'); do
  th=$(ls /proc/$p/task 2>/dev/null | wc -l)
  rss=$(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null)
  cpu=$(ps -o %cpu= -p "$p")
  echo "pid $p threads=$th rss=${rss}kB cpu=${cpu}%"
  total_threads=$((total_threads+th)); total_rss_kb=$((total_rss_kb+rss)); nprocs=$((nprocs+1))
done
echo "num_processes=$nprocs total_threads=$total_threads total_rss=$((total_rss_kb/1024))MB (nproc=$(nproc))"
echo
echo "=== system memory ==="
free -h
echo
echo "=== summary.json? ==="
ls -la "$OUT_DIR/summary.json" 2>/dev/null || echo "no summary.json yet"
echo
echo "=== latest log ==="
latest_log=$(ls -t "$OUT_DIR"/logs/*.log 2>/dev/null | head -1)
echo "latest_log=$latest_log"
if [ -n "$latest_log" ]; then
  ls -la "$latest_log"
  echo "--- tail ---"
  tail -10 "$latest_log"
  echo "--- error grep ---"
  grep -iE "traceback|killed|error|memoryerror|oom" "$latest_log" | tail -10
fi
