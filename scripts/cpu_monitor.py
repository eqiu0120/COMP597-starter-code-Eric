#!/usr/bin/env python3
"""
cpu_monitor.py
--------------
Logs CPU utilization (%) to a CSV file at a fixed interval.
Runs as a background process launched by job.sh alongside nvidia-smi.

Usage:
    python cpu_monitor.py <output_csv> [interval_seconds]
"""
import csv
import sys
import time

import psutil

output_csv = sys.argv[1]
interval   = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "cpu_percent", "ram_used_mb"])
    while True:
        ts       = time.strftime("%Y/%m/%d %H:%M:%S")
        cpu_pct  = psutil.cpu_percent(interval=None)
        ram_mb   = psutil.virtual_memory().used / 1e6
        writer.writerow([ts, cpu_pct, ram_mb])
        f.flush()
        time.sleep(interval)
