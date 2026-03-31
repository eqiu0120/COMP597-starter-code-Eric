import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

OUTDIR = "/home/2025/zqiu6/COMP597-starter-code-Eric/GPU_result"
os.makedirs(OUTDIR, exist_ok=True)

csv = sorted(glob.glob("/home/slurm/comp597/students/zqiu6/gpu_measurements/gpu_*.csv"))[-1]
print("Using:", csv)
df = pd.read_csv(csv)

df["timestamp"] = pd.to_datetime(df["timestamp"])
t = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

util = df[" utilization.gpu [%]"].astype(str).str.replace(" %","", regex=False).astype(float)
mem  = df[" memory.used [MiB]"].astype(str).str.replace(" MiB","", regex=False).astype(float)
pwr  = df[" power.draw [W]"].astype(str).str.replace(" W","", regex=False).astype(float)

plt.figure()
plt.plot(t, util)
plt.xlabel("Time (s)")
plt.ylabel("GPU Utilization (%)")
plt.title("GPU Utilization Over Time")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/gpu_util.png", dpi=200)

plt.figure()
plt.plot(t, mem)
plt.xlabel("Time (s)")
plt.ylabel("GPU Memory Used (MiB)")
plt.title("GPU Memory Usage Over Time")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/gpu_mem.png", dpi=200)

plt.figure()
plt.plot(t, pwr)
plt.xlabel("Time (s)")
plt.ylabel("GPU Power (W)")
plt.title("GPU Power Draw Over Time")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/gpu_power.png", dpi=200)

print("Plots saved to:", OUTDIR)