import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = {}

with open("results/runtime_benchmark.csv", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        run = int(row["Round"])
        runtime = row["Runtime"]
        speed = float(row["Tokens_per_sec"])

        if run not in data:
            data[run] = {}

        data[run][runtime] = speed

rounds = sorted(data)
scalar = [data[r]["scalar"] for r in rounds]
neon = [data[r]["neon"] for r in rounds]

plt.figure(figsize=(8, 4.5))
plt.plot(rounds, scalar, marker="o", linewidth=2,
         label="Original C")
plt.plot(rounds, neon, marker="s", linewidth=2,
         label="Arm NEON")

plt.title("Full Runtime Inference Throughput")
plt.xlabel("Benchmark Run")
plt.ylabel("Throughput (tokens/s)")
plt.xticks(rounds)
plt.ylim(0, 230)
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("figure10_runtime_benchmark.png", dpi=200)

print("Figure 10 saved successfully.")
print("File: figure10_runtime_benchmark.png")
