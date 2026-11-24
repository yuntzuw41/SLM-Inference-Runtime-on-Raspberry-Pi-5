import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labels = ["Original C", "Arm NEON"]
values = [2.40, 9.54]

plt.figure(figsize=(7, 4.5))
bars = plt.bar(labels, values, width=0.55)

plt.title("Effective MatVec Computational Performance")
plt.ylabel("Effective Performance (GFLOP/s)")
plt.ylim(0, 11)
plt.grid(axis="y", alpha=0.3)
plt.gca().set_axisbelow(True)

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.15,
        f"{value:.2f}",
        ha="center"
    )

plt.tight_layout()
plt.savefig("figure13_matvec_gflops.png", dpi=200)

print("Figure 13 saved successfully.")
print("File: figure13_matvec_gflops.png")
