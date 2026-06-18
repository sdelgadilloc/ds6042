import re, matplotlib
matplotlib.use("Agg")              # headless compute node — write a file, don't open a window
import matplotlib.pyplot as plt

steps, losses = [], []
for line in open("sft.log"):
    m = re.search(r"step\s+(\d+).*?loss:\s*([\d.]+)", line)
    if m:
        steps.append(int(m.group(1)))
        losses.append(float(m.group(2)))

plt.figure(figsize=(7, 4))
plt.plot(steps, losses, lw=1)
plt.xlabel("step"); plt.ylabel("loss")
plt.title(f"SFT · depth 6 · {len(steps)} steps")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("sft_loss.png", dpi=120)
print(f"wrote sft_loss.png ({len(steps)} points)")
