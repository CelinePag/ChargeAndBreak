"""
fig_arch.py — architecture diagram of the trained policy
========================================================
Kept as its own module (rather than inside plotML.py) because it is the one
figure that draws nothing from the run data: it is a schematic of the network,
and every number in it is the REAL one — 143 inputs, a 2x128 shared trunk,
three heads, 36,621 parameters — rather than a generic MLP cartoon.

Two things the diagram exists to make visible, because they are what makes
this network unusual and they are invisible in a layer list:

  * the feasibility MASK, which sets illegal actions to -inf BEFORE the
    softmax, so they receive probability exactly zero and gradient exactly
    zero — they are absent from the model's world, not merely discouraged;
  * the VALUE head, which exists only during PPO fine-tuning and shares the
    trunk with the actor, because the features that predict what to do are the
    same ones that predict how much cost remains.

Rendered by `python ML/code/plotML.py arch` (it is registered there), or
standalone with `python ML/code/fig_arch.py`.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

C_IN, C_H, C_OUT, C_VAL = "#0072B2", "#444444", "#D55E00", "#009E73"


def fig_arch(rows=None, instance=None, save=None):
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    ax.set_xlim(-0.35, 10.6)
    ax.set_ylim(-0.35, 5.6)
    ax.axis("off")

    def column(x, n_show, y_top, y_bot, color, r=0.085):
        ys = np.linspace(y_top, y_bot, n_show)
        for y in ys:
            ax.add_patch(mp.Circle((x, y), r, fc="white", ec=color, lw=1.2,
                                   zorder=3))
        ax.text(x, y_bot - 0.30, r"$\vdots$", ha="center", va="center",
                fontsize=11, color=color)
        return ys

    # ── input block, grouped by what the features mean ───────────────────
    ax.add_patch(mp.FancyBboxPatch((0.52, 3.42), 0.96, 1.28,
                                   boxstyle="round,pad=0.04",
                                   fc=C_IN, alpha=0.13, ec="none"))
    ax.add_patch(mp.FancyBboxPatch((0.52, 1.28), 0.96, 1.94,
                                   boxstyle="round,pad=0.04",
                                   fc=C_IN, alpha=0.06, ec="none"))
    y_in = column(1.0, 7, 4.40, 1.60, C_IN)
    y_h1 = column(3.4, 6, 4.20, 1.80, C_H)
    y_h2 = column(5.5, 6, 4.20, 1.80, C_H)

    for a in y_in:
        for b in y_h1:
            ax.plot([1.0, 3.4], [a, b], lw=0.22, color=C_IN, alpha=0.22,
                    zorder=1)
    for a in y_h1:
        for b in y_h2:
            ax.plot([3.4, 5.5], [a, b], lw=0.22, color=C_H, alpha=0.22,
                    zorder=1)

    ax.text(1.0, 5.02, r"state  $x \in \mathbb{R}^{143}$", ha="center",
            fontsize=10, weight="bold", color=C_IN)
    ax.text(0.42, 4.06, "23 dashboard\nSoC, HoS clocks,\nspread $h$, node type",
            ha="right", va="center", fontsize=7.3, color=C_IN)
    ax.text(0.42, 2.25,
            "120 look-ahead\n$K{=}20$ nodes $\\times$ 6\nleg, energy, charger,\n"
            "customer, queue, slack",
            ha="right", va="center", fontsize=7.3, color=C_IN)

    ax.annotate("", xy=(3.05, 5.02), xytext=(1.55, 5.02),
                arrowprops=dict(arrowstyle="->", lw=1.0, color="#666666"))
    ax.text(2.30, 5.16, r"$z=(x-\mu)\oslash\sigma$", ha="center", fontsize=8.2,
            color="#666666")

    # ── trunk ────────────────────────────────────────────────────────────
    ax.plot([3.10, 5.80], [4.62, 4.62], lw=1.0, color=C_H, alpha=0.55)
    ax.text(4.45, 4.74, "shared trunk", ha="center", fontsize=9.2,
            style="italic", color=C_H)
    for x, lab, prm in ((3.4, "Linear $143\\!\\to\\!128$\n+ ReLU", "18,432"),
                        (5.5, "Linear $128\\!\\to\\!128$\n+ ReLU", "16,512")):
        ax.text(x, 1.12, lab, ha="center", va="top", fontsize=8.4)
        ax.text(x, 0.44, f"{prm} params", ha="center", va="top", fontsize=7.4,
                color="#777777")

    # ── heads ────────────────────────────────────────────────────────────
    def head(y, title, sub, color, dashed=False):
        ax.annotate("", xy=(7.02, y), xytext=(5.68, 3.0),
                    arrowprops=dict(arrowstyle="->", lw=1.15, color=color,
                                    linestyle="--" if dashed else "-",
                                    shrinkA=6, shrinkB=2))
        ax.text(7.12, y + 0.15, title, fontsize=8.5, weight="bold", color=color)
        ax.text(7.12, y - 0.20, sub, fontsize=7.2, color="#555555")

    head(4.30, "action head   Linear $128\\!\\to\\!12$",
         r"1,548 params  $\rightarrow$  logits $\ell$", C_OUT)
    head(1.95, "duration head   Linear $128\\!\\to\\!1$",
         r"129 params  $\rightarrow$  softplus  $\rightarrow$  $\tau_c \geq 0$",
         C_OUT)
    head(0.80, "value head   Linear $128\\!\\to\\!1$",
         "PPO fine-tuning only (critic)", C_VAL, dashed=True)

    # ── masked softmax strip: the part a layer list cannot show ──────────
    ax.text(7.12, 3.68, "feasibility mask, then softmax:", fontsize=7.6,
            color="#555555")
    allowed = [1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
    best = 6
    for j, ok in enumerate(allowed):
        x0 = 7.12 + j * 0.245
        fc = (C_OUT if j == best else "white") if ok else "#DCDCDC"
        ax.add_patch(mp.Rectangle((x0, 3.14), 0.21, 0.35, fc=fc,
                                  ec="#999999" if ok else "#C8C8C8", lw=0.8))
    ax.text(7.12, 2.90,
            r"grey $=-\infty$ (illegal)     orange $=\arg\max$",
            fontsize=6.9, color="#777777")

    ax.text(5.1, -0.22,
            "36,621 parameters      one forward pass $\\approx$ 0.5 ms      "
            "no solver at deployment",
            ha="center", fontsize=8.8, weight="bold")

    fig.tight_layout()
    out = save or os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "figures", "ml_architecture")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(out)}.png|pdf")


if __name__ == "__main__":
    fig_arch()
