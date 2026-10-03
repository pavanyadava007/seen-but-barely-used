"""Explainer video for "Seen but Barely Used" (about 100 s, 1280x720, 24 fps).

Every number shown is taken from the paper (paper/main.tex, generated macros). Images come from paper/figures.
Usage:  python docs/video/make_video.py            -> docs/video/explainer.mp4 and docs/img/explainer.gif
Needs matplotlib, Pillow, numpy and an ffmpeg binary (FFMPEG env var, imageio-ffmpeg, or ffmpeg on PATH).
"""
import os
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIG = os.path.join(REPO, "paper", "figures")
W, H, FPS = 1280, 720, 24

# palette (validated: blue/orange pass lightness, chroma, CVD and contrast checks on the light surface)
BG, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29c"
VEH, PED, GRAY = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** 3


def new_fig():
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    return fig


def header(fig, kicker, title):
    fig.text(0.06, 0.90, kicker.upper(), fontsize=13, color=PED, weight="bold")
    fig.text(0.06, 0.82, title, fontsize=24, color=INK, weight="bold")


def caption(fig, text):
    fig.text(0.5, 0.055, text, fontsize=17, color=INK2, ha="center", va="center", wrap=True)


def image_ax(fig, path, rect):
    ax = fig.add_axes(rect)
    ax.imshow(Image.open(path))
    ax.axis("off")
    return ax


def hbars(ax, labels, values, colors, t, unit="", xmax=None, fmt="{:.1f}"):
    """Horizontal bars growing with t (0..1); value labels in ink, not series colour."""
    y = np.arange(len(labels))[::-1]
    xmax = xmax or max(values) * 1.25
    for yi, v, c, lab in zip(y, values, colors, labels):
        w = v * ease(t)
        ax.barh(yi, w, height=0.55, color=c, edgecolor=BG, linewidth=2)
        ax.text(-0.01 * xmax, yi, lab, ha="right", va="center", fontsize=15, color=INK)
        if t > 0.6:
            ax.text(w + 0.015 * xmax, yi, fmt.format(v) + unit, va="center", fontsize=16, color=INK, weight="bold")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="x", labelsize=11)
    ax.set_facecolor(BG)


# ----------------------------------------------------------------------------------------------- scenes
# each scene: (seconds, animated fraction, draw(fig, t))

def s_title(fig, t):
    fig.text(0.5, 0.62, "Seen but Barely Used", fontsize=46, weight="bold", ha="center")
    fig.text(0.5, 0.52, "Does a self-driving AI actually use the pedestrians it sees?", fontsize=22, color=INK2,
             ha="center")
    fig.text(0.5, 0.36, "Pavan Yadav Annappa", fontsize=18, ha="center")
    fig.text(0.5, 0.31, "github.com/pavanyadava007/seen-but-barely-used", fontsize=14, color=MUTED, ha="center")


def s_question(fig, t):
    header(fig, "The problem", "One network, from cameras to a planned path")
    steps = ["6 cameras", "one neural network", "planned path (3 s)"]
    for i, s in enumerate(steps):
        a = ease(t * 3 - i)
        x = 0.12 + i * 0.30
        box = FancyBboxPatch((x, 0.45), 0.22, 0.14, boxstyle="round,pad=0.01,rounding_size=0.02",
                             transform=fig.transFigure, fc=BG, ec=INK if i != 1 else PED, lw=2.5, alpha=a)
        fig.patches.append(box)
        fig.text(x + 0.11, 0.52, s, ha="center", va="center", fontsize=17, alpha=a,
                 weight="bold" if i == 1 else "normal")
        if i < 2 and a > 0.5:
            fig.patches.append(FancyArrowPatch((x + 0.235, 0.52), (x + 0.285, 0.52), transform=fig.transFigure,
                                               arrowstyle="-|>", mutation_scale=22, color=INK2, lw=2))
    if t > 0.75:
        fig.text(0.5, 0.30, "A driving score says whether the car drove well,\nnot whether it took pedestrians "
                 "into account.", ha="center", fontsize=19, color=INK2)
    caption(fig, "We open the network and follow every pedestrian from image to plan.")


def s_pipeline(fig, t):
    header(fig, "How SparseDrive works", "900 object notes, a planner, a memory, a collision check")
    nodes = [("Camera\nimages", 0.11, 0.52), ("900 object notes\n(what, where, how)", 0.305, 0.52),
             ("Planner reads\nthe top 50", 0.50, 0.52), ("Collision\ncheck", 0.695, 0.52),
             ("Planned\npath", 0.89, 0.52), ("Memory of\nearlier frames", 0.40, 0.27)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 1), (5, 2)]
    shown = int(ease(t) * (len(nodes) + 0.999))
    for i, (lab, x, y) in enumerate(nodes[:shown]):
        hl = i == 2
        fig.patches.append(FancyBboxPatch((x - 0.07, y - 0.07), 0.14, 0.14,
                                          boxstyle="round,pad=0.01,rounding_size=0.02", transform=fig.transFigure,
                                          fc=BG, ec=PED if hl else INK, lw=2.5))
        fig.text(x, y, lab, ha="center", va="center", fontsize=14, weight="bold" if hl else "normal")
    for a, b in edges:
        if a < shown and b < shown:
            (_, x0, y0), (_, x1, y1) = nodes[a], nodes[b]
            if y0 == y1:
                p0, p1 = (x0 + 0.073, y0), (x1 - 0.073, y1)
            else:
                p0, p1 = (x0 + (0.03 if x1 > x0 else -0.03), y0 + 0.07), (x1, y1 - 0.072)
            fig.patches.append(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>",
                                               mutation_scale=20, color=INK2, lw=2))
    caption(fig, "Question: when a note says \"pedestrian\", how much does that change the path?")


def s_scene(fig, t):
    header(fig, "What the model sees", "A real scene, pedestrians coloured by confidence")
    image_ax(fig, os.path.join(FIG, "f1_frame.jpg"), [0.04, 0.16, 0.92, 0.60])
    caption(fig, "Most pedestrians are found. Red = missed (here: one behind glass).")


def s_sees(fig, t):
    header(fig, "Finding 1", "It sees pedestrians clearly")
    for i, (big, small) in enumerate([("1.6 %", "of detected pedestrians\nare mistaken for another class"),
                                      ("20", "internal building blocks\nrespond specifically to pedestrians")]):
        a = ease(t * 2 - i * 0.5)
        fig.text(0.08, 0.60 - i * 0.24, big, fontsize=52, weight="bold", color=PED, alpha=a)
        fig.text(0.27, 0.62 - i * 0.24, small, fontsize=15, color=INK2, va="center", alpha=a)
    ax = fig.add_axes([0.66, 0.20, 0.30, 0.50])
    lab = ["10 pedestrian\nblocks off", "10 car\nblocks off", "10 random\nblocks off"]
    ped, car = [4.9, 2.1, 0.0], [2.0, 14.5, 0.0]
    y = np.arange(3)[::-1]
    k = ease(t * 1.4 - 0.3)
    ax.barh(y + 0.18, np.array(ped) * k, 0.34, color=PED, edgecolor=BG, linewidth=2, label="pedestrian detection")
    ax.barh(y - 0.18, np.array(car) * k, 0.34, color=VEH, edgecolor=BG, linewidth=2, label="car detection")
    ax.set_yticks(y, lab, fontsize=12)
    ax.set_xlim(0, 17)
    ax.set_xlabel("quality lost (AP points)", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    ax.set_facecolor(BG)
    caption(fig, "Switching off pedestrian blocks hurts pedestrians most; car blocks hurt cars most.")


def s_understands(fig, t):
    header(fig, "Finding 2", "It understands details, but does not act on them")
    items = ["walking towards the car's path", "standing on a zebra crossing", "being on the road",
             "being in a group", "emerging from behind an obstacle"]
    for i, it in enumerate(items):
        a = ease(t * 3 - i * 0.3)
        fig.text(0.08, 0.66 - i * 0.075, "✓  " + it, fontsize=18, alpha=a, color=INK)
    fig.text(0.08, 0.72, "Encoded inside the model:", fontsize=15, color=INK2)
    if t > 0.55:
        fig.text(0.58, 0.60, "Extra planner attention\nfor these pedestrians:", fontsize=17, color=INK2)
        fig.text(0.58, 0.42, "≈ 0", fontsize=64, weight="bold", color=PED)
        fig.text(0.72, 0.45, "(-0.01, range -0.08 to +0.06,\nfor walking towards the path)", fontsize=13,
                 color=MUTED)
    caption(fig, "The knowledge is there; the planner does not single these pedestrians out.")


def s_memory(fig, t):
    header(fig, "Finding 3", "Hidden and new pedestrians are the weak point")
    image_ax(fig, os.path.join(FIG, "f5_memory.jpg"), [0.03, 0.17, 0.50, 0.58])
    ax = fig.add_axes([0.62, 0.22, 0.34, 0.48])
    k = ease(t * 1.3 - 0.2)
    x = np.arange(2)
    occ, vis = [0.11, 0.55], [0.22, 0.71]
    ax.bar(x - 0.18, np.array(occ) * k, 0.34, color=PED, edgecolor=BG, linewidth=2, label="partly hidden")
    ax.bar(x + 0.18, np.array(vis) * k, 0.34, color=VEH, edgecolor=BG, linewidth=2, label="visible")
    if t > 0.7:
        for xi, a_, b_ in zip(x, occ, vis):
            ax.text(xi - 0.18, a_ + 0.02, f"{a_:.0%}", ha="center", fontsize=13, weight="bold")
            ax.text(xi + 0.18, b_ + 0.02, f"{b_:.0%}", ha="center", fontsize=13, weight="bold")
    ax.set_xticks(x, ["first sighting", "after 4+ sightings"], fontsize=12)
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("share detected", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.set_facecolor(BG)
    caption(fig, "Hidden pedestrians are found through memory, so a pedestrian who just appeared is often missed.")


def s_experiment(fig, t):
    header(fig, "The key experiment", "Hide pedestrians, measure how far the path moves")
    ax = fig.add_axes([0.10, 0.16, 0.80, 0.58])
    ax.set_xlim(-6, 6)
    ax.set_ylim(-1, 32)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((-4.5, -1), 9, 33, color="#ecebe6", zorder=0))
    ax.plot([0, 0], [-1, 32], ls=(0, (6, 6)), color=BG, lw=3, zorder=1)
    ax.add_patch(FancyBboxPatch((-0.9, 0), 1.8, 3.2, boxstyle="round,pad=0.1", fc=INK, ec=INK, zorder=3))
    ys = np.linspace(3.5, 30, 50)
    ax.plot(np.zeros_like(ys) + 0.0 * ys, ys, color=INK2, lw=4, zorder=2, label="planned path")
    shift = 0.06 * ease(t * 2 - 0.6) * (ys - 3.5) / 26.5
    ax.plot(shift, ys, color=PED, lw=3, ls="--", zorder=4, label="path after hiding the pedestrian")
    pa = 1 - ease(t * 2 - 0.4)
    ax.scatter([2.6], [18], s=900, color=PED, alpha=max(pa, 0.15), zorder=5, edgecolor=BG, linewidth=2)
    ax.text(3.3, 18, "pedestrian" if pa > 0.5 else "hidden from\nthe planner", va="center", fontsize=15,
            color=INK2)
    ax.legend(frameon=False, fontsize=13, loc="upper left")
    if t > 0.8:
        ax.text(-5.8, 9, "The path moves by\nabout 5 cm, invisible\nat this scale\n(a lane is 350 cm wide).", fontsize=15,
                color=INK, va="center")
    caption(fig, "Hidden from attention, from the collision check and from memory, in every frame.")


def s_headline(fig, t):
    header(fig, "Finding 4: the headline", "The planned path barely depends on pedestrians")
    ax1 = fig.add_axes([0.20, 0.47, 0.30, 0.22])
    hbars(ax1, ["one pedestrian", "one matched car"], [1.51, 3.72], [PED, VEH], ease(t * 1.6), " cm", xmax=5)
    ax1.set_title("Remove ONE agent (matched on distance)", fontsize=14, loc="left", color=INK2)
    ax2 = fig.add_axes([0.20, 0.15, 0.30, 0.22])
    hbars(ax2, ["all pedestrians", "all vehicles"], [5.6, 35.5], [PED, VEH], ease(t * 1.6 - 0.5), " cm", xmax=45)
    ax2.set_title("Remove ALL of a group, every frame", fontsize=14, loc="left", color=INK2)
    if t > 0.7:
        fig.text(0.62, 0.58, "For scale:", fontsize=16, color=INK2)
        fig.text(0.62, 0.47, "61 cm", fontsize=46, weight="bold", color=INK)
        fig.text(0.62, 0.40, "is the planner's normal error\nagainst the human driver", fontsize=15, color=INK2)
    caption(fig, "Removing all pedestrians changes the plan about a sixth as much as removing all vehicles.")


def s_exception(fig, t):
    header(fig, "The rare exception", "Big effects only when the planner switches manoeuvre")
    image_ax(fig, os.path.join(FIG, "f6_planner.jpg"), [0.10, 0.15, 0.80, 0.60])
    caption(fig, "Here hiding the pedestrian moves the path 74.5 cm. Most pedestrians: under 1 cm.")


def s_attention(fig, t):
    header(fig, "Finding 5", "Attention ranks agents, but does not measure reliance")
    ax = fig.add_axes([0.30, 0.22, 0.60, 0.48])
    labs = ["planner attention", "random guess", "physics rule\n(time to collision)"]
    vals = [0.90, 0.50, 0.34]
    hbars(ax, labs, vals, [PED, GRAY, VEH], ease(t * 1.5), "", xmax=1.0, fmt="{:.2f}")
    ax.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("how well it finds the agents that matter most (1 = perfect)", fontsize=12)
    caption(fig, "A pedestrian can top the attention list and still barely move the path.")


def s_navsim(fig, t):
    header(fig, "Finding 6: a second dataset (NAVSIM)", "Removing pedestrians does not change the score")
    ax = fig.add_axes([0.30, 0.20, 0.62, 0.50])
    labs = ["all pedestrians removed", "all vehicles removed", "all agents removed"]
    vals = np.array([0.001, -0.023, -0.030])
    lo, hi = np.array([-0.006, -0.036, -0.040]), np.array([0.007, -0.009, -0.019])
    y = np.arange(3)[::-1]
    k = ease(t * 1.5)
    ax.barh(y, vals * k, 0.5, color=[PED, VEH, GRAY], edgecolor=BG, linewidth=2)
    if t > 0.6:
        ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="none", ecolor=INK2, capsize=5, lw=1.5)
        for yi, v in zip(y, vals):
            ax.text(0.003 if v >= 0 else v - 0.003, yi + 0.33, f"{v:+.3f}", ha="left" if v >= 0 else "right",
                    fontsize=14, weight="bold")
    ax.axvline(0, color=INK2, lw=1.2)
    ax.set_yticks(y, labs, fontsize=14)
    ax.set_xlim(-0.05, 0.02)
    ax.set_xlabel("change in driving score (EPDMS), 95 % range", fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor(BG)
    caption(fig, "1,712 scenes: vehicles matter to the score, pedestrians do not.")


def s_takeaway(fig, t):
    header(fig, "Takeaway", "Seen, but barely used")
    lines = ["The model knows the pedestrian is there.",
             "Its plan hardly depends on that knowledge.",
             "Driving scores and attention maps would not reveal this.",
             "Test reliance directly: hide the pedestrian and measure."]
    for i, ln in enumerate(lines):
        a = ease(t * 3 - i * 0.4)
        fig.text(0.08, 0.66 - i * 0.10, f"{i + 1}.  {ln}", fontsize=22, alpha=a, weight="bold" if i == 1 else None)
    fig.text(0.08, 0.17, "Paper, code and data pipeline:", fontsize=15, color=INK2)
    fig.text(0.08, 0.11, "github.com/pavanyadava007/seen-but-barely-used", fontsize=18, color=VEH, weight="bold")


SCENES = [(5, 0.3, s_title), (8, 0.6, s_question), (9, 0.6, s_pipeline), (7, 0.1, s_scene), (9, 0.6, s_sees),
          (9, 0.6, s_understands), (9, 0.6, s_memory), (9, 0.7, s_experiment), (10, 0.6, s_headline),
          (7, 0.1, s_exception), (8, 0.6, s_attention), (9, 0.6, s_navsim), (9, 0.5, s_takeaway)]


def render(draw, t):
    fig = new_fig()
    draw(fig, t)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def frames():
    bg = np.array([int(BG[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
    fade = int(0.4 * FPS)
    for secs, anim, draw in SCENES:
        n, na = int(secs * FPS), int(secs * FPS * anim)
        held = None
        for i in range(n):
            if i < na:
                img = render(draw, i / max(na - 1, 1))
            else:
                if held is None:
                    held = render(draw, 1.0)
                img = held
            a = min(1.0, (i + 1) / fade, (n - i) / fade)
            if a < 1.0:
                img = (img.astype(np.float32) * a + bg * (1 - a)).astype(np.uint8)
            yield img


def ffmpeg_bin():
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


def main():
    ff = ffmpeg_bin()
    mp4 = os.path.join(HERE, "explainer.mp4")
    gif = os.path.join(REPO, "docs", "img", "explainer.gif")
    p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                          "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                          "-movflags", "+faststart", mp4], stdin=subprocess.PIPE)
    for k, img in enumerate(frames()):
        p.stdin.write(img.tobytes())
        if k % 240 == 0:
            print(f"frame {k}", flush=True)
    p.stdin.close()
    p.wait()
    # lighter GIF for the README (plays inline on GitHub)
    pal = os.path.join(HERE, "_palette.png")
    vf = "fps=8,scale=800:-1:flags=lanczos"
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", mp4, "-vf", vf + ",palettegen=max_colors=96", pal],
                   check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", mp4, "-i", pal, "-lavfi",
                    vf + " [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4", gif], check=True)
    os.remove(pal)
    print("wrote", mp4, "and", gif)


if __name__ == "__main__":
    main()
