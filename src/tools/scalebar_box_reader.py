"""Scale-bar reader anchored on the box corner, bounded by flood-filling the white interior.

Verified on 200 ROIs: the annotation box's 1-px dark border has its bottom-right corner at
exactly (W-10, H-10). The interior is uniformly near-white and fully enclosed by that border,
so flood-filling from just inside the corner recovers the exact interior -- unlike walking along
the border, which runs off into dark tissue whenever tissue touches the box (that produced
oversized boxes and bar lengths measured on the box edge instead of the bar).

Inside the interior: the bar is the widest dark horizontal span in the upper part (tick-to-tick);
the label text sits below it.
"""
import numpy as np
from PIL import Image
from scipy import ndimage

OFF = 9
DARK = 150
WHITE = 200

def read(path):
    im = Image.open(path).convert("L")
    W, H = im.size
    a = np.asarray(im)
    bx, by = W - 1 - OFF, H - 1 - OFF          # dark border corner
    if bx < 60 or by < 40: return None
    if a[by, bx] >= DARK: return None          # no border pixel at the anchor -> no box
    sx, sy = bx - 2, by - 2                    # a pixel just inside the interior
    if sx < 1 or sy < 1 or a[sy, sx] < WHITE: return None
    win = a[max(0, by - 60):by + 1, max(0, bx - 300):bx + 1]
    oy0, ox0 = max(0, by - 60), max(0, bx - 300)
    lab, n = ndimage.label(win >= WHITE)
    tag = lab[sy - oy0, sx - ox0]
    if tag == 0: return None
    ys, xs = np.where(lab == tag)
    t, b, l, r = ys.min() + oy0, ys.max() + oy0, xs.min() + ox0, xs.max() + ox0
    h, w = b - t + 1, r - l + 1
    if not (18 <= h <= 34) or not (35 <= w <= 260): return None
    if r < bx - 3 or b < by - 3: return None   # interior must reach the anchor corner
    inner = a[t:b + 1, l:r + 1]
    upper = inner[: max(3, int(h * 0.55))] < DARK
    span = 0
    for row in upper:
        if row.any():
            k = np.flatnonzero(row)
            span = max(span, int(k.max() - k.min() + 1))
    if span < 25 or span > w: return None
    return dict(W=W, H=H, box_w=float(w), box_h=float(h), bar_px=float(span), left=int(l), top=int(t))
