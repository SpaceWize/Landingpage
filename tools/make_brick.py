"""Procedural, seamlessly tiling red brick wall texture.

Builds a height map (bricks raised, mortar recessed, rough surfaces, ragged edges),
gives every brick its own colour, then lights the height map from the upper left so
the result has real shading rather than flat fills.

Usage: python make_brick.py OUT_PATH [seed]
"""
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

out = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
rng = np.random.default_rng(seed)

W, H = 1440, 960
ROWS = 20
RH = H // ROWS            # course height
MORTAR = 6.5              # joint width in px


def noise(beta, lo=0.0, hi=1e9):
    """Periodic fractal noise in [-1, 1]: white noise shaped in the frequency domain."""
    f = np.fft.fft2(rng.standard_normal((H, W)))
    fy = np.fft.fftfreq(H)[:, None] * H
    fx = np.fft.fftfreq(W)[None, :] * W
    r = np.sqrt(fx * fx + fy * fy)
    r[0, 0] = 1.0
    amp = r ** (-beta)
    amp[(r < lo) | (r > hi)] = 0.0
    amp[0, 0] = 0.0
    n = np.real(np.fft.ifft2(f * amp))
    n -= n.mean()
    # scale by spread rather than by the single largest value, so thresholds bite evenly
    return np.clip(n / (n.std() * 2.6 + 1e-9), -1, 1)


def blur(a, s):
    return gaussian_filter(a, s, mode="wrap")


yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

# warp the layout a little so courses and joints are not ruler straight
wx = xx + noise(2.2, 1, 40) * 6.0
wy = yy + noise(2.2, 1, 60) * 5.0 + noise(2.6, 1, 6) * 5.0

row = np.floor(wy / RH).astype(int) % ROWS
vy = wy % RH                                   # position inside the course

brick_id = np.zeros((H, W), dtype=np.int64)
edge = np.zeros((H, W))                        # distance to the nearest brick edge
bricks = []                                    # per brick: (length,)
next_id = 0
for r in range(ROWS):
    # split the course into bricks: mostly stretchers, some headers and odd lengths
    lens = []
    total = 0
    while total < W:
        kind = rng.random()
        if kind < 0.16:
            l = rng.normal(72, 6)              # header
        elif kind < 0.30:
            l = rng.normal(106, 9)            # three-quarter
        else:
            l = rng.normal(146, 10)            # stretcher
        l = float(np.clip(l, 54, 176))
        lens.append(l)
        total += l
    lens = np.array(lens) * (W / total)        # make the course close up exactly
    bounds = np.concatenate([[0.0], np.cumsum(lens)])
    off = rng.uniform(0, W)
    m = row == r
    ux = (wx[m] + off) % W
    idx = np.clip(np.searchsorted(bounds, ux, side="right") - 1, 0, len(lens) - 1)
    dx = np.minimum(ux - bounds[idx], bounds[idx + 1] - ux)
    dy = np.minimum(vy[m], RH - vy[m])
    brick_id[m] = next_id + idx
    edge[m] = np.minimum(dx, dy)
    next_id += len(lens)

N = next_id

# ragged edges: joints wander, and some bricks are chipped
rag = noise(1.7, 5, 160) * 4.2 + noise(1.0, 60, 500) * 1.6
chip_amt = rng.random(N) ** 2 * 12.0            # most bricks barely chipped, a few a lot
chips = np.clip(noise(1.9, 4, 90) - 0.1, 0, 1) * chip_amt[brick_id]
d = edge - MORTAR / 2 + rag - chips
mask = np.clip(d / 2.6, 0, 1)
mask = mask * mask * (3 - 2 * mask)            # smooth brick/mortar transition

# ---- height map -------------------------------------------------------------------------------
tilt = rng.normal(0, 0.09, N)[brick_id]        # bricks sit slightly proud or shy of the wall
rough_b = noise(1.4, 4, 600) * 0.20 + noise(0.5, 80, 700) * 0.09
pits = np.clip(noise(1.0, 25, 300) - 0.55, 0, 1) * 1.3          # small holes in the faces
bevel = np.clip(d / 7.0, 0, 1) ** 0.6                           # rounded arrises
h_brick = 0.70 + 0.28 * bevel + tilt + rough_b - pits * 0.35
rough_m = noise(0.8, 20, 700) * 0.07 + noise(1.8, 2, 40) * 0.06
h_mortar = 0.30 + rough_m
height = h_mortar * (1 - mask) + h_brick * mask
height = blur(height, 0.7)

# ---- colour -----------------------------------------------------------------------------------
palette = np.array([
    [150, 64, 42],    # brick red
    [132, 54, 38],    # deeper red
    [164, 78, 48],    # orange red
    [112, 46, 34],    # dark red brown
    [176, 96, 60],    # light, sandy orange
    [88, 42, 34],     # burnt
    [146, 80, 58],    # dusty
], dtype=np.float64)
weights = np.array([.26, .22, .17, .13, .08, .07, .07])
base = palette[rng.choice(len(palette), N, p=weights)]
base *= rng.normal(1.0, 0.10, (N, 1))                            # per brick brightness
warm = rng.normal(0, 1, (N, 1))                                  # per brick shift along red-orange only
base += warm * np.array([6.0, 5.0, 1.0])
col_b = base[brick_id]

mottle = noise(1.9, 2, 90)                                       # broad blotches
fine = noise(0.7, 60, 700)                                       # grit
col_b = col_b * (1 + mottle[..., None] * 0.30 + fine[..., None] * 0.18)
# arrises are darker and more burnt than the face
col_b = col_b * (0.74 + 0.26 * bevel[..., None])
# iron spots
spots = np.clip(noise(0.3, 120, 700) - 0.72, 0, 1)[..., None] * 3.0
col_b = col_b * (1 - np.clip(spots, 0, 0.75))
# kiln scorch: darker patches on some faces
scorch = np.clip(noise(2.0, 2, 40) - 0.30, 0, 1)[..., None]
col_b = col_b * (1 - scorch * 0.6) + np.array([52, 32, 30]) * scorch * 0.6
# pale mineral bloom and old mortar smears on the faces
bloom = np.clip(noise(1.6, 3, 110) - 0.38, 0, 1)[..., None]
col_b = col_b * (1 - bloom * 0.55) + np.array([186, 150, 128]) * bloom * 0.55
# mortar smeared a little way onto the brick edges
smear = np.clip(1 - d / 10.0, 0, 1)[..., None] * np.clip(noise(1.5, 4, 120) + 0.1, 0, 1)[..., None]
col_b = col_b * (1 - smear * 0.5) + np.array([170, 150, 132]) * smear * 0.5
# pits read dark
col_b *= (1 - pits[..., None] * 0.7)

col_m = np.array([158, 146, 130], dtype=np.float64) * (1 + noise(1.6, 2, 120)[..., None] * 0.16
                                                        + noise(0.6, 80, 700)[..., None] * 0.14)
dirt = np.clip(noise(2.0, 2, 24), 0, 1)[..., None]
col_m = col_m * (1 - dirt * 0.35) + np.array([92, 78, 66]) * dirt * 0.35
# brick dust staining the mortar next to each brick
near = blur(mask, 3.0)[..., None]
col_m = col_m * (1 - near * 0.30) + np.array([150, 92, 70]) * near * 0.30

col = col_m * (1 - mask[..., None]) + col_b * mask[..., None]

# ---- lighting ---------------------------------------------------------------------------------
gx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * 0.5
gy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * 0.5
STRENGTH = 22.0
nx, ny, nz = -gx * STRENGTH, -gy * STRENGTH, np.ones_like(height)
ln = np.sqrt(nx * nx + ny * ny + nz * nz)
nx, ny, nz = nx / ln, ny / ln, nz / ln
L = np.array([-0.52, -0.62, 0.59])
L /= np.linalg.norm(L)
diff = np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)
flat = L[2]                                                       # response of a flat surface
light = 0.42 + 0.58 * (diff / flat)

# contact shadow in the joints and under each course
ao = blur(1 - mask, 2.2) * (mask * 0.55 + 0.45)
shadow = np.clip(np.roll(np.roll(mask, 5, 0), 4, 1) - mask, 0, 1)  # bricks shade the joint below/right
shadow = blur(shadow, 1.6)
light = light * (1 - ao * 0.42) * (1 - shadow * 0.45)

col = col * light[..., None]

# slow tonal drift across the wall, then a touch of contrast and grain
col *= (1 + noise(2.6, 1, 5)[..., None] * 0.10)
# soot and rain streaks spanning several bricks
grime = np.clip(noise(2.3, 1, 14) - 0.1, 0, 1)[..., None]
col = col * (1 - grime * 0.28)
# pull the saturation back toward a sun-faded wall
gray = col.mean(axis=2, keepdims=True)
col = gray + (col - gray) * 0.86
col += rng.normal(0, 3.2, (H, W, 1))
img = np.clip(col, 0, 255).astype(np.uint8)

im = Image.fromarray(img, "RGB")
if out.lower().endswith(".webp"):
    im.save(out, "WEBP", quality=80, method=6)
elif out.lower().endswith((".jpg", ".jpeg")):
    im.save(out, "JPEG", quality=84, optimize=True, progressive=True)
else:
    im.save(out)
print("saved", out, im.size, "bricks:", N)
