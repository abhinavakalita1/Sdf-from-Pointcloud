"""
3D version of rays.py
──────────────────────
• 3D workspace: unit cube with same box obstacles
• Distance transform on 3D voxel grid → max-inscribed sphere radius
• Sphere expansion algorithm:
    seed start, goal + 10 random free points
    each keypress: sample points on sphere surfaces →
                   grow max-inscribed spheres from uncovered points
    coalesce: if new sphere fully inside existing → discard
              if existing inside new → replace
    terminate: when no new sphere of radius >= MIN_R can be placed
• Interactive matplotlib figure with slider for cross-section view
  + auto-renders each expansion step to PNG frames
"""

import numpy as np
import math, random, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import distance_transform_edt

t0 = time.time()

# ════════════════════════════════════════════════════════════════════
#  WORLD
# ════════════════════════════════════════════════════════════════════
N      = 80
WALL_T = 0.05
obs    = np.zeros((N,N,N), dtype=np.uint8)

def vox(v):  return int(np.clip(v*N, 0, N-1))
def fill(x0,y0,z0,x1,y1,z1):
    obs[vox(x0):vox(x1), vox(y0):vox(y1), vox(z0):vox(z1)] = 1

fill(0,0,0, WALL_T,1,1);      fill(1-WALL_T,0,0, 1,1,1)
fill(0,0,0, 1,WALL_T,1);      fill(0,1-WALL_T,0, 1,1,1)
fill(0,0,0, 1,1,WALL_T);      fill(0,0,1-WALL_T, 1,1,1)
fill(0.30, 0.30, 0.10,  0.50, 0.55, 0.90)   # pillar
fill(0.55, 0.10, 0.10,  0.90, 0.45, 0.65)   # slab
fill(0.15, 0.60, 0.55,  0.28, 0.78, 0.75)   # small cube

# 3D distance transform: value at voxel = distance to nearest obstacle (voxels)
# Convert to world units (each voxel = 1/N world units)
print("Computing 3D distance transform...")
free_map = (obs == 0).astype(np.uint8)
dt_vox   = distance_transform_edt(free_map)   # in voxel units
dt_world = dt_vox / N                         # in world [0,1] units
print(f"  Max inscribable radius in world: {dt_world.max():.3f}")

def w2v(pt):  return tuple(int(np.clip(pt[i]*N,0,N-1)) for i in range(3))

def max_r(pt):
    """Max sphere radius at world point pt."""
    ix,iy,iz = w2v(pt)
    return float(dt_world[ix,iy,iz])

# ════════════════════════════════════════════════════════════════════
#  SPHERE STATE
# ════════════════════════════════════════════════════════════════════
MIN_R   = 2.5 / N    # 2.5 voxels in world units
spheres = []         # list of {'c': np.array, 'r': float}

# Covered voxel map — 1 if inside any sphere
covered = np.zeros((N,N,N), dtype=np.uint8)

def paint_covered(c, r):
    """Mark voxels inside sphere (c,r) as covered."""
    cx,cy,cz = w2v(c)
    ri = int(math.ceil(r*N)) + 1
    x0,x1 = max(0,cx-ri), min(N,cx+ri)
    y0,y1 = max(0,cy-ri), min(N,cy+ri)
    z0,z1 = max(0,cz-ri), min(N,cz+ri)
    xs = np.arange(x0,x1); ys = np.arange(y0,y1); zs = np.arange(z0,z1)
    XX,YY,ZZ = np.meshgrid(xs,ys,zs, indexing='ij')
    dist2 = ((XX-cx)**2+(YY-cy)**2+(ZZ-cz)**2)
    covered[x0:x1,y0:y1,z0:z1][dist2 <= (r*N)**2] = 1

def add_sphere(pt):
    pt = np.array(pt, dtype=float)
    r  = max_r(pt)
    if r < MIN_R:
        return False

    for s in spheres:
        d = np.linalg.norm(pt - s['c'])
        if d + r <= s['r']:          # new inside existing → absorb
            return False
        if d + s['r'] <= r:          # existing inside new → replace
            s['c'] = pt.copy()
            s['r'] = r
            paint_covered(pt, r)
            return False

    spheres.append({'c': pt.copy(), 'r': r})
    paint_covered(pt, r)
    return True

# ════════════════════════════════════════════════════════════════════
#  SURFACE SAMPLING  (3D analogue of circumference sampling)
#  Sample points on sphere surface using Fibonacci lattice
# ════════════════════════════════════════════════════════════════════
def fibonacci_sphere(n):
    """n points uniformly distributed on unit sphere."""
    pts = []
    phi = math.pi * (math.sqrt(5) - 1)
    for i in range(n):
        y   = 1 - (i / (n-1)) * 2
        r   = math.sqrt(max(0, 1-y*y))
        th  = phi * i
        pts.append(np.array([r*math.cos(th), y, r*math.sin(th)]))
    return pts

SPHERE_DIRS = fibonacci_sphere(48)   # fixed direction set

def surface_candidates(s):
    """Points on sphere surface that land in uncovered free space."""
    out = []
    for d in SPHERE_DIRS:
        pt = s['c'] + s['r'] * d
        if np.any(pt < 0) or np.any(pt > 1):
            continue
        if max_r(pt) < MIN_R:
            continue
        ix,iy,iz = w2v(pt)
        if covered[ix,iy,iz] == 1:
            continue
        out.append(pt)
    return out

# ════════════════════════════════════════════════════════════════════
#  SEED
# ════════════════════════════════════════════════════════════════════
start  = np.array([0.12, 0.12, 0.50])
target = np.array([0.75, 0.75, 0.80])

add_sphere(start)
add_sphere(target)

rng = random.Random(42)
attempts = 0
while len(spheres) < 12 and attempts < 5000:
    attempts += 1
    pt = np.array([rng.random(), rng.random(), rng.random()])
    if max_r(pt) >= MIN_R:
        add_sphere(pt)

print(f"Seed spheres: {len(spheres)}")

# ════════════════════════════════════════════════════════════════════
#  EXPANSION  — run automatically for several rounds, save each step
# ════════════════════════════════════════════════════════════════════
SPHERE_COLOR = np.array([0.65, 0.80, 1.0])  # uniform colour (like 2D)
N_ROUNDS     = 6

def draw_scene(ax, title_suffix=""):
    ax.cla()
    ax.set_facecolor('#f0f4ff')
    ax.set_title(f"3D Sphere Expansion  {title_suffix}", fontsize=10)

    # Draw obstacle boxes (wireframe outlines)
    obs_boxes = [
        [(0.30,0.30,0.10),(0.50,0.55,0.90)],
        [(0.55,0.10,0.10),(0.90,0.45,0.65)],
        [(0.15,0.60,0.55),(0.28,0.78,0.75)],
    ]
    for (lo,hi) in obs_boxes:
        # 12 edges of a box
        xs = [lo[0],hi[0]]
        ys = [lo[1],hi[1]]
        zs = [lo[2],hi[2]]
        for x in xs:
            for y in ys:
                ax.plot([x,x],[y,y],[lo[2],hi[2]], color='gray', lw=0.8, alpha=0.5)
        for x in xs:
            for z in zs:
                ax.plot([x,x],[lo[1],hi[1]],[z,z], color='gray', lw=0.8, alpha=0.5)
        for y in ys:
            for z in zs:
                ax.plot([lo[0],hi[0]],[y,y],[z,z], color='gray', lw=0.8, alpha=0.5)

    # Draw spheres as wireframe icospheres
    # Use parametric sphere at lower resolution for speed
    u = np.linspace(0, 2*np.pi, 16)
    v = np.linspace(0, np.pi, 10)
    xu = np.outer(np.cos(u), np.sin(v))
    yu = np.outer(np.sin(u), np.sin(v))
    zu = np.outer(np.ones_like(u), np.cos(v))

    for s in spheres:
        cx,cy,cz = s['c']; r = s['r']
        ax.plot_surface(cx+r*xu, cy+r*yu, cz+r*zu,
                        color=SPHERE_COLOR, alpha=0.18,
                        linewidth=0, antialiased=False)

    # Start / Goal
    ax.scatter(*start,  s=100, c='limegreen',  zorder=5, depthshade=False)
    ax.scatter(*target, s=100, c='dodgerblue', zorder=5, depthshade=False)
    ax.text(*(start  +0.03), 'S', color='darkgreen', fontsize=11, fontweight='bold')
    ax.text(*(target +0.03), 'G', color='navy',      fontsize=11, fontweight='bold')

    ax.set(xlim=(0,1),ylim=(0,1),zlim=(0,1),xlabel='X',ylabel='Y',zlabel='Z')
    ax.text2D(0.02, 0.02, f"Spheres: {len(spheres)}",
              transform=ax.transAxes, fontsize=9, color='#333333')

# ── Save initial state ────────────────────────────────────────────
fig = plt.figure(figsize=(10,9))
ax  = fig.add_subplot(111, projection='3d')
draw_scene(ax, "(seed)")
plt.tight_layout()
plt.savefig("rays_3d_step0.png", dpi=120)
plt.close()
print("Saved step 0")

# ── Expand and save each round ────────────────────────────────────
for rnd in range(1, N_ROUNDS+1):
    parents   = list(spheres)
    added_any = False

    for s in parents:
        for pt in surface_candidates(s):
            if add_sphere(pt):
                added_any = True

    fig = plt.figure(figsize=(10,9))
    ax  = fig.add_subplot(111, projection='3d')
    draw_scene(ax, f"(round {rnd})")
    plt.tight_layout()
    fname = f"rays_3d_step{rnd}.png"
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Round {rnd}: spheres={len(spheres)}  new={added_any}  → {fname}")

    if not added_any:
        print("  Terminated — no new sphere of radius >= MIN_R can be placed.")
        break

# ── Final composite showing all steps side by side ────────────────
n_steps = min(rnd+1, N_ROUNDS+1)
fig, axes = plt.subplots(2, 3, figsize=(18, 12),
                         subplot_kw={'projection': '3d'})
axes = axes.flatten()

for step_i in range(min(6, n_steps)):
    img = plt.imread(f"/mnt/user-data/outputs/rays_3d_step{step_i}.png")
    axes[step_i].remove()
    axes[step_i] = fig.add_subplot(2, 3, step_i+1)
    axes[step_i].imshow(img)
    axes[step_i].axis('off')
    axes[step_i].set_title(f"Step {step_i}", fontsize=10)

plt.suptitle("3D Sphere Expansion — Keypress Simulation", fontsize=13)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/rays_3d_all.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"\nAll done in {time.time()-t0:.1f}s")