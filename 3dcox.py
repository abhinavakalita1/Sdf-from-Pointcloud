"""
3D Coxeter + Boundary + Ray Casting
Clean version — outer boundary hidden from viz, inner obstacles prominent
"""
import numpy as np
import math, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict

t0 = time.time()

# ════════════════════════════════════════════════════════════════════
#  WORLD
# ════════════════════════════════════════════════════════════════════
N      = 64
WALL_T = 0.05
obs    = np.zeros((N,N,N), dtype=np.uint8)

def vox(v):   return int(np.clip(v*N, 0, N-1))
def fill(x0,y0,z0,x1,y1,z1):
    obs[vox(x0):vox(x1), vox(y0):vox(y1), vox(z0):vox(z1)] = 1

# Outer shell
fill(0,0,0, WALL_T,1,1);      fill(1-WALL_T,0,0, 1,1,1)
fill(0,0,0, 1,WALL_T,1);      fill(0,1-WALL_T,0, 1,1,1)
fill(0,0,0, 1,1,WALL_T);      fill(0,0,1-WALL_T, 1,1,1)

# Obstacle 1 — tall pillar
fill(0.30, 0.30, 0.10,  0.50, 0.55, 0.90)
# Obstacle 2 — wide slab
fill(0.55, 0.10, 0.10,  0.90, 0.45, 0.65)
# Obstacle 3 — small cube
fill(0.15, 0.60, 0.55,  0.28, 0.78, 0.75)

start  = np.array([0.12, 0.12, 0.50])
target = np.array([0.75, 0.75, 0.80])

def w2v(pt):  return tuple(int(np.clip(pt[i]*N,0,N-1)) for i in range(3))
def free(pt): ix,iy,iz=w2v(pt); return obs[ix,iy,iz]==0

assert free(start),  "Start in obstacle"
assert free(target), "Target in obstacle"

# ════════════════════════════════════════════════════════════════════
#  BCC TETRAHEDRAL TILING
# ════════════════════════════════════════════════════════════════════
TSTEP = 0.11
PERMS = [
    [(0,0,0),(1,0,0),(1,1,0),(1,1,1)],
    [(0,0,0),(1,0,0),(1,0,1),(1,1,1)],
    [(0,0,0),(0,1,0),(1,1,0),(1,1,1)],
    [(0,0,0),(0,1,0),(0,1,1),(1,1,1)],
    [(0,0,0),(0,0,1),(1,0,1),(1,1,1)],
    [(0,0,0),(0,0,1),(0,1,1),(1,1,1)],
]

def gen_tets(step):
    ns = int(1.0/step)+1
    for ix in range(ns):
        for iy in range(ns):
            for iz in range(ns):
                ox,oy,oz = ix*step, iy*step, iz*step
                for p in PERMS:
                    yield [(ox+p[k][0]*step, oy+p[k][1]*step, oz+p[k][2]*step)
                           for k in range(4)]

def tet_cen(t): return np.mean(t, axis=0)

def tet_free(t):
    c = tet_cen(t)
    if np.any(c<0) or np.any(c>1): return False
    return obs[w2v(c)]==0

print("Tiling...")
free_tets, obs_tets = [], []
for t in gen_tets(TSTEP):
    arr = np.array(t)
    if np.any(arr<-0.01) or np.any(arr>1.01): continue
    (free_tets if tet_free(t) else obs_tets).append(t)
print(f"  free={len(free_tets)}  obs={len(obs_tets)}")

# ════════════════════════════════════════════════════════════════════
#  BOUNDARY EXTRACTION
# ════════════════════════════════════════════════════════════════════
def faces(tet):
    idx=[0,1,2,3]
    return [tuple(sorted([tet[i] for i in idx if i!=o],
                         key=lambda p:(round(p[0],5),round(p[1],5),round(p[2],5))))
            for o in idx]

print("Extracting boundaries...")
ff = defaultdict(int)   # face free-count
fo = defaultdict(int)   # face obs-count
for t in free_tets:
    for f in faces(t): ff[f]+=1
for t in obs_tets:
    for f in faces(t): fo[f]+=1

bfaces = [f for f in ff if fo.get(f,0)>0]
print(f"  Boundary faces: {len(bfaces)}")

# Connected components via shared-edge BFS
e2f = defaultdict(list)
for fi,face in enumerate(bfaces):
    pts=list(face)
    for i in range(3):
        e=tuple(sorted([pts[i],pts[(i+1)%3]],
                       key=lambda p:(round(p[0],5),round(p[1],5),round(p[2],5))))
        e2f[e].append(fi)

vis=[False]*len(bfaces)
comps=[]
def bfs(s):
    q,comp=[s],[]
    vis[s]=True
    while q:
        fi=q.pop(); comp.append(fi)
        pts=list(bfaces[fi])
        for i in range(3):
            e=tuple(sorted([pts[i],pts[(i+1)%3]],
                           key=lambda p:(round(p[0],5),round(p[1],5),round(p[2],5))))
            for nfi in e2f[e]:
                if not vis[nfi]: vis[nfi]=True; q.append(nfi)
    return comp

for fi in range(len(bfaces)):
    if not vis[fi]:
        c=bfs(fi)
        if len(c)>5: comps.append(c)

comps.sort(key=len,reverse=True)
Blbls=[f"B{i+1}" for i in range(len(comps))]
print(f"  Components: {[f'{Blbls[i]}={len(c)}tri' for i,c in enumerate(comps)]}")

# ════════════════════════════════════════════════════════════════════
#  RAY CASTING  (Möller–Trumbore, +X ray)
# ════════════════════════════════════════════════════════════════════
def mt(orig, tri):
    v0,v1,v2 = (np.array(tri[k],float) for k in range(3))
    d=np.array([1.,0.,0.])
    e1,e2=v1-v0,v2-v0
    h=np.cross(d,e2); a=np.dot(e1,h)
    if abs(a)<1e-9: return None
    f=1/a; s=orig-v0; u=f*np.dot(s,h)
    if u<0 or u>1: return None
    q=np.cross(s,e1); v=f*np.dot(d,q)
    if v<0 or u+v>1: return None
    t=f*np.dot(e2,q)
    return t if t>1e-6 else None

def n_cross(pt, ci):
    return sum(1 for fi in comps[ci] if mt(pt, bfaces[fi]) is not None)

s_cr = [(Blbls[i], n_cross(start,  i)) for i in range(len(comps))]
g_cr = [(Blbls[i], n_cross(target, i)) for i in range(len(comps))]
s_sig = tuple(n%2 for _,n in s_cr)
g_sig = tuple(n%2 for _,n in g_cr)
same  = s_sig==g_sig

print("\n====== START ======")
for l,n in s_cr: print(f"  {l}: {n} → {'IN' if n%2 else 'OUT'}")
print("====== GOAL =======")
for l,n in g_cr: print(f"  {l}: {n} → {'IN' if n%2 else 'OUT'}")
verdict = "SAME REGION — PATH MAY EXIST" if same else "DIFFERENT REGIONS — PATH CANNOT EXIST"
print(f"  {verdict}")

# ════════════════════════════════════════════════════════════════════
#  DRAW
#  Left:  Coxeter tiling (free tets) + inner boundary surfaces
#  Right: boundary surfaces only + ray intersection dots
# ════════════════════════════════════════════════════════════════════
# Color map — skip B1 (outer shell) in visualization, show B2+ prominently
BCOLS = [
    (0.9,0.3,0.3),   # B1
    (0.2,0.7,0.2),   # B2
    (0.6,0.2,0.9),   # B3
    (0.1,0.7,0.8),   # B4
    (0.9,0.6,0.1),   # B5
]
# Alpha for filled surfaces
BALPHAS = [0.08, 0.35, 0.40, 0.40, 0.40]  # B1 nearly invisible

fig = plt.figure(figsize=(17,9), facecolor='#f8f8f8')

# ── LEFT: tiling + surfaces ──────────────────────────────────────
ax1 = fig.add_subplot(121, projection='3d')
ax1.set_facecolor('#f0f4ff')
ax1.set_title("BCC Tetrahedral Tiling  +  Boundary Surfaces", fontsize=10, pad=8)

# Tet wireframes (sparse sample)
step = max(1, len(free_tets)//250)
for tet in free_tets[::step]:
    pts = np.array(tet)
    fi  = [[0,1,2],[0,1,3],[0,2,3],[1,2,3]]
    pc  = Poly3DCollection([[pts[k] for k in f] for f in fi],
                           alpha=0.03, edgecolor=(0.6,0.6,0.8),
                           linewidth=0.25, facecolor=(0.85,0.90,1.0))
    ax1.add_collection3d(pc)

# Boundary surfaces (B1 very faint, others solid)
for ci,(comp,lbl) in enumerate(zip(comps,Blbls)):
    col   = BCOLS[ci%len(BCOLS)]
    alpha = BALPHAS[ci] if ci<len(BALPHAS) else 0.35
    tris  = [list(bfaces[fi]) for fi in comp]
    step2 = max(1, len(tris)//600)
    pc    = Poly3DCollection(tris[::step2], alpha=alpha,
                             facecolor=col, edgecolor='none')
    ax1.add_collection3d(pc)
    if ci>0:   # only label inner obstacles
        pts2  = np.array([p for fi in comp for p in bfaces[fi]])
        cx,cy,cz = pts2.mean(axis=0)
        ax1.text(cx,cy,cz+0.04, lbl, fontsize=12, fontweight='bold',
                 color=col, ha='center')

ax1.quiver(*start,  0.85-start[0],  0,0, color='blue',  lw=1.5, arrow_length_ratio=0.04)
ax1.quiver(*target, 0.85-target[0], 0,0, color='darkorange', lw=1.5, arrow_length_ratio=0.04)
ax1.scatter(*start,  s=100, c='lime',       zorder=5, depthshade=False)
ax1.scatter(*target, s=100, c='dodgerblue', zorder=5, depthshade=False)
ax1.text(*(start  + [-0.02,0.02,0.03]), 'S', color='darkgreen', fontsize=11, fontweight='bold')
ax1.text(*(target + [-0.02,0.02,0.03]), 'G', color='navy',      fontsize=11, fontweight='bold')
ax1.set(xlim=(0,1),ylim=(0,1),zlim=(0,1),xlabel='X',ylabel='Y',zlabel='Z')

# ── RIGHT: surfaces + ray hits ────────────────────────────────────
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_facecolor('#f8f0f8')
ax2.set_title("Boundary Surfaces  +  Ray Intersection Hits", fontsize=10, pad=8)

for ci,(comp,lbl) in enumerate(zip(comps,Blbls)):
    col   = BCOLS[ci%len(BCOLS)]
    alpha = 0.10 if ci==0 else 0.50
    tris  = [list(bfaces[fi]) for fi in comp]
    step2 = max(1, len(tris)//600)
    pc    = Poly3DCollection(tris[::step2], alpha=alpha,
                             facecolor=col, edgecolor=(0,0,0) if ci>0 else 'none',
                             linewidth=0.15)
    ax2.add_collection3d(pc)
    if ci>0:
        pts2 = np.array([p for fi in comp for p in bfaces[fi]])
        cx,cy,cz = pts2.mean(axis=0)
        ax2.text(cx,cy,cz+0.04, lbl, fontsize=12, fontweight='bold',
                 color=col, ha='center')

# Ray hit dots
for ci,comp in enumerate(comps):
    col = BCOLS[ci%len(BCOLS)]
    for fi in comp:
        t=mt(start, bfaces[fi])
        if t: ax2.scatter(*(start+t*np.array([1,0,0])), s=55, c=[col], marker='o', zorder=8)
    for fi in comp:
        t=mt(target, bfaces[fi])
        if t: ax2.scatter(*(target+t*np.array([1,0,0])), s=55, c=[col], marker='^', zorder=8)

ax2.quiver(*start,  0.85-start[0],  0,0, color='blue',       lw=1.5, arrow_length_ratio=0.04)
ax2.quiver(*target, 0.85-target[0], 0,0, color='darkorange', lw=1.5, arrow_length_ratio=0.04)
ax2.scatter(*start,  s=100, c='lime',       zorder=5, depthshade=False)
ax2.scatter(*target, s=100, c='dodgerblue', zorder=5, depthshade=False)
ax2.text(*(start  + [-0.02,0.02,0.03]), 'S', color='darkgreen', fontsize=11, fontweight='bold')
ax2.text(*(target + [-0.02,0.02,0.03]), 'G', color='navy',      fontsize=11, fontweight='bold')
ax2.set(xlim=(0,1),ylim=(0,1),zlim=(0,1),xlabel='X',ylabel='Y',zlabel='Z')

# ── Summary text ─────────────────────────────────────────────────
lines  = ["── START ──────────────────────"]
lines += [f"  {l}: {n} crossings  →  {'INSIDE' if n%2 else 'outside'}" for l,n in s_cr]
lines += ["── GOAL ───────────────────────"]
lines += [f"  {l}: {n} crossings  →  {'INSIDE' if n%2 else 'outside'}" for l,n in g_cr]
lines += ["───────────────────────────────", verdict]
vcol = '#006600' if same else '#cc0000'
fig.text(0.5, 0.01, "\n".join(lines), ha='center', va='bottom',
         fontsize=8.5, fontfamily='monospace',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                   edgecolor='#888888', alpha=0.95))

plt.tight_layout(rect=[0, 0.22, 1, 1])
out = "main_3d.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nSaved → {out}  ({time.time()-t0:.1f}s)")