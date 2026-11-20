"""
Finite elements from first principles with FEniCSx (DOLFINx)
=============================================================
  1. Convergence of the finite element method: Poisson equation with a
     manufactured solution, Lagrange P1 and P2 elements, L2 and H1 error rates
     compared with the theoretical h^(p+1) and h^p
  2. Stress concentration at a hole: plane-stress linear elasticity written in
     weak form (quarter model with symmetry), compared with
       - Kirsch's analytical solution for an infinite plate (Kt = 3)
       - Heywood's formula for plates of finite width
     plus a mesh convergence study of the peak stress

Run with the system Python that has DOLFINx (OpenFOAM/venv not needed):
    python3 fem_verification.py
Written for DOLFINx 0.9 to 0.11.
"""
import os
import math
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
import ufl
import gmsh
from dolfinx import fem, mesh as dmesh
import dolfinx.fem.petsc as fpetsc

try:                                    # DOLFINx >= 0.10
    from dolfinx.io import gmsh as gmsh_io
except ImportError:                     # DOLFINx 0.8 - 0.9
    from dolfinx.io import gmshio as gmsh_io

OUT = "results"
os.makedirs(OUT, exist_ok=True)
COMM = MPI.COMM_WORLD
_counter = itertools.count()


# ------------------------------------------------------------------
# small helpers that hide API differences between DOLFINx versions
# ------------------------------------------------------------------
def solve_linear(a, L, bcs):
    opts = {"ksp_type": "preonly", "pc_type": "lu"}
    try:
        problem = fpetsc.LinearProblem(a, L, bcs=bcs, petsc_options=opts,
                                       petsc_options_prefix=f"p{next(_counter)}_")
    except TypeError:
        problem = fpetsc.LinearProblem(a, L, bcs=bcs, petsc_options=opts)
    out = problem.solve()
    return out[0] if isinstance(out, tuple) else out


def interp_points(V):
    pts = V.element.interpolation_points
    return pts() if callable(pts) else pts


def integrate(expr):
    return COMM.allreduce(fem.assemble_scalar(fem.form(expr)), op=MPI.SUM)


# ------------------------------------------------------------------
# 1. CONVERGENCE STUDY (Poisson, manufactured solution)
# ------------------------------------------------------------------
def poisson(n, degree):
    msh = dmesh.create_unit_square(COMM, n, n)
    V = fem.functionspace(msh, ("Lagrange", degree))
    x = ufl.SpatialCoordinate(msh)
    u_exact = ufl.sin(ufl.pi * x[0]) * ufl.sin(ufl.pi * x[1])
    f = 2 * ufl.pi ** 2 * u_exact                         # -laplace(u_exact) = f
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx       # weak form
    L = f * v * ufl.dx
    msh.topology.create_connectivity(1, 2)
    facets = dmesh.exterior_facet_indices(msh.topology)
    zero = fem.Function(V)
    bc = fem.dirichletbc(zero, fem.locate_dofs_topological(V, 1, facets))
    uh = solve_linear(a, L, [bc])
    e_l2 = math.sqrt(integrate((uh - u_exact) ** 2 * ufl.dx))
    e_h1 = math.sqrt(integrate(ufl.inner(ufl.grad(uh - u_exact), ufl.grad(uh - u_exact)) * ufl.dx))
    return 1.0 / n, e_l2, e_h1, V.dofmap.index_map.size_global


# ------------------------------------------------------------------
# 2. PLATE WITH A HOLE (plane stress, quarter model)
# ------------------------------------------------------------------
E, NU, SIGMA0 = 210e3, 0.3, 100.0           # MPa, MPa
R = 1.0                                      # hole radius [mm]


def plate_mesh(W, h_hole, h_far):
    """Quarter model of a plate of width W (along y) and length 2W (along x), hole of radius R at the origin."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("plate")
    occ = gmsh.model.occ
    rect = occ.addRectangle(0, 0, 0, W, W / 2)
    disk = occ.addDisk(0, 0, 0, R, R)
    surf, _ = occ.cut([(2, rect)], [(2, disk)])
    occ.synchronize()
    tags = {"left": [], "bottom": [], "right": [], "top": [], "hole": []}
    for _, c in gmsh.model.getBoundary(surf, oriented=False):
        x0, y0, _, x1, y1, _ = gmsh.model.getBoundingBox(1, c)
        if abs(x0) < 1e-8 and abs(x1) < 1e-8:
            tags["left"].append(c)
        elif abs(y0) < 1e-8 and abs(y1) < 1e-8:
            tags["bottom"].append(c)
        elif abs(x0 - W) < 1e-8 and abs(x1 - W) < 1e-8:
            tags["right"].append(c)
        elif abs(y0 - W / 2) < 1e-8 and abs(y1 - W / 2) < 1e-8:
            tags["top"].append(c)
        else:
            tags["hole"].append(c)
    for i, name in enumerate(["left", "bottom", "right", "top", "hole"], 1):
        gmsh.model.addPhysicalGroup(1, tags[name], i, name=name)
    gmsh.model.addPhysicalGroup(2, [surf[0][1]], 1, name="plate")
    f = gmsh.model.mesh.field
    d = f.add("Distance")
    f.setNumbers(d, "CurvesList", tags["hole"])
    f.setNumber(d, "Sampling", 200)
    t = f.add("Threshold")
    f.setNumber(t, "InField", d)
    f.setNumber(t, "SizeMin", h_hole)
    f.setNumber(t, "SizeMax", h_far)
    f.setNumber(t, "DistMin", 0.2 * R)
    f.setNumber(t, "DistMax", 4 * R)
    f.setAsBackgroundMesh(t)
    for o in ("MeshSizeExtendFromBoundary", "MeshSizeFromPoints", "MeshSizeFromCurvature"):
        gmsh.option.setNumber("Mesh." + o, 0)
    gmsh.model.mesh.generate(2)
    out = gmsh_io.model_to_mesh(gmsh.model, COMM, 0, gdim=2)
    gmsh.finalize()
    if hasattr(out, "mesh"):
        return out.mesh, out.facet_tags
    return out[0], out[2]


def plate_with_hole(W, h_hole, h_far=None, degree=2):
    h_far = h_far or W / 8
    msh, _ = plate_mesh(W, h_hole, h_far)
    # boundary markers built directly from the coordinates (independent of the Gmsh tags)
    fdim = 1
    marks = []
    for tag, test in ((1, lambda x: np.isclose(x[0], 0.0)), (2, lambda x: np.isclose(x[1], 0.0)),
                      (3, lambda x: np.isclose(x[0], W))):
        f_ids = dmesh.locate_entities_boundary(msh, fdim, test)
        marks.append((f_ids, np.full(len(f_ids), tag, dtype=np.int32)))
    ids = np.concatenate([m[0] for m in marks])
    vals = np.concatenate([m[1] for m in marks])
    order = np.argsort(ids)
    ft = dmesh.meshtags(msh, fdim, ids[order].astype(np.int32), vals[order])
    V = fem.functionspace(msh, ("Lagrange", degree, (2,)))
    mu = E / (2 * (1 + NU))
    lam = E * NU / ((1 + NU) * (1 - 2 * NU))
    lam_ps = 2 * lam * mu / (lam + 2 * mu)               # plane stress

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sigma(w):
        return 2 * mu * eps(w) + lam_ps * ufl.tr(eps(w)) * ufl.Identity(2)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    traction = fem.Constant(msh, np.array([SIGMA0, 0.0]))
    a = ufl.inner(sigma(u), eps(v)) * ufl.dx
    L = ufl.dot(traction, v) * ds(3)                      # tension on the right edge
    # symmetry: ux = 0 on the left edge (x = 0), uy = 0 on the bottom edge (y = 0)
    Vx, _ = V.sub(0).collapse()
    Vy, _ = V.sub(1).collapse()
    left = ft.find(1)
    bottom = ft.find(2)
    bcs = [fem.dirichletbc(fem.Function(Vx), fem.locate_dofs_topological((V.sub(0), Vx), 1, left), V.sub(0)),
           fem.dirichletbc(fem.Function(Vy), fem.locate_dofs_topological((V.sub(1), Vy), 1, bottom), V.sub(1))]
    uh = solve_linear(a, L, bcs)

    S = fem.functionspace(msh, ("Discontinuous Lagrange", degree - 1))
    sxx = fem.Function(S)
    sxx.interpolate(fem.Expression(sigma(uh)[0, 0], interp_points(S)))
    xy = S.tabulate_dof_coordinates()[:, :2]
    vals = sxx.x.array
    near = np.hypot(xy[:, 0], xy[:, 1] - R) < 1e-6        # the point (0, R): top of the hole
    s_max = vals[near].max() if near.any() else vals.max()
    return dict(W=W, h=h_hole, dofs=V.dofmap.index_map.size_global * 2, s_max=s_max, xy=xy, sxx=vals)


def heywood_kt(d_over_w):
    """Kt referred to the net-section stress, finite-width plate in tension (Heywood)."""
    return 2 + (1 - d_over_w) ** 3


# ==================================================================
def main():
    report = []

    def log(text=""):
        print(text)
        report.append(text)

    import dolfinx
    log(f"FEniCSx / DOLFINx {dolfinx.__version__}\n")

    # ---------------- 1. Convergence ----------------
    log("1. CONVERGENCE: -laplace(u) = f on the unit square, u_exact = sin(pi x) sin(pi y)")
    conv = {}
    for p in (1, 2):
        conv[p] = [poisson(n, p) for n in (4, 8, 16, 32, 64)]
        log(f"  P{p}: {'h':>8} {'dofs':>7} {'L2 error':>11} {'rate':>5} {'H1 error':>11} {'rate':>5}")
        for i, (h, el2, eh1, nd) in enumerate(conv[p]):
            if i:
                h0, el20, eh10, _ = conv[p][i - 1]
                r2, r1 = math.log(el20 / el2) / math.log(h0 / h), math.log(eh10 / eh1) / math.log(h0 / h)
                log(f"      {h:>8.4f} {nd:>7} {el2:>11.3e} {r2:>5.2f} {eh1:>11.3e} {r1:>5.2f}")
            else:
                log(f"      {h:>8.4f} {nd:>7} {el2:>11.3e} {'':>5} {eh1:>11.3e}")
        log(f"     theory: L2 rate {p + 1}, H1 rate {p}")

    # ---------------- 2. Plate with a hole ----------------
    log("\n2. PLATE WITH A HOLE IN TENSION (plane stress, P2 elements, quarter model)")
    log("  Mesh convergence, W = 20 d (almost infinite plate):")
    W_inf = 40.0 * R
    mesh_runs = [plate_with_hole(W_inf, h) for h in (0.4, 0.2, 0.1, 0.05)]
    for m in mesh_runs:
        net = SIGMA0 * W_inf / (W_inf - 2 * R)
        log(f"    hole element size {m['h']:5.3f} mm, {m['dofs']:>7} dofs: peak {m['s_max']:7.2f} MPa, "
            f"Kt(gross) = {m['s_max'] / SIGMA0:.4f}, Kt(net) = {m['s_max'] / net:.4f}")
    kt_inf = mesh_runs[-1]["s_max"] / (SIGMA0 * W_inf / (W_inf - 2 * R))
    log(f"  Kirsch (infinite plate): Kt = 3.000; Heywood at d/W = {2 * R / W_inf:.3f}: "
        f"{heywood_kt(2 * R / W_inf):.4f}; FEA: {kt_inf:.4f}")

    log("\n  Finite-width plates (Kt referred to the net section):")
    ratios = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    width_runs = []
    for dw in ratios:
        W = 2 * R / dw
        m = plate_with_hole(W, 0.05)
        net = SIGMA0 * W / (W - 2 * R)
        m["kt"] = m["s_max"] / net
        width_runs.append(m)
        log(f"    d/W = {dw:.2f}: FEA Kt = {m['kt']:.4f}, Heywood {heywood_kt(dw):.4f} "
            f"({(m['kt'] / heywood_kt(dw) - 1) * 100:+.1f}%)")

    # ================= FIGURES =================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, k, name in ((axes[0], 1, "L2"), (axes[1], 2, "H1 seminorm")):
        for p, mk in ((1, "o"), (2, "s")):
            h = np.array([c[0] for c in conv[p]])
            e = np.array([c[k] for c in conv[p]])
            ax.loglog(h, e, mk + "-", label=f"P{p} elements")
            rate = p + 1 if k == 1 else p
            ax.loglog(h, e[-1] * (h / h[-1]) ** rate, "k:", lw=1)
            ax.text(h[1], e[-1] * (h[1] / h[-1]) ** rate * 1.6, f"slope {rate}", fontsize=8)
        ax.set_xlabel("Element size h")
        ax.set_ylabel(f"{name} error")
        ax.set_title(f"{name} error: dotted lines are the theoretical rates")
        ax.grid(alpha=0.3, which="both")
        ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/convergence_rates.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    dd = np.linspace(0, 0.65, 100)
    ax.plot(dd, heywood_kt(dd), "k-", label="Heywood formula")
    ax.plot(ratios, [m["kt"] for m in width_runs], "o", color="tab:red", ms=7, label="FEniCSx")
    ax.plot(0, 3, "s", color="tab:blue", ms=8, label="Kirsch, infinite plate")
    ax.set_xlabel("Hole diameter / plate width, d/W")
    ax.set_ylabel("Kt (net section)")
    ax.set_title("Stress concentration factor vs plate width")
    ax.grid(alpha=0.3)
    ax.legend()
    ax = axes[1]
    m = mesh_runs[-1]
    xy, s = m["xy"], m["sxx"]
    on_axis = (np.abs(xy[:, 0]) < 1e-6) & (xy[:, 1] >= R) & (xy[:, 1] < 5 * R)
    o = np.argsort(xy[on_axis, 1])
    ax.plot(xy[on_axis, 1][o] / R, s[on_axis][o] / SIGMA0, ".", ms=4, color="tab:red", label="FEniCSx")
    yy = np.linspace(R, 5 * R, 200)
    ax.plot(yy / R, 0.5 * (2 + (R / yy) ** 2 + 3 * (R / yy) ** 4), "k-", label="Kirsch")
    ax.set_xlabel("Distance from the hole centre y / r (along x = 0)")
    ax.set_ylabel("σxx / σ0")
    ax.set_title("Stress decay away from the hole (W = 20 d)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/stress_concentration.png", dpi=150)
    plt.close(fig)

    m = width_runs[2]
    xy, s = m["xy"], m["sxx"]
    sel = (xy[:, 0] < 4 * R) & (xy[:, 1] < 4 * R)
    tri = mtri.Triangulation(xy[sel, 0], xy[sel, 1])
    cen = np.c_[xy[sel, 0][tri.triangles].mean(1), xy[sel, 1][tri.triangles].mean(1)]
    tri.set_mask(np.hypot(cen[:, 0], cen[:, 1]) < R)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cf = ax.tricontourf(tri, s[sel] / SIGMA0, levels=30, cmap="turbo")
    ax.add_patch(plt.Circle((0, 0), R, color="w"))
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"σxx / σ0 around the hole (d/W = {ratios[2]}), quarter model")
    fig.colorbar(cf, ax=ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/stress_field.png", dpi=150)
    plt.close(fig)

    with open(f"{OUT}/report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\nFigures and report saved in '{OUT}'.")


if __name__ == "__main__":
    main()
