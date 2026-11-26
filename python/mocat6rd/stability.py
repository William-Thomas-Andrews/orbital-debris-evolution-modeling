"""
Jacobians, equilibria and stability with JAX for MOCAT-6RD.

Model: Gusmini et al., "Effects of Orbit Raising and Deorbiting in Source-Sink
Evolutionary Models", JSR 61(3), 2024. Variable conventions follow
mocat6rd_variable_reference.pdf (equation numbers refer to the paper).

    dSh/dt = s lam_Sh - vartheta_h_minus Sh + Theta_h_plus Tinv Sr             + Cdot_Sh
    dSr/dt = s lam_Sr + (-Theta_h_plus + Theta_r_plus - Theta_r_minus
                         - (1-P_r) I) Tinv Sr                                  + Cdot_Sr
    dSd/dt = P_m vartheta_h_minus Sh + (-Theta_d_minus + Theta_d_plus
                         - (1-P_d) I) Tinv Sd                                  + Cdot_Sd
    dD/dt  = (1-P_m) vartheta_h_minus Sh + (1-P_r) Tinv Sr + (1-P_d) Tinv Sd
             + Fdot_d_D + Cdot_D
    dN/dt  = Fdot_d_N + Cdot_N
    dB/dt  = Fdot_d_B + Cdot_B

's' scales the post-deployment launch profile (s = 1 is the paper's profile). Launches enter
additively, so dF/ds = Lambda_hat and the Jacobian dF/dP does not depend on s.

Units: km, years, kg throughout (anything computed in SI is converted).
Everything that does not depend on the state is a module-level constant; everything that
does is a pure jnp function, so it can go straight into jax.jacfwd / jax.jacrev.

Main entry points
    system(P, s)                    dP/dt                         (also fits optimistix: fn(y, args))
    jacobian(P, s)                  dF/dP, jitted, 6n x 6n
    initial_population()            Sept 2022 population (Fig. 4)
    find_equilibrium(s, P0)         Newton root of system(., s) = 0
    stability(P, s)                 eigenvalues of the Jacobian, sorted by real part
    trace_equilibria(s_max)         follow the stable, non-negative equilibrium branch in s
    simulate(P0, t_end, s)          stiff time integration (BDF + exact Jacobian)
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx
from scipy.integrate import solve_ivp
from scipy.special import ellipe, ellipk

jax.config.update("jax_enable_x64", True)  # float64: important for stiff systems / eigenvalues


################################################################################################
# Initial setup
################################################################################################
n = 36            # number of shells
h_min = 200.0     # km
h_max = 2000.0    # km
d = (h_max - h_min) / n  # shell thickness = 50 km

R_E = 6378.137                   # Earth radius, km
mu_SI = 3.986004418e14           # m^3/s^2
mu_km = mu_SI * 1e-9             # km^3/s^2
YEAR_S = 365.25 * 86400.0        # seconds per year

h_bot = h_min + d * jnp.arange(n)  # nx1; lower shell edges, km
h_top = h_bot + d                  # nx1; upper shell edges, km
h = h_bot + d / 2                  # nx1; shell midpoints, km  [choice]
R = R_E + h                        # nx1; radius at midpoint, km

I = jnp.identity(n)  # nxn

# species order for collision tables: S, D, N, B
iS, iD, iN, iB = 0, 1, 2, 3


def _data_file(name):
    """Find data/<name> next to this package or above the working directory (works in notebooks too)."""
    starts = [Path.cwd()]
    if "__file__" in globals():
        starts.insert(0, Path(__file__).resolve().parent)
    for start in starts:
        for base in (start, *start.parents):
            f = base / "data" / name
            if f.is_file():
                return f
    raise FileNotFoundError(f"data/{name} not found above {', '.join(map(str, starts))}")


def _load_csv(name):
    table = np.genfromtxt(_data_file(name), delimiter=",", names=True)
    if table.shape != (n,) or not np.allclose(table["h_bot_km"], np.asarray(h_bot)):
        raise ValueError(f"{name}: expected {n} rows matching the 50 km shells {h_min:.0f}-{h_max:.0f} km")
    return table
################################################################################################
################################################################################################


################################################################################################
# State (the unknowns)
################################################################################################
# P = [Sh, Sr, Sd, D, N, B] stacked -> 6n x 1 (216 unknowns). Use unpack(P) inside the system.
n_species = 6
n_state = n_species * n  # 216

idx_Sh = slice(0 * n, 1 * n)
idx_Sr = slice(1 * n, 2 * n)
idx_Sd = slice(2 * n, 3 * n)
idx_D = slice(3 * n, 4 * n)
idx_N = slice(4 * n, 5 * n)
idx_B = slice(5 * n, 6 * n)
species_names = ("S_h", "S_r", "S_d", "D", "N", "B")


def unpack(P):
    """P (6n,) -> S_h, S_r, S_d, D, N, B (each n,)."""
    return P[idx_Sh], P[idx_Sr], P[idx_Sd], P[idx_D], P[idx_N], P[idx_B]


def pack(S_h, S_r, S_d, D, N, B):
    """Inverse of unpack."""
    return jnp.concatenate([S_h, S_r, S_d, D, N, B])


def total_S(S_h, S_r, S_d):
    """S = Sh + Sr + Sd (nx1). Computed from the state, not stored."""
    return S_h + S_r + S_d
################################################################################################
################################################################################################


################################################################################################
# Launches
################################################################################################
# Constant (post-deployment) profile for s = 1; the system scales it by s. The paper divides
# the Fig. 3 values by 10. Fig. 3 is in data/gusmini2024_fig3_launch_profile.csv: columns
# raising_YYYY (Fig. 3a, by TARGET shell) and direct_YYYY (Fig. 3b), YYYY = 2023..2033, #/yr.
# 2033 is the held-constant profile; earlier years are the deployment ramp.
launch_reduction = 10.0
launch_year = 2033

_fig3 = _load_csv("gusmini2024_fig3_launch_profile.csv")
lambda_S_h = jnp.asarray(_fig3[f"direct_{launch_year}"]) / launch_reduction   # nx1; #/yr, direct injection
lambda_tgt = jnp.asarray(_fig3[f"raising_{launch_year}"]) / launch_reduction  # nx1; #/yr, by target shell

# [inferred] injection shells: targets 200-400 km -> shell 1 (idx 0), higher -> shell 5 (idx 4)
inj_shells = (0, 4)
batch_id = jnp.where(h_bot < 400.0, 0, 1)  # nx1; which batch each target shell belongs to

lambda_S_r = (jnp.zeros(n)                                    # nx1; nonzero only at the injection shells
              .at[inj_shells[0]].set(jnp.sum(jnp.where(batch_id == 0, lambda_tgt, 0.0)))
              .at[inj_shells[1]].set(jnp.sum(jnp.where(batch_id == 1, lambda_tgt, 0.0))))

P_fb = 0.3 # a moderate estimate of 30% probability that rocket body parts are lost in space
S_per_B = 20 # a moderate estimate of how many satellites are carried/launched per rocket body part
lambda_B = P_fb * lambda_S_r / S_per_B

P_fn = 0.15 # an aggressive estimage of 15% probability that satellite launch debris on launch
P_fnb = 0.5 # a moderate estimate on the probability a rocket ejects debris on launch
lambda_N = P_fn * lambda_S_r + P_fnb * lambda_B

P_fd = 0.05 # an agressive estamate of 5% percent probability that satellite becomes derelict on launch
lambda_D = P_fd * lambda_S_r
lambda_S_r = (1 - P_fd) * lambda_S_r # update lambda_S_r by multiplying

# TODO: perhaps instead of doing this and dropping everything off at starlink height, make a percentage 
#       go to tgt heights by using lambda_tgt

_zeros = jnp.zeros(n)  # no launches of S_d
Lambda_hat = pack(lambda_S_h, lambda_S_r, _zeros, lambda_D, lambda_N, lambda_B)  # 6n x 1; launches at s = 1
################################################################################################
################################################################################################


################################################################################################
# Post-mission disposal
################################################################################################
change_t = 5.0                     # Delta t: mean operational lifetime; yr
vartheta_h_minus = 1 / change_t    # 1/yr (scalar, same every shell, Eq. 6)
P_m = 0.96                         # PMD success fraction -> Sd; failures -> D (Eq. 5)
################################################################################################
################################################################################################


################################################################################################
# Orbit transfer
################################################################################################
epsilon = 5e-5               # m/s^2 (Table 4; = 5e-8 km/s^2)
epsilon_km = epsilon * 1e-3  # km/s^2
ebar = 0.01

# Crossing time per shell, Eqs. 20-23. Elliptic argument 4e/(1+e)^2 is the PARAMETER m
# (SciPy convention). For e = 0.01 the modulus/parameter choice changes tau by < 0.1%.
_m_ell = 4 * ebar / (1 + ebar) ** 2
f_e = 2 / (1 - ebar) * ellipe(_m_ell) + 2 / (1 + ebar) * ellipk(_m_ell)  # = 2*pi at e = 0
_a0 = R_E + h_bot  # km
_af = R_E + h_top  # km
C1 = (epsilon_km * (1 - ebar**2) * f_e / (2 * jnp.pi * jnp.sqrt(mu_km))) ** 2  # 1/(km s^2)
C2 = epsilon_km * (1 - ebar**2) * f_e / (jnp.pi * jnp.sqrt(_a0 * mu_km))      # nx1; 1/(km s), sgn(cos a)=+1
_c = 1 / _a0 - 1 / _af
# smallest positive root of C1 t^2 - C2 t + c = 0, written in the cancellation-free form
tau_ot = 2 * _c / (C2 + jnp.sqrt(C2**2 - 4 * C1 * _c)) / YEAR_S  # nx1; yr (~0.013-0.019)

T = jnp.diag(tau_ot)         # nxn; yr
Tinv = jnp.diag(1 / tau_ot)  # nxn; 1/yr (never invert T numerically)

# [inferred] fraction of raising sats that stop at each shell:
#   vartheta_h_plus_i = lam_tgt_i / sum_{k >= i, same batch} lam_tgt_k
# Independent of s. Where nothing is left to deposit (0/0), set 1 (everyone stops).
_same_batch_above = (batch_id[None, :] == batch_id[:, None]) & (jnp.arange(n)[None, :] >= jnp.arange(n)[:, None])
_remaining = _same_batch_above.astype(float) @ lambda_tgt  # nx1
vartheta_h_plus = jnp.where(_remaining > 0, lambda_tgt / jnp.where(_remaining > 0, _remaining, 1.0), 1.0)
# the top shell of each batch deposits everything that is left
vartheta_h_plus = vartheta_h_plus.at[inj_shells[1] - 1].set(1.0).at[n - 1].set(1.0)

Theta_h_plus = jnp.diag(vartheta_h_plus)  # nxn (Eq. 27)

Theta_r_minus = jnp.diag(1 - vartheta_h_plus)                            # nxn (Eq. 28)
Theta_r_plus = jnp.diag(1 - vartheta_h_plus[:-1], k=-1)                   # nxn; entry (i+1, i) = 1 - vartheta_i
Theta_r_plus = Theta_r_plus.at[jnp.array(inj_shells), :].set(0.0)        # zero rows at injection shells

Theta_d_minus = jnp.identity(n)               # nxn (Eq. 29)
Theta_d_plus = jnp.diag(jnp.ones(n - 1), k=1)  # nxn; superdiagonal ones, last row zero

P_r = 0.9995  # transit success while raising; failures -> D (Eq. 33)
P_d = 0.999   # transit success while deorbiting; failures -> D

# Linear transfer operators as they appear in the equations (all 1/yr, nxn):
K_h_r = Theta_h_plus @ Tinv                                                   # Sr -> Sh deposit
K_r_r = (-Theta_h_plus + Theta_r_plus - Theta_r_minus - (1 - P_r) * I) @ Tinv  # Sr self
K_d_d = (-Theta_d_minus + Theta_d_plus - (1 - P_d) * I) @ Tinv                 # Sd self
K_D_r = (1 - P_r) * Tinv                                                      # Sr failures -> D
K_D_d = (1 - P_d) * Tinv                                                      # Sd failures -> D
################################################################################################
################################################################################################


################################################################################################
# Drag flux
################################################################################################
# CIRA-72 static exponential atmosphere (Vallado Table 8-4), rows with h0 >= 200 km.
# columns: h0 [km], rho0 [kg/m^3], H [km]
_cira72 = np.array([
    [200, 2.789e-10, 37.105],
    [250, 7.248e-11, 45.546],
    [300, 2.418e-11, 53.628],
    [350, 9.518e-12, 53.298],
    [400, 3.725e-12, 58.515],
    [450, 1.585e-12, 60.828],
    [500, 6.967e-13, 63.822],
    [600, 1.454e-13, 71.835],
    [700, 3.614e-14, 88.667],
    [800, 1.170e-14, 124.64],
    [900, 5.245e-15, 181.05],
    [1000, 3.019e-15, 268.00],
])
_band = np.searchsorted(_cira72[:, 0], np.asarray(h), side="right") - 1  # band containing each h
_h0, _rho0, _H = _cira72[_band].T
rho = jnp.asarray(_rho0 * np.exp(-(np.asarray(h) - _h0) / _H))  # nx1; kg/m^3 (Eq. 18)

c_D = 2.2
# Table 5, species order S, D, N, B
A = jnp.array([1.741, 1.741, 0.020, 7.419])     # m^2
m = jnp.array([223.0, 223.0, 0.640, 1421.0])    # kg
diam = jnp.array([1.490, 1.490, 0.180, 3.070])  # m
B_c = c_D * A / m  # m^2/kg (Eq. 17): [0.0172, 0.0172, 0.0688, 0.0115]

R_m = R * 1e3                    # nx1; m
_ms_to_kmyr = 1e-3 * YEAR_S      # m/s -> km/yr
v_D = -rho * B_c[iD] * jnp.sqrt(mu_SI * R_m) * _ms_to_kmyr  # nx1; km/yr (Eq. 16), negative
v_N = -rho * B_c[iN] * jnp.sqrt(mu_SI * R_m) * _ms_to_kmyr  # nx1; km/yr
v_B = -rho * B_c[iB] * jnp.sqrt(mu_SI * R_m) * _ms_to_kmyr  # nx1; km/yr

# Eq. 15 as constant operators: Fdot_d_Q = (1/d) (I - Theta_d_plus) diag(v_Q) Q   (1/yr, nxn)
L_drag_D = (I - Theta_d_plus) @ jnp.diag(v_D) / d
L_drag_N = (I - Theta_d_plus) @ jnp.diag(v_N) / d
L_drag_B = (I - Theta_d_plus) @ jnp.diag(v_B) / d


def Fdot_d_D(D):
    return L_drag_D @ D  # nx1; #/yr


def Fdot_d_N(N):
    return L_drag_N @ N  # nx1; #/yr


def Fdot_d_B(B):
    return L_drag_B @ B  # nx1; #/yr
################################################################################################
################################################################################################


################################################################################################
# Collisions
################################################################################################
vimp = 10.0 * YEAR_S  # km/yr (10 km/s = 3.156e8 km/yr)
vimp_kms = 10.0       # km/s (for the breakup model)
r = diam / 2 * 1e-3   # km; species order S, D, N, B
M = m                 # kg

sigma = (r[:, None] + r[None, :]) ** 2  # 4x4; km^2 (Eq. 13)
V = 4 / 3 * jnp.pi * ((R_E + h_top) ** 3 - (R_E + h_bot) ** 3)  # nx1; km^3
phi = jnp.pi * vimp * sigma[:, :, None] / V[None, None, :]       # 4x4xn; 1/yr (Eq. 12)

alpha = 0.05     # fraction of collisions an active sat fails to avoid (with D, N, B)
alpha_a = 0.005  # fraction of collisions an active sat fails to avoid (with S)
delta = 10.0     # ratio disabling : lethal trackable debris
delta_B = 10.0

L_c = 0.1  # m; smallest fragment counted in N
M_p = jnp.minimum(M[:, None], M[None, :])  # 4x4; kg
n_f_c = 0.1 * L_c ** -1.71 * (M[:, None] + M[None, :]) ** 0.75       # 4x4; catastrophic
n_f_nc = 0.1 * L_c ** -1.71 * (M_p * vimp_kms ** 2) ** 0.75          # 4x4; non-catastrophic
_involves_N = (jnp.arange(4)[:, None] == iN) | (jnp.arange(4)[None, :] == iN)
n_f = jnp.where(_involves_N, n_f_nc, n_f_c)  # 4x4 (Table 1); SS=498, SB=1324, BB=1996, *N=116


def _safe_frac(num, den):
    """num/den, with 0 value and 0 gradient where den <= 0 (B = S = 0 in a shell)."""
    ok = den > 0
    return jnp.where(ok, num / jnp.where(ok, den, 1.0), 0.0)


def Gamma(S, B):
    """4x4xn (Eq. 10). Constant except the two state-dependent entries S/(B+S), B/(B+S)."""
    one = jnp.ones_like(S)
    fS = _safe_frac(S, B + S)
    fB = _safe_frac(B, B + S)
    return jnp.stack([
        jnp.stack([-alpha_a * one, -(delta + alpha) * one, -(delta + alpha) * one, -(delta_B + alpha) * fS]),
        jnp.stack([delta * one, -1.0 * one, -1.0 * one, delta_B * one]),
        jnp.stack([alpha * one, 1.0 * one, 1.0 * one, 1.0 * one]),
        jnp.stack([-(delta_B + alpha) * fB, -(1 + delta_B) * one, -(1 + delta_B) * one, -(1 + delta_B) * one]),
    ])


def Cdot_add(S, D, N, B):
    """Eq. 11 -> (Cdot_S_add, Cdot_D_add, Cdot_N_add, Cdot_B_add), each nx1. N already carries n_f."""
    p, nf = phi, n_f
    C_S_add = jnp.zeros_like(S)
    C_B_add = jnp.zeros_like(S)
    C_D_add = (p[iS, iN] * delta * N * S + p[iS, iB] * delta_B * B * S
               + p[iN, iB] * delta_B * N * B + p[iB, iB] * delta_B * B**2)
    C_N_add = (nf[iS, iS] * alpha_a * p[iS, iS] * S**2 + nf[iS, iD] * alpha * p[iS, iD] * S * D
               + nf[iS, iB] * alpha * p[iS, iB] * S * B + nf[iD, iD] * p[iD, iD] * D**2
               + nf[iD, iB] * p[iD, iB] * D * B + nf[iB, iB] * p[iB, iB] * B**2)
    return C_S_add, C_D_add, C_N_add, C_B_add


_nf_weight = jnp.ones((4, 4)).at[iN, :].set(n_f[iN, :])[:, :, None]  # n_f weight on the N row


def Cdot(P):
    """Collision terms for all six species (each nx1, #/yr), phase split included.

    Cdot_i = sum_j Gamma_ij phi_ij Q_i Q_j + Cdot_i_add   (Q = [S, D, N, B])
    The N row is weighted by n_f per pair. Active sub-population x in {h, r, d}
    loses its own satellites with the total S as partner:
        Cdot_Sx = S_x * sum_j Gamma_Sj phi_Sj Q_j   (sums to Cdot_S).
    """
    S_h, S_r, S_d, D, N, B = unpack(P)
    S = total_S(S_h, S_r, S_d)
    Q = jnp.stack([S, D, N, B])  # 4xn
    G = Gamma(S, B)              # 4x4xn
    rate = jnp.sum(_nf_weight * G * phi * Q[None, :, :], axis=1)  # 4xn: sum_j (.)_ij Q_j
    _, C_D_add, C_N_add, _ = Cdot_add(S, D, N, B)                 # Cdot_S_add = Cdot_B_add = 0

    Cdot_Sh = S_h * rate[iS]
    Cdot_Sr = S_r * rate[iS]
    Cdot_Sd = S_d * rate[iS]
    Cdot_D = D * rate[iD] + C_D_add
    Cdot_N = N * rate[iN] + C_N_add
    Cdot_B = B * rate[iB]
    return Cdot_Sh, Cdot_Sr, Cdot_Sd, Cdot_D, Cdot_N, Cdot_B
################################################################################################
################################################################################################


################################################################################################
# System
################################################################################################
def system(P, s=1.0):
    """dP/dt for launch scale s. Signature fn(y, args) so it plugs straight into optimistix."""
    S_h, S_r, S_d, D, N, B = unpack(P)
    C_h, C_r, C_d, C_D, C_N, C_B = Cdot(P)

    dS_h = s * lambda_S_h - vartheta_h_minus * S_h + K_h_r @ S_r + C_h
    dS_r = s * lambda_S_r + K_r_r @ S_r + C_r
    dS_d = P_m * vartheta_h_minus * S_h + K_d_d @ S_d + C_d
    dD = s * lambda_D + (1 - P_m) * vartheta_h_minus * S_h + K_D_r @ S_r + K_D_d @ S_d + Fdot_d_D(D) + C_D
    dN = s * lambda_N + Fdot_d_N(N) + C_N
    dB = s * lambda_B + Fdot_d_B(B) + C_B
    return pack(dS_h, dS_r, dS_d, dD, dN, dB)


system_jit = jax.jit(system)
jacobian = jax.jit(jax.jacfwd(system))  # dF/dP, 6n x 6n; independent of s (launches are additive)
################################################################################################
################################################################################################


################################################################################################
# Initial population
################################################################################################
def initial_population():
    """Sept 2022 population from Fig. 4 (data/gusmini2024_fig4_initial_population.csv).

    The paper gives only S_h, D, N, B; raising and deorbiting satellites start at 0.
    """
    pop = _load_csv("gusmini2024_fig4_initial_population.csv")
    z = np.zeros(n)
    return pack(*(jnp.asarray(x) for x in (pop["S_h"], z, z, pop["D"], pop["N"], pop["B"])))
################################################################################################
################################################################################################


################################################################################################
# Equilibria and stability
################################################################################################
def find_equilibrium(s, P0=None, rtol=1e-10, atol=1e-8, max_steps=100):
    """Newton root of system(P, s) = 0 starting from P0 (default: zeros).

    From zeros, the first Newton step lands on the collision-free equilibrium, which is a good
    start for small s. For larger s, warm-start from a nearby solution (see trace_equilibria).
    Returns (P, converged). Converged does NOT mean physical: check P >= 0 and stability(P, s).
    """
    P0 = jnp.zeros(n_state) if P0 is None else jnp.asarray(P0, dtype=float)
    sol = optx.root_find(system, optx.Newton(rtol=rtol, atol=atol), P0, args=s,
                         max_steps=max_steps, throw=False)
    P = sol.value
    ok = bool(sol.result == optx.RESULTS.successful) and bool(jnp.all(jnp.isfinite(P)))
    return P, ok


def stability(P, s=1.0):
    """Eigenvalues of the Jacobian at P, sorted by decreasing real part.

    eig[0].real < 0 -> linearly stable. Note the slowest drag mode (~ -3.3e-5 /yr, rocket bodies
    in the top shell) usually sets eig[0], so it does not warn of an approaching instability.
    """
    ev = np.linalg.eigvals(np.asarray(jacobian(P, s)))
    return ev[np.argsort(-ev.real)]


def is_physical(P, tol=1e-9):
    """True if every population is non-negative (up to round-off relative to the largest one)."""
    P = np.asarray(P)
    return bool(np.all(np.isfinite(P)) and P.min() >= -tol * max(np.abs(P).max(), 1.0))


def trace_equilibria(s_max=1.0, s_start=1e-8, growth=1.5, min_growth=1 + 1e-6, verbose=False):
    """Follow the stable, non-negative equilibrium branch from s_start up to s_max.

    Natural-parameter continuation: predict with dP/ds = -J^{-1} Lambda_hat, correct with Newton,
    accept only converged, non-negative, stable solutions, and shrink the step when a step fails.
    Returns a dict with
        s, P, max_re   arrays along the accepted branch
        s_crit         last accepted s (equal to s_max if the branch reaches it)
        reason         'reached s_max', or why the branch was lost just above s_crit
                       ('no convergence', 'negative', 'unstable')
    """
    P, ok = find_equilibrium(s_start)
    if not (ok and is_physical(P)):
        raise RuntimeError(f"no physical equilibrium at s_start = {s_start}")
    s, fac, reason = s_start, growth, "reached s_max"
    out_s, out_P, out_re = [s], [P], [stability(P, s)[0].real]
    while s < s_max:
        if fac < min_growth:
            break
        s_try = min(s * fac, s_max)
        dP_ds = jnp.linalg.solve(jacobian(P, s), -Lambda_hat)
        P_try, ok = find_equilibrium(s_try, P + (s_try - s) * dP_ds)
        if not ok:
            reason = "no convergence"
        elif not is_physical(P_try):
            reason = "negative"
        else:
            max_re = stability(P_try, s_try)[0].real
            if max_re >= 0:
                reason = "unstable"
            else:
                s, P = s_try, P_try
                out_s.append(s); out_P.append(P); out_re.append(max_re)
                reason = "reached s_max"
                if verbose:
                    print(f"  s = {s:.6e}  max Re(eig) = {max_re:+.3e}  total N = {float(P[idx_N].sum()):.4g}")
                fac = min(fac ** 2, growth)  # grow the step back after a success
                continue
        fac = fac ** 0.5  # failed step: halve it (in log s)
    return dict(s=np.array(out_s), P=np.array(out_P), max_re=np.array(out_re), s_crit=s, reason=reason)
################################################################################################
################################################################################################


################################################################################################
# Time integration
################################################################################################
def simulate(P0, t_end, s=1.0, t_eval=None, rtol=1e-8, atol=1e-8, N_blowup=1e9):
    """Integrate dP/dt = system(P, s) from P0 over [0, t_end] years.

    Uses BDF with the exact JAX Jacobian: the system is stiff (rates span ~1e2 to ~1e-5 /yr),
    so explicit solvers like RK45 would crawl. Stops early (sol.status = 1) once total N
    exceeds N_blowup, which is how a runaway shows up; sol.t_events[0] gives the time.
    """
    def blowup(_t, y):
        return y[idx_N].sum() - N_blowup
    blowup.terminal = True

    return solve_ivp(lambda _t, y: np.asarray(system_jit(y, s)), (0.0, t_end), np.asarray(P0, dtype=float),
                     method="BDF", jac=lambda _t, y: np.asarray(jacobian(y, s)),
                     t_eval=t_eval, rtol=rtol, atol=atol, events=blowup)
################################################################################################
################################################################################################


if __name__ == "__main__":
    print(f"State size: {n_state} ({n} shells x {n_species} species); "
          f"launches at s = 1: {float(Lambda_hat.sum()):.0f} /yr")

    # Stable equilibrium branch as the launch scale grows
    branch = trace_equilibria(s_max=1.0)
    s_crit = branch["s_crit"]
    print(f"\nStable equilibria exist up to s = {s_crit:.4e} "
          f"({s_crit * float(Lambda_hat.sum()):.1f} launches/yr); above it: {branch['reason']}")

    # Equilibrium and stability at half the critical launch rate
    s_half = 0.5 * s_crit
    P_eq, ok = find_equilibrium(s_half, branch["P"][np.searchsorted(branch["s"], s_half) - 1])
    eig = stability(P_eq, s_half)
    print(f"\nAt s = {s_half:.4e}: converged = {ok}, physical = {is_physical(P_eq)}, "
          f"max Re(eig) = {eig[0].real:+.3e}")
    for name, x in zip(species_names, unpack(P_eq)):
        print(f"  {name:>3}: total {float(x.sum()):10.4g}   peak at {float(h[jnp.argmax(x)]):.0f} km")

    # No launches from the Sept 2022 population
    years = [0, 10, 50, 100, 200, 500, 1000, 2000]
    sol = simulate(initial_population(), 1e4, s=0.0, t_eval=years)
    print("\nNo launches from the Sept 2022 population (total N):")
    for t, y in zip(sol.t, sol.y.T):
        print(f"  t = {t:6.0f} yr   N = {y[idx_N].sum():10.4g}")
    if sol.t_events[0].size:
        print(f"  runaway: N passes 1e9 at t = {sol.t_events[0][0]:.0f} yr")
