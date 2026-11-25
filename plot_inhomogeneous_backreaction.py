"""Single-band production check and first-band energy-transfer estimate.

The upper panel checks the single-band result

    ln(1 + n_k) = pi R / 2,   R = kappa^2 A^2 / (m_chi^3 H),

against direct mode-function evolution.  The lower panel integrates the
phase-space flux of modes through k/a = m_chi/2 for the delayed-release
benchmark of Sec. V E:

    d rho_fluc / dN + 4 rho_fluc = m_chi^4 n_k / (32 pi^2).

The plotted backreaction fraction has the unknown hierarchy m_chi/f scaled
out, so the physical result is obtained by multiplying by (m_chi/f)^2.

The lower panel integrates the local narrow-band production formula; it
does not directly evolve the full two-field fluctuation spectrum. It assumes
initially unoccupied crossing modes, slow background evolution, relativistic
products, and negligible rescattering and stabilizer depletion. A small
energy fraction does not by itself bound the light-field variance, primordial
long-wavelength fluctuations, or nonlinear fragmentation after release.
"""
import sys

import numpy as np
from scipy.integrate import solve_ivp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from resonance import X


def exact_band_crossing():
    """Return direct mode-function results in the narrow-resonance regime."""
    # A small Mathieu parameter, q=2b/omega=0.02, cleanly isolates the
    # narrow first band while keeping the integrations inexpensive.
    b, omega = 2.0, 200.0
    R = np.array([0.05, 0.10, 0.20, 0.40, 0.65, 1.00])
    occupation = []
    for value in R:
        hubble = b**2 / (omega * value)
        _, number = X(b, omega, hubble, 0.6, 0.4)
        occupation.append(number)
    return R, np.asarray(occupation)


def delayed_release_backreaction():
    """Integrate rho_fluc/rho_chi with (m_chi/f)^2 divided out."""
    r = 0.5
    m_tilde = 1.0
    m_phi = m_tilde / np.sqrt(1.0 - r**2)
    m_chi = 1.0e4 * m_tilde
    H_i, H_c = m_tilde, m_tilde / 100.0
    delta_N = 0.5 * np.log(H_i / H_c)

    A_threshold = np.sqrt(2.0) * m_tilde / (r * m_phi)
    A_i = A_threshold * np.exp(1.5 * delta_N)
    kappa = r * m_chi * m_phi
    R_i = kappa**2 * A_i**2 / (m_chi**3 * H_i)

    def derivative(N, scaled_fraction):
        amplitude = A_i * np.exp(-1.5 * N)
        resonance = R_i * np.exp(-N)
        occupation = np.expm1(0.5 * np.pi * resonance)
        source = occupation / (16.0 * np.pi**2 * amplitude**2)
        return -scaled_fraction + source

    solution = solve_ivp(
        derivative,
        (0.0, delta_N),
        (0.0,),
        rtol=1.0e-11,
        atol=1.0e-14,
        dense_output=True,
    )
    N = np.linspace(0.0, delta_N, 300)
    scaled_fraction = solution.sol(N)[0]
    return N, scaled_fraction, R_i, R_i * np.exp(-delta_N)


def main(output):
    resonance, occupation = exact_band_crossing()
    N, fraction, R_i, R_c = delayed_release_backreaction()

    plt.rcParams.update({"font.size": 9})
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.2))

    grid = np.linspace(0.0, 1.05, 300)
    axes[0].plot(grid, 0.5 * np.pi * grid, color="0.35", ls="--", lw=1,
                 label=r"$\pi R/2$")
    axes[0].plot(resonance, np.log1p(occupation), "o", ms=3.5,
                 color="#1f77b4", label="mode equation")
    axes[0].set_xlim(0.0, 1.05)
    axes[0].set_ylim(0.0, 1.75)
    axes[0].set_xlabel(r"$R=\kappa^2A^2/(m_\chi^3H)$")
    axes[0].set_ylabel(r"$\ln(1+n_k)$")
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].plot(N, 1.0e5 * fraction, color="#d95f02", lw=1.4)
    axes[1].set_xlim(0.0, N[-1])
    axes[1].set_ylim(0.0, 1.2)
    axes[1].set_xlabel(r"$N=\ln a$")
    axes[1].set_ylabel(
        r"$10^5\,(\rho_{\delta\phi}/\rho_\chi)(f/m_\chi)^2$"
    )
    axes[1].text(
        0.05,
        0.93,
        rf"$R_i={R_i:.2f}\;\to\;R_c={R_c:.2f}$",
        transform=axes[1].transAxes,
        va="top",
    )

    fig.tight_layout(h_pad=0.8)
    fig.savefig(output)
    print(f"R_i={R_i:.6g}, R_c={R_c:.6g}")
    print(
        "rho_fluc/rho_chi at release = "
        f"{fraction[-1]:.6g} (m_chi/f)^2"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "InhomogeneousBackreaction.pdf")
