# Inverted Pendulum in the Sky — numerical codes

Numerical codes accompanying

> T. Numasawa and M. Yamazaki, *Inverted Pendulum in the Sky*.

The codes reproduce the figures and the numerical checks quoted in the paper.
All quantities are in the normalized units of the paper (the axion field is normalized by its decay constant, and V0 = m_phi^2).

## Requirements

Python 3.9+ with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

Each script runs standalone, e.g. `python plot_backgrounds.py Backgrounds.pdf`.
Longer runs use `concurrent.futures` to parallelize over parameters.

## Contents

Section, equation and figure numbers refer to arXiv v1 of the paper.

| Script | Paper |
|---|---|
| `kapitza_toy.py` | Sec. II: stabilization for b^2 > 2 V0 (Eq. (9)) with the cos(phi) coupling; Floquet threshold |
| `resonance.py` | Sec. II C, Eq. (13): parametric resonance across the first Mathieu band in de Sitter |
| `plot_poteff.py` | Fig. 1: effective potential |
| `coupled_axions.py`, `plot_spinning_chi.py` | Figs. 2, 3: spinning axion (Sec. III A) |
| `plot_floquet_A.py` | Fig. 4: Floquet exponent vs. amplitude (Sec. III B) |
| `spinning_top.py`, `plot_spinning_top.py` | Fig. 5: inverted spinning top (Sec. IV) |
| `backgrounds.py`, `plot_backgrounds.py` | Fig. 6 and the background independence of the stabilized phase (Secs. V A, V B); the figure uses `fastkapitza.py` |
| `fastkapitza.py` | Fast fixed-step RK4 integrator (numba) for the model of Sec. III B with large m_chi, used for Figs. 6 and 7 |
| `release_airy.py` | Sec. V B, Eq. (38): Airy matching at the release (m_chi = 1e5, `fastkapitza.py`) |
| `overdamped.py` | Sec. V C, Eq. (42): overdamped regime in radiation domination |
| `delayed_fast.py`, `plot_delayed.py`, `delayed_data.txt` | Sec. V C, Eq. (43) and Fig. 7: delayed release (m_chi = 3e4; `delayed.py` is the scipy version with m_chi = 1000) |
| `scaling.py` | Sec. V D, Eq. (45): general classification (freeze-out for omega_eff ~ a^-sigma, H ~ a^-gamma) |
| `isocurvature.py`, `dephasing.py`, `dlnn.py`, `scan_abundance.py` | Sec. VI: nonlinear dephasing, d ln n / d ln delta_i, and the scan of n(delta_i) (m_chi = 1e4) |
| `floquet_chi.py`, `plot_floquet_chi.py`, `plot_msd.py` | Sec. III B: the non-hierarchical example with a periodic coupling cos(chi/g) |

## License

The code in this repository is released under the MIT License (see `LICENSE`).
