# Fuzzy set–based autonomous navigation — implementation & analysis

Re-implementation of E. A. Rastkhiz, H. Schwartz, I. Lambadaris,
*A fuzzy set-based methodology for autonomous navigation*, Fuzzy Sets and Systems 518 (2025) 109485
(the PDF in this folder is `Fuzzy logic paper`).

## Layout

| Path | Contents |
|---|---|
| `fppa/fuzzy.py` | BMFs (6), (8); Yager t-conorm (1), (7); saturations (`exact`, paper's `tanh`, range-preserving `softmin`); analytic gradients |
| `fppa/environment.py` | Atomic obstacles (2)–(5), fuzzy path (8)–(12), environment (13)–(16) |
| `fppa/planner.py` | Contour level (17), speed bound (18), FPPA update (22), Algorithm 1 loop, APF baseline, unicycle signals (23) |
| `experiments/scenarios.py` | Case studies 1–3 (Sec. 4) |
| `experiments/make_figures.py` | All figures + `figures/results.json` |
| `experiments/test_gradients.py` | Analytic vs. finite-difference gradient check |
| `slides/fppa_slides.tex` / `.pdf` | Beamer deck (compile from inside `slides/`) |

## Run

```bash
python3 -m experiments.test_gradients     # max rel. error ~2e-9
python3 -m experiments.make_figures       # ~2.5 min, writes figures/
cd slides && pdflatex fppa_slides.tex && pdflatex fppa_slides.tex
```

Requires numpy, scipy, matplotlib (and Pillow for the GIF).

## Assumptions (not stated in the paper)

- Case 1 path along x2 = 0 (as in Fig. 6; the text says x2 = 20).
- Case 2 global path digitised from Fig. 11; moving obstacles: O6 velocity (0, −3), O7 velocity (2, 0).
- Case 2/3 use b = 1.25 and γ = 0.75 from Case 1 (not listed for Case 2).
- Eq. (10) is implemented as H D Hᵀ (consistent with eq. (3)).
- The environment is evaluated as one m-ary union of all atoms, because the tanh-smoothed operator is not associative.
- Baseline is a generic path-guided APF, **not** the IAPF of the paper's ref. [37].
- Simulation stops when the projection on the path is within 1.5 segments of the end (level curves are closed).

## Main findings

- With the paper's `tanh` smoothing and γ = 0.75, the path core only reaches
  tanh(γ√(2π)) = 0.9545 (Poisson summation), so α = 0.99 is an infeasible set-point.
  The vehicle rides the ridge; in Case 3 it collides with O2 at t = 6.4 s.
- A feasible α (0.90), or the range-preserving `softmin` saturation with α = 0.99, completes all cases.
  Case 1: 78.1 s (paper ≈ 78 s). The predicted lateral offset is 1.1325 m; the simulated offset is 1.1341 m.
- The APF baseline stalls inside the concave obstacle at (96.1, 0).
