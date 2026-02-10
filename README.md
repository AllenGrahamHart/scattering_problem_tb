# Scattering Problem Terminal Bench Science

This repo contains a forward/inverse model for powder neutron diffraction from stacking-disordered ice, plus a LaTeX problem statement intended for Terminal Bench Science.

## Layout
- `latex/problem.tex`: LaTeX source for the task write-up
- `powder_diffraction/`: forward model, Markov stacking disorder, powder averaging, and convolution
- `validation_plots/`: generated plots (if you run the plotting scripts)

## Build the LaTeX
From the repo root:

```bash
cd latex
pdflatex -interaction=nonstopmode -halt-on-error problem.tex
```

Run `pdflatex` a second time if you want references to resolve.
