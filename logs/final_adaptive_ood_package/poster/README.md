# Adaptive OOD Poster

This directory is now self-contained with the template dependencies from `CCN19_poster_memory_model_QL.zip`:

- `beamerposter.sty`
- `beamerthemeconfposter.sty`
- `imgs/pton-shield.png`
- original template reference files: `main.tex`, `sample.bib`, `scratch.tex`

The actual project poster is:

- `adaptive_ood_poster.tex`

Compile from this directory so the local theme and relative figure paths resolve:

```bash
cd logs/final_adaptive_ood_package/poster
pdflatex -interaction=nonstopmode adaptive_ood_poster.tex
pdflatex -interaction=nonstopmode adaptive_ood_poster.tex
```

If you prefer `latexmk`:

```bash
cd logs/final_adaptive_ood_package/poster
latexmk -pdf -interaction=nonstopmode adaptive_ood_poster.tex
```

The poster uses figures from:

- `../figures/`
- `../prefix_uniqueness/figures/`
- `../dataset_ood/figures/`

I kept the fallback theme as `beamerthemeconfposter.fallback.sty`, but the active poster now uses the real template theme from the zip.
