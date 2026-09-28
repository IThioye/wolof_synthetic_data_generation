# Final report

This directory contains the clean, publishable report package.

- `PFE_Template_aivancity_EN.tex`: main LaTeX source.
- `references.bib`: bibliography database.
- `logo-aivancity-latex.png`, `pdfa.xmpi`, and
  `PFE_Template_aivancity_EN.xmpdata`: required compilation assets.
- `PFE_PGE5_THIOYE_Ibrahima_2026.pdf`: compiled submission PDF.

The local `Final Report/` directory is a working directory and is intentionally
ignored because it also contains auxiliary LaTeX files and intermediate PDFs.

## Compilation

From this directory, run:

```bash
pdflatex PFE_Template_aivancity_EN.tex
biber PFE_Template_aivancity_EN
pdflatex PFE_Template_aivancity_EN.tex
pdflatex PFE_Template_aivancity_EN.tex
```

The final source still uses the institution's template and should be reviewed
against its redistribution conditions before making the repository public.
