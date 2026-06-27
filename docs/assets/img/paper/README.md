# Paper figure assets for GitHub Pages

Export figures from the PDF and save them here to show on the project website.

## Recommended files

| File | Suggested paper source | Used in |
|------|------------------------|---------|
| `teaser.png` | Main method / teaser figure | Hero or Paper Figures section |
| `method.png` | Architecture diagram | How it works |
| `results.png` | Main results table or bar chart | Key Findings |
| `qualitative.png` | Qualitative examples (optional) | Results section |

## Export tips

- PNG, **1600 px** wide (or 2× PDF crop) for sharp display
- Transparent background optional for method diagrams
- After adding files, edit `docs/index.html` and replace placeholder `src` paths, e.g.:

```html
<img src="assets/img/paper/teaser.png" alt="SpatioRoute method figure from paper" />
```

## Current status

The site uses **SVG diagrams** in `docs/assets/img/` until paper PNGs are added here.
