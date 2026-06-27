# Publishing to GitHub

This folder is intended to become the standalone repository:

**https://github.com/PudPawat/SPATIOROUTE-Dynamic-Prompt-Routing-for-Zero-Shot-Spatial-Reasoning**

## 1. Prepare a self-contained copy (recommended)

From `experiments/spatioroute/`:

```bash
chmod +x scripts/prepare_github_release.sh
./scripts/prepare_github_release.sh /path/to/Qwen_playground
```

This copies `videolm/`, `utils/`, and evaluation scripts into `vendor/` so the repo runs without the full playground.

## 2. Initialize git and push

```bash
cd experiments/spatioroute   # or your exported copy of this folder as repo root

git init
git add .
git commit -m "Initial release: SpatioRoute code and project page"

git branch -M main
git remote add origin git@github.com:PudPawat/SPATIOROUTE-Dynamic-Prompt-Routing-for-Zero-Shot-Spatial-Reasoning.git
git push -u origin main
```

If the GitHub repo already exists with a README, pull first:

```bash
git pull origin main --rebase
git push -u origin main
```

## 3. Enable GitHub Pages

1. Open **Settings → Pages** on the GitHub repo.
2. **Build and deployment → Source**: Deploy from a branch.
3. **Branch**: `main` / folder **`/docs`**.
4. Save. After ~1–2 minutes the site will be live at:

   **https://pudpawat.github.io/SPATIOROUTE-Dynamic-Prompt-Routing-for-Zero-Shot-Spatial-Reasoning/**

(Optional) Add a custom domain under Pages settings.

## 4. Add paper figures to the website

1. Export PNGs from the paper PDF (see `docs/assets/img/paper/README.md`).
2. Save as e.g. `docs/assets/img/paper/teaser.png`.
3. Edit `docs/index.html` — replace placeholder `<img src="...">` under **Paper Figures**.
4. Commit and push; Pages redeploys automatically.

## 5. Repository layout note

When this folder is the **root** of the GitHub repo, Python modules live at:

```
routing/
preprompts/
eval/
experiments/spatioroute/   ← only if you keep nested layout; see below
```

### Option A — Push this folder as repo root (recommended)

Move contents of `experiments/spatioroute/*` to the repo root before `git init`, and update imports from `experiments.spatioroute.*` to top-level packages (or keep `experiments/spatioroute` path and set `PYTHONPATH=.`).

### Option B — Keep nested `experiments/spatioroute/`

Run all commands from repo root with:

```bash
export PYTHONPATH="${PWD}:${PWD}/vendor:${PYTHONPATH}"
python -m experiments.spatioroute.preprompts.generate_rule ...
```

The included `prepare_github_release.sh` sets up `vendor/` for Option B.

## 6. Badges (optional, add to README)

```markdown
[![Paper](https://img.shields.io/badge/arXiv-2605.18209-b31b1b)](https://arxiv.org/abs/2605.18209)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://pudpawat.github.io/SPATIOROUTE-Dynamic-Prompt-Routing-for-Zero-Shot-Spatial-Reasoning/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
```
