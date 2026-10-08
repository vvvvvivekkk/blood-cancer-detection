# Kickoff prompt for Claude Code

Paste this as your first message in Claude Code, from inside the repo folder that already contains `CLAUDE.md`, `README.md`, and `ENDGOAL.md`.

---

Read `CLAUDE.md`, `ENDGOAL.md`, and `README.md` fully before doing anything.

We are building the project described there from scratch: a replication of the IEEE paper "Blood Cancer Identification using Hybrid Ensemble Deep Learning Technique" (ICEARS 2023) — a hybrid ensemble (weighted soft-voting + stacked meta-classifier) over three fine-tuned CNN backbones (MobileNetV2, EfficientNetB0, DenseNet121) on the Kaggle "Blood Cells Cancer (ALL)" 4-class dataset, plus a Streamlit app with Grad-CAM.

Work through the milestones in `ENDGOAL.md` strictly in order: M0 → M1 → M2 → M3 → M4 → M5 → M6 → M7. For each milestone:

1. Tell me in 3–5 lines what you are about to create.
2. Implement it.
3. Run the `--smoke` pipeline and `pytest -q`; paste the output.
4. Tick the milestone's boxes in `ENDGOAL.md` only for items you have actually verified.
5. Commit with a conventional commit message.
6. Stop and wait for my "next" before starting the following milestone.

Constraints that matter most:
- I don't have a GPU on this machine. Everything must run end-to-end on CPU in `--smoke` mode using `data/sample/`. Real training happens later on Colab via `notebooks/train_colab.ipynb` — build that notebook as a thin wrapper that calls `src/`, never duplicate logic in it.
- `src/config.py` is the single source of truth for `CLASS_NAMES`, paths, image size, seed, hyperparameters.
- The test split is read only by `src/evaluate.py`. Ensemble weights and the meta-classifier are fit on the validation split.
- Numbers in `results/` and the README results table must be produced by scripts, never typed by hand.
- Every Streamlit page must render with no data and no models present and show the button that fixes it.
- Keep it simple: Streamlit UI, pinned `requirements.txt`, no Docker unless I ask.

For M0, start by creating the exact repo layout from `CLAUDE.md`, `requirements.txt` (pinned, TF 2.15, Python 3.10), `.gitignore`, `src/config.py`, 6 tiny placeholder sample images in `data/sample/` (generate synthetic 64×64 cell-like blobs with PIL if needed and label them per class), and a first passing test. Then show me the tree and the test output.

Begin with M0.

---

## Follow-up prompts (use as you go)

**After each milestone:**
> next

**When real training is done on Colab and you've copied `models/` and `results/` back:**
> I've placed the trained `.keras` models and the Colab `results/` into the repo. Run `python -m src.ensemble`, then `python -m src.evaluate --write-readme-table`, regenerate all figures, update the README results table and `ENDGOAL.md` target-number checks, and tell me whether the ensemble beats the best single backbone. If not, propose and try the fallbacks listed in ENDGOAL.md.

**For the report:**
> Create `docs/paper_mapping.md` mapping each section of the paper (abstract → methodology → ensemble → results → conclusion) to the file and function in this repo that implements it, and export every figure in `results/figures/` plus an architecture diagram PNG into `docs/report_assets/`.

**If you get stuck on a Keras/TF API:**
> Check the installed TF version with `python -c "import tensorflow as tf; print(tf.__version__)"` and use only APIs that exist in that version. Don't guess layer names — print `model.summary()` or iterate `model.layers` to find the Grad-CAM target layer.
