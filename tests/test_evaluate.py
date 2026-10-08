import json

from src import config as C
from src import evaluate

KEYS = C.BACKBONES + C.ENSEMBLES


def test_metrics_json_complete(smoke):
    m = json.loads(C.METRICS_JSON.read_text())
    assert list(m["models"]) == KEYS
    for v in m["models"].values():
        for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision",
                  "weighted_recall", "weighted_f1", "kappa", "roc_auc_ovr", "per_class", "confusion_matrix"):
            assert k in v
        assert set(v["per_class"]) == set(C.CLASS_NAMES)
        assert len(v["confusion_matrix"]) == 4
    assert m["best_ensemble"] in C.ENSEMBLES
    assert C.MODE_FILE.read_text() == m["best_ensemble"]
    assert m["meta"]["smoke"] is True and m["meta"]["n_test"] == 4


def test_figures_and_tables_written(smoke):
    figs = {p.name for p in C.FIGURES_DIR.iterdir()}
    want = {f"cm_{k}.png" for k in KEYS} | {f"training_curves_{n}.png" for n in C.BACKBONES}
    want |= {"roc_ensemble.png", "model_comparison_bar.png", "gradcam_examples.png"}
    assert want <= figs
    table = (C.RESULTS_DIR / "comparison_table.md").read_text(encoding="utf-8")
    assert all(C.DISPLAY_NAMES[k] in table for k in KEYS)
    assert "Target" in (C.RESULTS_DIR / "NOTES.md").read_text(encoding="utf-8")


def test_readme_injection_replaces_only_between_markers(smoke, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{evaluate.START}\nold\n{evaluate.END}\noutro\n", encoding="utf-8")
    evaluate.write_readme_table(readme=readme)
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n") and text.endswith("outro\n") and "old" not in text
    assert "| Ensemble-Stacking" in text and text.count(evaluate.START) == 1


def test_only_evaluate_reads_test_csv():
    readers = sorted(p.name for p in (C.REPO / "src").glob("*.py") if '"test.csv"' in p.read_text(encoding="utf-8"))
    assert readers == ["evaluate.py"]
