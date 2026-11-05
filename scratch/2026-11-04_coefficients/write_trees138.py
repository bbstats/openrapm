"""params/booster_params_trees138.json: today's LightGBM settings with defense's trees at 330 instead of 130."""
import json
from pathlib import Path

spec = json.loads(Path("params/booster_params_lgb1_tune2.json").read_text(encoding="utf-8"))
assert spec["D"]["params"]["n_estimators"] == 130
spec["D"]["params"]["n_estimators"] = 330
spec["note"] = ("experiment 138, 2026-11-04: lgb1_tune2 with defense's trees at 330 instead of 130, which reach 95% of what "
                "the trees could fit instead of 69%; offense's as they are")
Path("params/booster_params_trees138.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
