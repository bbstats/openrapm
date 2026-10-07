"""The shot-quality search's learner plug-ins: the tuned regression and boosted trees fitted ON TOP of it.

Every learner returns a `Fitted` whose `predict_raw(f)` is the logit of a league-average shooter in a league-average
arena BEFORE the relevel -- the same quantity shotmodel.ShotModel.predict_raw gives -- so the harness (shotsearch:
relevel_fast, SALL) scores every family the same way.  The registered protocol is DECISIONS.md, "The shot-quality
search"; the spec's section 2 is the design.

Why the trees learn a CORRECTION and never the make itself.  Trees trained on makes lost to the regression on the
tracked seasons (48.6 / 56.6 squared points against 38.7 / 44.8, scripts/126) because they learned who takes which
shots: good shooters take the hard ones, so hard shots look easier than they are.  The regression avoids that with a
shooter-season and an arena-season offset, fitted and then set to zero when pricing.  Trees cannot carry such a term
(a shooter id is an input they would interact with), so they start from the regression's margin WITH its offsets:

    training margin   base.margin_train(train, with_offsets=True)    X.beta + shooter-season + arena-season
    tree inputs       shotfeatures.tree_features(f) + 'glm'           'glm' = X.beta, the offset-free logit
    pricing           base.predict_raw(f) + trees(f)                  offsets zeroed; trees only, no margin

A good shooter's make is already explained by his offset, so the trees gain nothing by proxying who shoots.  The
'nooff' control starts the trees from X.beta alone (no offsets) and shows the trap in the ablation table.

The four libraries, each told the margin its own way:
    xgb      DMatrix(base_margin=); base_score fixed at 0.5 (logit 0), so a DMatrix WITHOUT a base margin predicts
             the trees alone with output_margin=True; device cuda when the build and the card allow it
    lgbm     Dataset(init_score=), boost_from_average off; predict(raw_score=True) without an init score is the trees
    cat      Pool(baseline=), boost_from_average off; RawFormulaVal on rows without a baseline is the trees; GPU when
             it accepts a baseline, else CPU
    chimera  no offset argument at all (booster.py starts from loss.init), so the margin travels INSIDE the label:
             y = margin + 1000 * made, unpacked by OffsetLogloss.  Never pass cat_features (the ordered target
             encoder would read the packed label) or random_effects (RMSE only).  The class lives at module level
             because a bag's members fit in spawned worker processes that must import and unpickle it.

Early stopping is on the rows the driver marks `holdout` (the two allowed training seasons nearest the block, the
owner's one-holdout rule); they are left out of the trees' fit.  `refit=True` (finalists only) then refits on every row
at best_iter x n_all / n_fit rounds, the learning rate pinned.  Sample weights (closeness) are scaled to mean 1 on the
fitted rows, so a booster's Hessian-unit knobs mean the same in every library (chimeraboost normalises internally).

Hyperparameters use each library's native names.  SPACES holds the spec's search ranges and `suggest(trial, kind)`
turns them into optuna suggestions.  Penalties are entered per FULL_ROWS (1.5M) training rows; `scale_penalties`
rescales them for a smaller sample (the driver calls it; the fit functions take their params literally).

Model layer: frames in, arrays out.
"""
from __future__ import annotations

import functools
import inspect
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import shotfeatures
from . import shotmodel as sm

try:                                    # chimeraboost's documented base class; the objective must subclass it
    from chimeraboost import CustomObjective as _ChimeraObjective
except ImportError:                     # pragma: no cover - chimeraboost is installed in .venv
    _ChimeraObjective = object

BOOSTERS = ("xgb", "lgbm", "cat", "chimera")
MAX_ROUNDS = 5000                       # an upper bound only: early stopping on the holdout picks the count
PATIENCE = 100                          # rounds without a better holdout loss (a tiny signal plateaus slowly)
FULL_ROWS = 1_500_000                   # the row budget penalties are entered for (shotsearch.FULL_ROWS)
PACK = 1000.0                           # chimera's packed label: y = margin + PACK * made
GLM_COLUMN = "glm"


# ------------------------------------------------------------------------------------------ the fitted learner
@dataclass
class Fitted:
    """One fitted learner.  `model` is the regression whose offset-free logit every prediction starts from; a
    booster adds its trees' correction to it.  `info` holds what the driver logs (rows, rounds, holdout losses,
    fit seconds, device)."""
    kind: str
    model: sm.ShotModel
    booster: object = None
    columns: tuple = ()
    features: Callable | None = None
    params: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)

    def correction(self, f: pd.DataFrame) -> np.ndarray:
        """The trees' logit correction alone (zero for the regression): the 'f' a stack scales by alpha."""
        return self._parts(f)[1]

    def predict_raw(self, f: pd.DataFrame) -> np.ndarray:
        """The offset-free logit before the relevel: the regression's X.beta plus the trees' correction."""
        eta, corr = self._parts(f)
        return eta + corr

    def _parts(self, f):
        eta = self.model.predict_raw(f)                       # refuses a season it trained on, or a neighbour
        if self.booster is None:
            return eta, np.zeros(len(f))
        A, cols = tree_inputs(f, eta, self.features)
        if tuple(cols) != tuple(self.columns):
            raise ValueError(f"the tree inputs changed since the fit: {cols} against {list(self.columns)}")
        return eta, _PREDICT[self.kind](self.booster, A)


# ------------------------------------------------------------------------------------------ the regression
def glm_takes() -> set:
    """The keyword arguments shotmodel.fit accepts besides the frame, the blocks and the weights."""
    return set(inspect.signature(sm.fit).parameters) - {"train", "blocks", "weights"}


def fit_glm(train: pd.DataFrame, w, params: dict | None = None) -> Fitted:
    """shotmodel.fit with the keys of `params` it accepts (read from its signature, so its options pass straight
    through: standardise, ridge_int, era_lambda, knots, inter, the half-lives and the `block` they need); `blocks`
    defaults to shotmodel.BLOCKS, offsets on.  Keys fit does not take are listed in info["ignored"]."""
    params = dict(params or {})
    sig = inspect.signature(sm.fit).parameters
    var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.values())
    takes = glm_takes()
    kw, ignored = {"offsets": True}, []
    for k, v in params.items():
        if k == "blocks":
            continue
        if var_kw or k in takes:
            kw[k] = v
        else:
            ignored.append(k)
    blocks = tuple(params.get("blocks", sm.BLOCKS))
    t0 = time.time()
    model = sm.fit(train, blocks, weights=w, **kw)
    return Fitted(kind="glm", model=model, params=params,
                  info=dict(n_fit=len(train), fit_seconds=time.time() - t0, ignored=ignored, passed=sorted(kw)))


# ------------------------------------------------------------------------------------------ the tree inputs
def tree_inputs(f: pd.DataFrame, glm: np.ndarray, features: Callable | None = None) -> tuple[np.ndarray, list]:
    """The trees' input matrix (float32, NaN kept as missing) and its column names: `features(f)` (default
    shotfeatures.tree_features, the whitelist), the sub-model code if those lack a 'sub' column (one joint model,
    the sub-model a feature), and the regression's offset-free logit as the column 'glm'.  The names are checked
    against shotfeatures.FORBIDDEN again here, because a learner may be handed any feature function."""
    X = (features or shotfeatures.tree_features)(f)
    if not isinstance(X, pd.DataFrame):
        raise TypeError("a feature function must return a DataFrame with named columns")
    if len(X) != len(f):
        raise ValueError(f"the feature function returned {len(X)} rows for {len(f)}")
    cols = [str(c) for c in X.columns]
    shotfeatures.check_names(cols)                          # raises on an identity, a scorer tag, an outcome
    if GLM_COLUMN in cols:
        raise ValueError(f"the feature function may not supply '{GLM_COLUMN}': it is the regression's logit")
    parts = [X.to_numpy(np.float32)]
    if "sub" not in cols:
        sub = sm.submodel_of(f)
        parts.append(np.where(sub == "rim", 0.0, np.where(sub == "mid", 1.0, 2.0)).astype(np.float32)[:, None])
        cols.append("sub")
    parts.append(np.asarray(glm, dtype=np.float32)[:, None])
    cols.append(GLM_COLUMN)
    return np.ascontiguousarray(np.hstack(parts), dtype=np.float32), cols


# ------------------------------------------------------------------------------------------ chimeraboost's packed label
def pack(margin, made) -> np.ndarray:
    """chimeraboost's label: the base margin travels inside it, y = margin + 1000 * made."""
    m = np.asarray(margin, dtype=np.float64)
    if not np.all(np.isfinite(m)) or np.max(np.abs(m), initial=0.0) >= PACK / 2:
        raise ValueError("a packed margin must be finite and inside +-500, or the unpacking misreads the make")
    return m + PACK * np.asarray(made, dtype=np.float64)


def unpack(y) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    made = (y > PACK / 2).astype(np.float64)
    return made, y - PACK * made


class OffsetLogloss(_ChimeraObjective):
    """Binary log loss with a per-row offset, for chimeraboost (which has no init score).  The label is
    pack(margin, made); p = sigmoid(raw + margin); grad = p - made, hess = max(p(1 - p), 1e-6) as its built-in
    Logloss; eval = the weighted mean log loss; init 0 and the identity transform, so predict() returns the trees'
    logit correction alone.  Stateless and at module level: bag members unpickle it in worker processes."""
    name = "OffsetLogloss"
    is_classification = False
    adjusts_leaves = False

    def init(self, y, sample_weight=None):
        return 0.0

    def grad_hess(self, y, raw):
        made, off = unpack(y)
        p = np.exp(-np.logaddexp(0.0, -(raw + off)))
        return p - made, np.maximum(p * (1.0 - p), 1e-6)

    def eval(self, y, raw, sample_weight=None):
        made, off = unpack(y)
        z = raw + off
        return float(np.average(np.logaddexp(0.0, z) - made * z, weights=sample_weight))

    def transform(self, raw):
        return raw


# ------------------------------------------------------------------------------------------ devices
@functools.lru_cache(maxsize=None)
def xgb_device() -> str:
    """'cuda' when xgboost was built with CUDA and a one-round fit on the card works, else 'cpu'."""
    import xgboost as xgb
    if not xgb.build_info().get("USE_CUDA"):
        return "cpu"
    try:
        d = xgb.DMatrix(np.array([[0.0], [1.0], [0.0], [1.0]]), label=[0, 1, 0, 1], base_margin=np.zeros(4))
        xgb.train(dict(device="cuda", tree_method="hist", objective="binary:logistic", base_score=0.5,
                       verbosity=0), d, num_boost_round=1)
        return "cuda"
    except Exception:
        return "cpu"


@functools.lru_cache(maxsize=None)
def cat_task_type() -> str:
    """'GPU' when CatBoost sees a card and accepts a baseline there (the spec's condition), else 'CPU'."""
    try:
        from catboost import CatBoost, Pool
        from catboost.utils import get_gpu_device_count
        if get_gpu_device_count() < 1:
            return "CPU"
        rng = np.random.default_rng(0)
        X = rng.normal(size=(256, 2))
        y = (rng.random(256) < 0.5).astype(int)
        m = CatBoost(dict(loss_function="Logloss", boost_from_average=False, iterations=2, task_type="GPU",
                          verbose=False, allow_writing_files=False, gpu_ram_part=0.1))
        m.fit(Pool(X, label=y, baseline=rng.normal(size=256)))
        return "GPU"
    except Exception:
        return "CPU"


# ------------------------------------------------------------------------------------------ the four libraries
def _pop(p: dict, names, default):
    """Remove every alias in `names` from p; the first present wins."""
    out = default
    found = False
    for n in names:
        if n in p:
            v = p.pop(n)
            if not found:
                out, found = v, True
    return out


@dataclass
class _Rows:
    A: np.ndarray
    y: np.ndarray
    w: np.ndarray
    m: np.ndarray


def _xgb_fit(fit: _Rows, ho: _Rows | None, params: dict, cols, rounds: int | None):
    import xgboost as xgb
    p = dict(params)
    n_max = int(_pop(p, ("num_boost_round", "n_estimators"), MAX_ROUNDS))
    patience = int(_pop(p, ("early_stopping_rounds",), PATIENCE))
    q = dict(tree_method="hist", device=xgb_device(), eval_metric="logloss", seed=0, verbosity=0)
    q.update(p)
    q.update(objective="binary:logistic", base_score=0.5)        # logit(0.5) = 0: the trees alone, no constant
    dfit = xgb.DMatrix(fit.A, label=fit.y, weight=fit.w, base_margin=fit.m, feature_names=list(cols))
    if rounds is None:
        dho = xgb.DMatrix(ho.A, label=ho.y, weight=ho.w, base_margin=ho.m, feature_names=list(cols))
        bst = xgb.train(q, dfit, num_boost_round=n_max, evals=[(dho, "holdout")],
                        early_stopping_rounds=patience, verbose_eval=False)
        best = int(bst.best_iteration) + 1
        lib = float(bst.best_score)
        bst = bst[:best]
    else:
        bst = xgb.train(q, dfit, num_boost_round=int(rounds), verbose_eval=False)
        best, lib = int(rounds), None
    bst.set_param({"device": "cpu"})                              # price anywhere, no device-mismatch warnings
    return bst, best, lib, q["device"]


def _xgb_predict(bst, A):
    import xgboost as xgb
    return bst.predict(xgb.DMatrix(A, feature_names=bst.feature_names), output_margin=True).astype(np.float64)


_LGB_ROUNDS = ("num_iterations", "num_iteration", "n_iter", "num_tree", "num_trees", "num_round", "num_rounds",
               "nrounds", "num_boost_round", "n_estimators", "max_iter")
_LGB_PATIENCE = ("early_stopping_round", "early_stopping_rounds", "early_stopping", "n_iter_no_change")


def _lgbm_fit(fit: _Rows, ho: _Rows | None, params: dict, cols, rounds: int | None):
    import lightgbm as lgb
    p = dict(params)
    n_max = int(_pop(p, _LGB_ROUNDS, MAX_ROUNDS))
    patience = int(_pop(p, _LGB_PATIENCE, PATIENCE))
    q = dict(metric="binary_logloss", verbosity=-1, num_threads=12, seed=0)
    q.update(p)
    q.update(objective="binary", boost_from_average=False)
    dfit = lgb.Dataset(fit.A, label=fit.y, weight=fit.w, init_score=fit.m, feature_name=list(cols),
                       params=q, free_raw_data=False)
    if rounds is None:
        dho = lgb.Dataset(ho.A, label=ho.y, weight=ho.w, init_score=ho.m, reference=dfit, params=q)
        bst = lgb.train(q, dfit, num_boost_round=n_max, valid_sets=[dho], valid_names=["holdout"],
                        callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)])
        best = int(bst.best_iteration)
        lib = float(next(iter(bst.best_score["holdout"].values())))          # the first (stopping) metric
        bst = lgb.Booster(model_str=bst.model_to_string(num_iteration=best))
    else:
        bst = lgb.train(q, dfit, num_boost_round=int(rounds))
        best, lib = int(rounds), None
    return bst, best, lib, "cpu"


def _lgbm_predict(bst, A):
    return bst.predict(A, raw_score=True).astype(np.float64)       # no init score at predict: the trees alone


def _cat_fit(fit: _Rows, ho: _Rows | None, params: dict, cols, rounds: int | None):
    from catboost import CatBoost, Pool
    p = dict(params)
    n_max = int(_pop(p, ("iterations", "num_boost_round", "n_estimators", "num_trees"), MAX_ROUNDS))
    patience = int(_pop(p, ("od_wait", "early_stopping_rounds"), PATIENCE))
    q = dict(eval_metric="Logloss", random_seed=0, verbose=False, allow_writing_files=False,
             task_type=cat_task_type())
    q.update(p)
    q.update(loss_function="Logloss", boost_from_average=False)
    if q.get("bootstrap_type", "Bayesian") != "Bayesian":
        q.pop("bagging_temperature", None)          # CatBoost allows it with the Bayesian bootstrap only
    pfit = Pool(fit.A, label=fit.y, weight=fit.w, baseline=fit.m, feature_names=list(cols))
    if rounds is None:
        pho = Pool(ho.A, label=ho.y, weight=ho.w, baseline=ho.m, feature_names=list(cols))
        model = CatBoost(dict(q, iterations=n_max, od_type="Iter", od_wait=patience, use_best_model=True))
        model.fit(pfit, eval_set=pho)
        best = int(model.get_best_iteration()) + 1
        lib = float(model.get_best_score()["validation"][q["eval_metric"]])
    else:
        model = CatBoost(dict(q, iterations=int(rounds)))
        model.fit(pfit)
        best, lib = int(rounds), None
    return model, best, lib, q["task_type"]


def _cat_predict(model, A):
    return np.asarray(model.predict(A, prediction_type="RawFormulaVal"), dtype=np.float64)


_CHIMERA_BANNED = ("cat_features", "random_effects", "loss", "groups", "eval_metric")


def _chimera_fit(fit: _Rows, ho: _Rows | None, params: dict, cols, rounds: int | None, lr: float | None = None):
    from chimeraboost import ChimeraBoostRegressor
    p = dict(params)
    bad = [k for k in _CHIMERA_BANNED if k in p]
    if bad:
        raise ValueError(f"{bad} may not be set with the packed-label objective (the target encoder and the "
                         "random effects would read the packed label)")
    n_max = int(_pop(p, ("n_estimators",), MAX_ROUNDS))
    patience = int(_pop(p, ("early_stopping_rounds",), PATIENCE))
    p.setdefault("random_state", 0)
    if rounds is None:
        est = ChimeraBoostRegressor(loss=OffsetLogloss(), n_estimators=n_max, early_stopping_rounds=patience, **p)
        est.fit(fit.A, pack(fit.m, fit.y), eval_set=(ho.A, pack(ho.m, ho.y), ho.w), sample_weight=fit.w)
        if est.estimators_ is not None:                     # a bag: members stop on the same shared holdout
            best = int(round(np.median([m.best_iteration_ for m in est.estimators_])))
            lib = None
        else:
            best = int(est.best_iteration_)
            hist = est.validation_history_
            lib = float(min(hist)) if len(hist) else None
    else:
        if lr is not None:
            p["learning_rate"] = lr                         # the rate that chose the count, not the no-stop default
        est = ChimeraBoostRegressor(loss=OffsetLogloss(), n_estimators=int(rounds), early_stopping=False, **p)
        est.fit(fit.A, pack(fit.m, fit.y), sample_weight=fit.w)
        best, lib = int(rounds), None
    return est, best, lib, "cpu"


def _chimera_lr(est) -> float:
    model = est.estimators_[0].model_ if est.estimators_ is not None else est.model_
    return float(model.lr_)


def _chimera_predict(est, A):
    return np.asarray(est.predict_raw(A), dtype=np.float64)       # init 0, identity transform: the trees alone


_FIT = {"xgb": _xgb_fit, "lgbm": _lgbm_fit, "cat": _cat_fit, "chimera": _chimera_fit}
_PREDICT = {"xgb": _xgb_predict, "lgbm": _lgbm_predict, "cat": _cat_predict, "chimera": _chimera_predict}


def _logloss(y, eta, w) -> float:
    return float(np.average(np.logaddexp(0.0, eta) - y * eta, weights=w))


# ------------------------------------------------------------------------------------------ the boosters
def fit_booster(kind: str, train: pd.DataFrame, w, params: dict | None, base, holdout: np.ndarray | None,
                *, offsets: bool = True, refit: bool = False, features: Callable | None = None) -> Fitted:
    """Boosted trees learning a correction on top of `base` (a ShotModel fitted on `train`, or its Fitted).

    The training margin is base.margin_train(train, with_offsets=True): X.beta plus the shooter-season and
    arena-season offsets, so the trees cannot learn who shoots (`offsets=False` is the control that can).  The
    inputs are tree_inputs: the whitelisted features plus 'glm' = X.beta without offsets.  `holdout` marks the
    early-stopping rows (left out of the fit); without one, params must name the round count.  `refit=True` refits
    on every row at best_iter x n_all / n_fit rounds.  Prediction: base.predict_raw(f) + the trees alone.
    A kind named '<kind>_nooff' is the control: the same as offsets=False."""
    if kind.endswith("_nooff"):
        kind, offsets = kind.removesuffix("_nooff"), False
    if kind not in _FIT:
        raise ValueError(f"unknown booster {kind!r}; one of {sorted(_FIT)}")
    if isinstance(base, Fitted):
        base = base.model
    if not isinstance(base, sm.ShotModel):
        raise TypeError("a booster needs the regression it corrects: a ShotModel fitted on these rows")
    params = dict(params or {})
    n = len(train)
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    hold = np.zeros(n, bool) if holdout is None else np.asarray(holdout, dtype=bool)
    if len(w) != n or len(hold) != n:
        raise ValueError("w and holdout must have one entry per training row")
    if hold.all():
        raise ValueError("every row is in the holdout: nothing left to fit")

    t0 = time.time()
    glm = base.margin_train(train, with_offsets=False)
    margin = base.margin_train(train, with_offsets=True) if offsets else glm
    A, cols = tree_inputs(train, glm, features)
    y = train["made"].to_numpy(np.float64)
    fitted = ~hold

    def rows(m, norm):
        """The rows in mask m, weights scaled to mean 1 over the rows in `norm` (the ones the trees fit)."""
        return _Rows(A=A[m], y=y[m], w=w[m] / w[norm].mean(), m=margin[m])

    if hold.any():
        ho = rows(hold, fitted)
        booster, best, lib, device = _FIT[kind](rows(fitted, fitted), ho, params, cols, None)
        corr = _PREDICT[kind](booster, ho.A)
        info = dict(holdout_loss=_logloss(ho.y, ho.m + corr, ho.w), holdout_loss_lib=lib,
                    holdout_loss_base=_logloss(ho.y, ho.m, ho.w))
    else:
        if not any(k in params for k in _ROUND_KEYS[kind]):
            raise ValueError(f"no holdout to stop on: give the round count ({_ROUND_KEYS[kind][0]}) explicitly")
        booster, best, lib, device = _FIT[kind](rows(fitted, fitted), None, params, cols,
                                                int(_pop(dict(params), _ROUND_KEYS[kind], MAX_ROUNDS)))
        info = {}
    rounds = best
    if refit and hold.any():
        rounds = max(1, int(round(best * n / fitted.sum())))
        every = np.ones(n, bool)
        extra = dict(lr=_chimera_lr(booster)) if kind == "chimera" else {}
        booster, _, _, device = _FIT[kind](rows(every, every), None, params, cols, rounds, **extra)
    info.update(n_fit=int(fitted.sum()), n_holdout=int(hold.sum()), best_iter=int(best), rounds=int(rounds),
                refit=bool(refit and hold.any()), offsets=bool(offsets), device=device,
                fit_seconds=time.time() - t0)
    return Fitted(kind=kind, model=base, booster=booster, columns=tuple(cols), features=features, params=params,
                  info=info)


_ROUND_KEYS = {"xgb": ("num_boost_round", "n_estimators"), "lgbm": _LGB_ROUNDS,
               "cat": ("iterations", "num_boost_round", "n_estimators", "num_trees"), "chimera": ("n_estimators",)}


# ------------------------------------------------------------------------------------------ the registry
def _glm_entry(train, w, params, base=None, holdout=None, **_):
    return fit_glm(train, w, params)


def _booster_entry(kind, offsets):
    def entry(train, w, params, base=None, holdout=None, **kw):
        return fit_booster(kind, train, w, params, base, holdout, offsets=offsets, **kw)
    entry.__name__ = f"fit_{kind}{'' if offsets else '_nooff'}"
    return entry


# fit(train, w, params, base=None, holdout=None, **kw) -> Fitted, one per family; '<kind>_nooff' is the control
LEARNERS: dict = {"glm": _glm_entry}
for _k in BOOSTERS:
    LEARNERS[_k] = _booster_entry(_k, True)
    LEARNERS[f"{_k}_nooff"] = _booster_entry(_k, False)


def fit(kind: str, train: pd.DataFrame, w, params: dict | None = None, base=None, holdout=None, **kw) -> Fitted:
    """LEARNERS[kind](train, w, params, base, holdout, **kw)."""
    return LEARNERS[kind](train, w, params, base=base, holdout=holdout, **kw)


# ------------------------------------------------------------------------------------------ the search spaces
@dataclass(frozen=True)
class P:
    """One search dimension: 'float' / 'int' over [lo, hi] (log scale if `log`), or 'cat' over `choices`;
    `when` = (parent, value) draws it only when the parent took that value."""
    type: str
    lo: float = 0.0
    hi: float = 1.0
    log: bool = False
    choices: tuple = ()
    when: tuple | None = None


GLM_INTER = tuple(getattr(sm, "INTER_BLOCKS", ()))     # spot2d, start_x_time, clock_x_dist, putback_x_dist, context_x

# The spec's ranges (section 2).  Penalties are per FULL_ROWS training rows (see scale_penalties).  The regression's
# helper dimensions (decay_*, era, knots_*, clock, inter_*, feat_*) are turned into shotmodel.fit's own arguments by
# suggest; every other name is a shotmodel.fit keyword.
SPACES: dict = {
    "glm": {
        "decay_past": P("cat", choices=(True, False)),
        "half_life_past": P("float", 1.0, 30.0, log=True, when=("decay_past", True)),
        "decay_future": P("cat", choices=(True, False)),
        "half_life_future": P("float", 1.0, 30.0, log=True, when=("decay_future", True)),
        "ridge": P("float", 1e-3, 100.0, log=True),
        "ridge_int": P("float", 0.1, 1e3, log=True),
        "era": P("cat", choices=(False, True)),
        "era_lambda": P("float", 1.0, 1e4, log=True, when=("era", True)),
        "lam_shooter": P("float", 5.0, 500.0, log=True),
        "lam_arena": P("float", 5.0, 500.0, log=True),
        "rounds": P("cat", choices=(2, 4, 6)),
        "knots_dist": P("cat", choices=(4, 6, 8)),
        "knots_tposs": P("cat", choices=(3, 5, 7)),
        "knots_on": P("cat", choices=(3, 5)),
        "clock": P("cat", choices=("bands", "fine")),
        **{f"inter_{t}": P("cat", choices=(False, True)) for t in GLM_INTER},
    },
    "xgb": {
        "eta": P("float", 0.02, 0.15, log=True),
        "max_depth": P("int", 3, 10),
        "min_child_weight": P("float", 5.0, 5000.0, log=True),
        "gamma": P("float", 0.0, 5.0),
        "lambda": P("float", 0.1, 100.0, log=True),
        "alpha": P("float", 1e-3, 10.0, log=True),
        "subsample": P("float", 0.5, 1.0),
        "colsample_bytree": P("float", 0.4, 1.0),
        "max_bin": P("cat", choices=(64, 128, 256)),
        "grow_policy": P("cat", choices=("depthwise", "lossguide")),
        "max_leaves": P("int", 16, 256, log=True, when=("grow_policy", "lossguide")),
    },
    "lgbm": {
        "learning_rate": P("float", 0.02, 0.15, log=True),
        "num_leaves": P("int", 7, 255, log=True),
        "min_data_in_leaf": P("int", 500, 50_000, log=True),
        "lambda_l2": P("float", 0.1, 100.0, log=True),
        "lambda_l1": P("float", 1e-3, 10.0, log=True),
        "feature_fraction": P("float", 0.4, 1.0),
        "bagging_fraction": P("float", 0.5, 1.0),
        "max_bin": P("cat", choices=(63, 127, 255)),
        "path_smooth": P("float", 0.0, 100.0),
        "extra_trees": P("cat", choices=(False, True)),
    },
    "cat": {
        "depth": P("int", 4, 10),
        "learning_rate": P("float", 0.02, 0.2, log=True),
        "l2_leaf_reg": P("float", 1.0, 100.0, log=True),
        "random_strength": P("float", 0.0, 5.0),
        "bagging_temperature": P("float", 0.0, 2.0),
        "border_count": P("cat", choices=(64, 128, 254)),
        "grow_policy": P("cat", choices=("SymmetricTree", "Depthwise")),
        "min_data_in_leaf": P("int", 100, 20_000, log=True, when=("grow_policy", "Depthwise")),
    },
    "chimera": {
        "depth": P("int", 4, 10),
        "learning_rate": P("float", 0.02, 0.2, log=True),
        "l2_leaf_reg": P("float", 0.1, 100.0, log=True),
        "min_child_weight": P("float", 1.0, 5000.0, log=True),
        "subsample": P("float", 0.5, 1.0),
        "colsample": P("float", 0.4, 1.0),
        "max_bins": P("cat", choices=(64, 128, 255)),
        "linear_leaves": P("cat", choices=(False, True)),
        "linear_lambda": P("float", 0.3, 30.0, log=True, when=("linear_leaves", True)),
    },
}
FIXED: dict = {"glm": {"standardise": True}, "lgbm": {"bagging_freq": 1}, "chimera": {"linear_leaves": False}}

# The Hessian-unit (or row-count) knobs, entered per FULL_ROWS rows.
HESSIAN_KEYS: dict = {
    "glm": ("ridge", "ridge_int", "era_lambda", "lam_shooter", "lam_arena"),
    "xgb": ("min_child_weight", "lambda", "reg_lambda", "alpha", "reg_alpha", "gamma", "min_split_loss"),
    "lgbm": ("min_data_in_leaf", "min_sum_hessian_in_leaf", "lambda_l2", "lambda_l1", "min_gain_to_split",
             "path_smooth"),
    "cat": ("l2_leaf_reg", "min_data_in_leaf"),
    "chimera": ("l2_leaf_reg", "min_child_weight", "linear_lambda"),
}
_INT_KEYS = {"min_data_in_leaf"}


def suggest(trial, kind: str, fixed: dict | None = None, only=None, toggles=()) -> dict:
    """Optuna suggestions for `kind` from SPACES, on top of FIXED and `fixed`.  `only` limits the draw to those
    names (the rest come from `fixed`: "the XGBoost plateau translated plus 10 trials over num_leaves ...").
    A '<kind>_nooff' kind draws from its booster's space.

    For the regression the result is shotmodel.fit's own keywords: `blocks` (shotmodel.BLOCKS, 'clockfine' for
    'clock' when the clock is 'fine', plus each feature block in `toggles` -- the step-1 survivors, names in
    shotfeatures.BLOCKS -- switched on), `inter` (the interaction blocks switched on), `knots` ({"dist", "tposs",
    "on"}), a half-life of inf on a side whose decay is off, no era_lambda when the era block is off, and
    standardise on.  The driver adds `block` (the rated seasons) per fit, and since the half-lives multiply their
    own closeness into the weights, `w` must then carry no closeness of its own."""
    kind = kind.removesuffix("_nooff")
    space = dict(SPACES[kind])
    if kind == "glm":
        space.update({f"feat_{t}": P("cat", choices=(False, True)) for t in toggles})
    out = dict(FIXED.get(kind, {}))
    out.update(fixed or {})
    for name, d in space.items():
        if only is not None and name not in only:
            continue
        if d.when is not None and out.get(d.when[0]) != d.when[1]:
            out.pop(name, None)
            continue
        if d.type == "float":
            out[name] = trial.suggest_float(name, d.lo, d.hi, log=d.log)
        elif d.type == "int":
            out[name] = trial.suggest_int(name, int(d.lo), int(d.hi), log=d.log)
        else:
            out[name] = trial.suggest_categorical(name, list(d.choices))
    if kind == "glm":
        out = _glm_params(out)
    return out


def _glm_params(out: dict) -> dict:
    """SPACES['glm']'s helper dimensions turned into shotmodel.fit keywords (see suggest)."""
    out = dict(out)
    blocks = list(out.pop("blocks", sm.BLOCKS))
    if out.pop("clock", "bands") == "fine":
        blocks = ["clockfine" if b == "clock" else b for b in blocks]
    inter = list(out.pop("inter", ()))
    for k in [k for k in out if k.startswith(("feat_", "inter_"))]:
        on = out.pop(k)
        name = k.split("_", 1)[1]
        if on:
            (blocks if k.startswith("feat_") else inter).append(name)
    out["blocks"] = tuple(dict.fromkeys(blocks))
    out["inter"] = tuple(dict.fromkeys(inter))
    knots = {k: out.pop(f"knots_{k}") for k in ("dist", "tposs", "on") if f"knots_{k}" in out}
    if knots:
        out["knots"] = {**(out.get("knots") or {}), **knots}
    for side in ("past", "future"):
        if not out.pop(f"decay_{side}", True):
            out[f"half_life_{side}"] = float("inf")          # shotmodel.closeness_weights: inf = no decay that side
    out.pop("era", None)                                    # era_lambda present (era on) or absent (no era block)
    return out


def scale_penalties(kind: str, params: dict, factor: float) -> dict:
    """Penalties are entered per FULL_ROWS training rows; for a sample of factor x FULL_ROWS rows (the search's
    750k: factor 0.5) every Hessian-unit or row-count knob is multiplied by `factor`, so a knob found on the search
    sample means the same in a 1.5M build.  The fit functions never scale on their own."""
    keys = HESSIAN_KEYS[kind.removesuffix("_nooff")]
    out = dict(params)
    for k in keys:
        if k in out and out[k] is not None:
            v = out[k] * factor
            out[k] = max(1, int(round(v))) if k in _INT_KEYS else float(v)
    return out


def translate_xgb(p: dict, kind: str) -> dict:
    """An XGBoost setting in another library's native names (the spec's "the XGBoost pick translated").  CatBoost
    has no column sampling on the GPU, so colsample_bytree is dropped there."""
    get = p.get
    if kind == "xgb":
        return dict(p)
    if kind == "lgbm":
        lossguide = get("grow_policy") == "lossguide"
        depth = int(get("max_depth", 6))
        out = dict(learning_rate=get("eta", 0.1), max_depth=-1 if lossguide else depth,
                   num_leaves=int(get("max_leaves", 31)) if lossguide else int(min(2 ** depth, 255)),
                   min_sum_hessian_in_leaf=get("min_child_weight", 1.0), lambda_l2=get("lambda", 1.0),
                   lambda_l1=get("alpha", 0.0), min_gain_to_split=get("gamma", 0.0),
                   bagging_fraction=get("subsample", 1.0), bagging_freq=1,
                   feature_fraction=get("colsample_bytree", 1.0), max_bin=int(get("max_bin", 256)) - 1)
        return out
    if kind == "cat":
        out = dict(learning_rate=get("eta", 0.1), depth=int(min(get("max_depth", 6), 10)),
                   l2_leaf_reg=get("lambda", 3.0), border_count=int(min(get("max_bin", 254), 254)))
        if get("subsample", 1.0) < 1.0:
            out.update(bootstrap_type="Bernoulli", subsample=get("subsample"))
        return out
    if kind == "chimera":
        return dict(learning_rate=get("eta", 0.1), depth=int(get("max_depth", 6)), l2_leaf_reg=get("lambda", 1.0),
                    min_child_weight=get("min_child_weight", 1.0), subsample=get("subsample", 1.0),
                    colsample=get("colsample_bytree", 1.0), max_bins=int(min(get("max_bin", 128), 255)),
                    linear_leaves=False)
    raise ValueError(kind)
