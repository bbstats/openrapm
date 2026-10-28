"""The box-score prior answered by a pretrained tabular network instead of fitted trees (the owner, 2026-10-28: "kumo
medium or small - could we use that for prior(s)? ... test it").

NVIDIA's Kumo Tabular (https://huggingface.co/blog/nvidia/kumo-tabular, the `structured-data-models` library, imported
as `sdm`) is a transformer pretrained on millions of artificial tables.  It is never fitted to this project's rows.  It
is shown them at answer time: the training rows with their labels and the rows to answer, all in one pass, and it
answers each row from the training rows it resembles.  That is called in-context learning; "context" below is the
set of training rows the network is shown.

Words used below
    training rows   box_prior.training_rows' rows, the same ones a LightGBM booster is fitted on: inputs, label,
                    row weight
    member          one of the network's answers that are averaged into a prior; each member is shown its own draw
                    of the training rows and its own random reordering and rescaling of the inputs (the library's
                    default preparation)
    draw            a member's training rows: `context_rows` of them drawn without replacement, each with chance
                    rising with its row weight, all of them when there are no more than that
    answer          the mean of the network's 999 quantiles of the label for a row, so the prior is the mean of
                    its predicted distribution, as a LightGBM booster's least-squares answer is a mean

The network cannot take row weights, so the weights decide which rows each member is shown.  A row is drawn by the
Efraimidis-Spirakis rule: a random number u between 0 and 1 for each row, and the `context_rows` rows with the largest
u ^ (1 / weight) are kept.  A row of weight 4 is kept before a row of weight 1 whenever u_4 ^ (1/4) > u_1, and so
on.  Example: about 69,000 training rows for the offense, the heaviest 20,000 carrying 84% of the weight; a member
shown 20,000 rows sees nearly every heavy one and few of the light ones, the bench players whose labels rest on a
few hundred possessions.

An `InContextPrior` stands where a LightGBM booster stands in box_prior.py, one per player fold, with the same
`predict`, so the folds keep every player's own label away from his prior exactly as they do for the trees.  Each
answer costs a pass of the network over the member's training rows, the rows to answer riding along, so answers are
kept (`answers`) and `prime` answers many rows in one pass: `openrapm folds` asks the same priors about 13 rebuilt
seasons, about 6,000 rows a side.  The rows to answer go through the network in batches of exactly `query_rows` rows,
the last one filled up with copies of its first row, so a row's answer never depends on which rows were asked with it.

Settings (params/booster_params_kumo_*.json, read by data.config.booster_params): `size` small (28.5 million
parameters) or medium (62.5 million), `members`, `context_rows`, `query_rows`, `draw`, which picks another set of
random draws for the noise control, and `draw_by`: "row_weight" as above, or "equal", every training row the same
chance whatever its weight, so the bench players' rows are shown as often as there are of them (experiment 118's
second look: drawn by weight, the prior got the bench wrong).

On the RTX 3070 (scratch/2026-10-28_kumo/bench.py): one member on 20,000 training rows, 600 rows to answer, takes
0.7 s small and 1.7 s medium, at most 0.8 GB of GPU memory.

The weights download from the Hugging Face hub on first use (nvidia/Kumo-Tabular, revision v1.0.1) into the hub's
cache, and are read from there afterwards; this module itself opens no file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["LEARNER", "SIZES", "DRAW_BY", "InContextPrior", "context_draws", "network"]

LEARNER = "kumo_tabular"         # the learner a params file names for this prior
SIZES = ("small", "medium")
DRAW_BY = ("row_weight", "equal")    # what decides a training row's chance to be shown (`draw_by`)
_NETWORKS: dict = {}             # size -> the network, loaded once per process


def network(size: str):
    """The pretrained regression network of `size`, on the GPU, loaded once per process."""
    if size not in SIZES:
        raise ValueError(f"the Kumo network's size is one of {SIZES}, not {size!r}")
    if size not in _NETWORKS:
        import sdm
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("the Kumo prior runs on a CUDA GPU; this process sees none")
        _NETWORKS[size] = sdm.models.KumoTabular(task="regression", size=size, device="cuda")
    return _NETWORKS[size]


def context_draws(weights: np.ndarray, members: int, context_rows: int, draw: int, seed: int) -> np.ndarray:
    """[members, rows shown] positions of the training rows each member is shown, sorted: every row when there are no
    more than `context_rows`, else `context_rows` of them by the Efraimidis-Spirakis rule (the module docstring), each
    member from its own random numbers, seeded by (`draw`, `seed`, member).

    Example: weights 1, 1, 8 and two rows shown: the third row is kept by nearly every member.
    """
    weights = np.asarray(weights, dtype=float)
    if (weights <= 0).any():
        raise ValueError("every training row needs a positive weight to be drawn")
    if len(weights) <= context_rows:
        return np.tile(np.arange(len(weights)), (members, 1))
    out = []
    for member in range(members):
        u = np.random.default_rng([int(draw), int(seed), member, 7039]).random(len(weights))
        keys = np.log(np.maximum(u, np.finfo(float).tiny)) / weights        # log of u ^ (1 / weight)
        out.append(np.sort(np.argpartition(-keys, context_rows)[:context_rows]))
    return np.stack(out)


def _network_answers(size: str, context_x: np.ndarray, context_y: np.ndarray, rows: np.ndarray,
                     generator_seed: int) -> np.ndarray:
    """The network's answer for each of `rows` [rows, inputs], each member shown its own training rows (context_x
    [members, shown, inputs], context_y [members, shown]): the mean of its 999 quantiles, the members' quantiles
    averaged by the library first (a trimmed mean, the 20% most extreme members' values left out)."""
    import torch
    model = network(size)
    cuda = torch.device("cuda")
    members = context_x.shape[0]
    x = torch.from_numpy(np.ascontiguousarray(context_x, dtype=np.float32)).to(cuda)
    y = torch.from_numpy(np.ascontiguousarray(context_y, dtype=np.float32)[..., None]).to(cuda)
    q = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float32)).to(cuda)
    with torch.amp.autocast("cuda", torch.float16), torch.inference_mode():
        out = model(x_context=x, y_context=y, x_query=q.expand(members, *q.shape), num_estimators=None,
                    generator=torch.Generator(cuda).manual_seed(int(generator_seed)))
    return out.numerical.float().mean(dim=-1).cpu().numpy().astype(float)


@dataclass
class InContextPrior:
    """One booster's worth of the box-score prior: the training rows it answers from and how it answers."""
    inputs: np.ndarray              # [training rows, inputs], float32
    label: np.ndarray               # [training rows]
    weights: np.ndarray             # [training rows]: the row weights, times the era kernel when there is one
    size: str = "small"
    members: int = 8
    context_rows: int = 20000
    query_rows: int = 8192
    draw: int = 0                   # the settings' draw: another number, other random draws
    seed: int = 0                   # the bag's seed (box_prior.fit_prior), 0 without bags
    answers: dict = field(default_factory=dict, repr=False)    # row bytes -> answer, for rows already asked

    @classmethod
    def from_rows(cls, params: dict, inputs, label, weights, kernel=None, seed: int = 0) -> "InContextPrior":
        """The prior on box_prior's training rows with `params` (a params file's side, data.config.booster_params):
        the row weights rescaled to mean 1 and times the era kernel, as box_prior.fit_lightgbm weighs them; every
        weight 1 with `draw_by` "equal"."""
        if params.get("draw_by", "row_weight") not in DRAW_BY:
            raise ValueError(f"draw_by is one of {DRAW_BY}, not {params['draw_by']!r}")
        weights = np.asarray(weights, dtype=float)
        if params.get("draw_by", "row_weight") == "equal":
            weights = np.ones_like(weights)
        weights = weights / weights.mean()
        if kernel is not None:
            weights = weights * np.asarray(kernel, dtype=float)
        return cls(inputs=np.asarray(inputs, dtype=np.float32), label=np.asarray(label, dtype=np.float32),
                   weights=weights, size=str(params.get("size", "small")), members=int(params.get("members", 8)),
                   context_rows=int(params.get("context_rows", 20000)),
                   query_rows=int(params.get("query_rows", 8192)), draw=int(params.get("draw", 0)), seed=int(seed))

    def __getstate__(self) -> dict:
        # the kept answers belong to this process's rows; a saved or sent prior starts without them
        return {**self.__dict__, "answers": {}}

    def predict(self, inputs) -> np.ndarray:
        """The answer for each row of `inputs` [rows, inputs], each row not asked before answered now (`prime`)."""
        matrix = np.asarray(inputs, dtype=float)
        self.prime(matrix)
        return np.array([self.answers[row.tobytes()] for row in matrix], dtype=float)

    def prime(self, inputs) -> None:
        """Answer every row of `inputs` not asked before, in batches of exactly `query_rows` rows, and keep the
        answers."""
        matrix = np.asarray(inputs, dtype=float)
        keys = [row.tobytes() for row in matrix]
        new = {}
        for key, row in zip(keys, matrix):
            if key not in self.answers and key not in new:
                new[key] = row
        if not new:
            return
        rows = np.stack(list(new.values()))
        shown = context_draws(self.weights, self.members, self.context_rows, self.draw, self.seed)
        context_x, context_y = self.inputs[shown], self.label[shown]
        answers = []
        for start in range(0, len(rows), self.query_rows):
            batch = rows[start:start + self.query_rows]
            filled = np.concatenate([batch, np.repeat(batch[:1], self.query_rows - len(batch), axis=0)])
            answers.append(_network_answers(self.size, context_x, context_y, filled,
                                            1_000_003 * self.draw + self.seed)[:len(batch)])
        self.answers.update(zip(new, np.concatenate(answers)))
