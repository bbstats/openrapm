"""Toy check: Optuna's GP sampler with two trials in flight and a constraint, as check 155's tune runs it."""
from collections import deque

import optuna
from optuna.trial import TrialState

optuna.logging.set_verbosity(optuna.logging.WARNING)
sampler = optuna.samplers.GPSampler(seed=0, n_startup_trials=4, deterministic_objective=True,
                                    constraints_func=lambda frozen: frozen.user_attrs.get("constraint", [1.0]))
study = optuna.create_study(direction="minimize", sampler=sampler)
study.enqueue_trial({"x": 0.5, "y": 0.5})
flight = deque()
for _ in range(2):
    trial = study.ask()
    flight.append((trial, trial.suggest_float("x", 0, 1), trial.suggest_float("y", 0, 1)))
done = 0
while flight:
    trial, x, y = flight.popleft()
    if done + len(flight) + 1 < 16:
        nxt = study.ask()
        flight.append((nxt, nxt.suggest_float("x", 0, 1), nxt.suggest_float("y", 0, 1)))
    trial.set_user_attr("constraint", [x - 0.8])
    study.tell(trial, (x - 0.3) ** 2 + (y - 0.7) ** 2)
    done += 1
feasible = [t for t in study.get_trials(states=(TrialState.COMPLETE,)) if t.user_attrs["constraint"][0] <= 0]
best = min(feasible, key=lambda t: t.value)
print(len(study.trials), best.number, round(best.value, 4), best.params)
