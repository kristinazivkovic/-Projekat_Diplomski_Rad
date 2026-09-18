"""Model Confidence Set (Hansen, Lunde & Nason): iteratively eliminate the
worst model until the survivors are statistically indistinguishable.

=====================================================================
MCS MEMBERSHIP IS NOT A RANKING OF POINT LOSSES
=====================================================================
This is the misreading the module exists to prevent, and it must be
stated this way in the thesis text too.

The MCS answers "which models cannot be STATISTICALLY DISTINGUISHED from
the best one, at this sample size?", not "which models have the lowest
average loss?". Those two questions give different answers, and the
difference is not subtle:

  * A model with a HIGHER average QLIKE than some ELIMINATED model can
    still remain in the MCS. Elimination depends on the loss difference
    relative to its HAC standard error, not on the difference itself --
    a model that is slightly worse but VERY STABLE survives, while a
    model with a better average but an unstable difference series is
    dropped.
  * There is NO ordering inside the MCS. `in_mcs=True` for two models
    means they cannot be told apart from each other either -- not that
    the one with the lower average loss is "better". Sorting MCS members
    by average QLIKE and reading that order as a result defeats the
    entire point of the test.
  * MCS size measures SAMPLE POWER, not model quality. A short test
    period gives a wide MCS (nothing is distinguishable); a long one
    gives a narrow MCS. Membership counts rising or falling between
    horizons does not mean the models became more alike, only that the
    power of the test changed.

So the MCS is reported in the thesis as a binary membership flag
ALONGSIDE the point-loss table (results/tables.py), never instead of it,
and never as a derived ordering.

The result is a SET, not a ranking."""

from __future__ import annotations

import numpy as np

from projekat.evaluation.dm_test import diebold_mariano


def model_confidence_set(losses: dict[str, np.ndarray], *, alpha: float = 0.10, horizon: int = 1) -> set[str]:
    """losses: {ime_modela: niz dnevnih gubitaka, usklađen između modela}.
    horizon: prognozni horizont čiji su ovo gubici -- prosleđuje se
    diebold_mariano kao h, tako da HAC varijansa koristi max(h-1,1)
    pomeraja (lags) umesto da uvek pretpostavlja h=1, što bi
    potcenjivalo varijansu na h=5/h=22 usled preklapajućih ciljnih
    vrednosti (targets.py) i pravilo test anti-konzervativnim.
    Vraća skup imena modela u MCS-u pri poverenju 1-alpha."""
    survivors = list(losses.keys())

    while len(survivors) > 1:
        # statistika eliminacije: najgori prosečan relativni učinak u
        # odnosu na prosek preživelih modela
        avg_losses = {m: np.mean(losses[m]) for m in survivors}
        mean_loss = np.mean(list(avg_losses.values()))
        worst = max(survivors, key=lambda m: avg_losses[m] - mean_loss)

        # testiraj najgori naspram najboljeg preostalog u parovima; ako
        # nije značajno gori od svakog drugog preživelog, prekini eliminaciju
        best = min(survivors, key=lambda m: avg_losses[m])
        if worst == best:
            break
        result = diebold_mariano(losses[worst], losses[best], h=horizon)
        if result.p_value >= alpha:
            break
        survivors.remove(worst)

    return set(survivors)
