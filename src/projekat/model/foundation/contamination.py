"""Provera kontaminacije predtreniranjem -- PRVOKLASAN, ZAPISAN ARTEFAKT.

Nijedan ttm_* broj se ne prijavljuje bez ovog artefakta. Poredi objavljeni
datum odsecanja podataka za predtreniranje PRIKOVANOG checkpoint-a
(config.TTM_PRETRAIN_CUTOFF, vezan za config.TTM_REVISION) sa periodom
evaluacije, i ZAPISUJE nalaz -- bez obzira na ishod.

Dani se broje kao DANI TRGOVANJA (XNYS sesije), ne kalendarski dani: to je
jedinica u kojoj postoje prognoze i gubici, pa je udeo kontaminiranih dana
direktno udeo kontaminiranih prognoza.

Politika kada JESTE nađena kontaminacija: prijavi preklapanje eksplicitno,
kvantifikuj pogođeni period, i ZADRŽI model u analizi, jasno obeležen kao
kontaminiran. Nikad tiho ne izbacuj model i nikad tiho ne nastavljaj kao
da je čist.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from projekat.config import RESULTS_DIR, TTM_MODEL_ID, TTM_PRETRAIN_CUTOFF, TTM_REVISION

CONTAMINATION_FILENAME = "ttm_contamination_check.json"


@dataclass(frozen=True)
class ContaminationReport:
    model_id: str
    revision: str
    pretrain_cutoff: str
    test_start: str
    test_end: str
    contaminated: bool
    overlap_start: str | None
    overlap_end: str | None
    overlap_days: int              # dani TRGOVANJA pre odsecanja
    test_days: int                 # dani TRGOVANJA u periodu evaluacije
    overlap_share_of_test: float
    day_unit: str
    note: str


def _sessions(start, end) -> pd.DatetimeIndex:
    from projekat.data.calendar import sessions_between

    return pd.DatetimeIndex(sessions_between(str(pd.Timestamp(start).date()), str(pd.Timestamp(end).date())))


def check_contamination(test_start, test_end) -> ContaminationReport:
    """Preklapanje = dani trgovanja perioda evaluacije pre datuma odsecanja
    predtreniranja prikovanog checkpoint-a."""
    cutoff = pd.Timestamp(TTM_PRETRAIN_CUTOFF)
    start, end = pd.Timestamp(test_start), pd.Timestamp(test_end)
    # Prazan ili obrnut period evaluacije NIJE "čisto" -- to je greška
    # pozivaoca. Ranije je end < start tiho davalo test_days = 0, nula
    # preklapanja i nalaz "CLEAN", tj. artefakt koji tvrdi suprotno od
    # stvarnog nalaza (81.9% kontaminirano) i koji se može uzeti kao rezultat.
    if end < start:
        raise ValueError(
            f"period evaluacije je obrnut: test_end {end.date()} je pre test_start {start.date()}"
        )
    sessions = _sessions(start, end)
    sessions = sessions.tz_localize(None) if sessions.tz is not None else sessions
    if len(sessions) == 0:
        raise ValueError(
            f"period evaluacije {start.date()}..{end.date()} ne sadrži nijednu XNYS sesiju; "
            "provera kontaminacije nema nad čim da se izvede"
        )
    overlap = sessions[sessions < cutoff]
    test_days = len(sessions)
    share = len(overlap) / test_days if test_days else 0.0
    base = dict(model_id=TTM_MODEL_ID, revision=TTM_REVISION, pretrain_cutoff=TTM_PRETRAIN_CUTOFF,
                test_start=str(start.date()), test_end=str(end.date()), test_days=test_days,
                day_unit="XNYS trading sessions")

    if len(overlap):
        return ContaminationReport(
            **base,
            contaminated=True,
            overlap_start=str(overlap[0].date()),
            overlap_end=str(overlap[-1].date()),
            overlap_days=len(overlap),
            overlap_share_of_test=share,
            note=(
                f"CONTAMINATED: {len(overlap)} of {test_days} test trading days ({share:.1%}) fall "
                f"before the pinned checkpoint's pretraining cutoff {TTM_PRETRAIN_CUTOFF}. The model "
                f"is KEPT IN the analysis and must be reported labelled as contaminated over this period."
            ),
        )
    return ContaminationReport(
        **base,
        contaminated=False,
        overlap_start=None,
        overlap_end=None,
        overlap_days=0,
        overlap_share_of_test=0.0,
        note=f"CLEAN: the whole test period starts on or after the pinned checkpoint's pretraining cutoff {TTM_PRETRAIN_CUTOFF}.",
    )


def write_contamination_report(report: ContaminationReport, *, out_dir: Path | None = None) -> Path:
    """Zapiši nalaz -- uvek, i kad je čisto i kad nije."""
    directory = out_dir or RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CONTAMINATION_FILENAME
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path


def check_and_write(test_start, test_end, *, out_dir: Path | None = None) -> ContaminationReport:
    report = check_contamination(test_start, test_end)
    write_contamination_report(report, out_dir=out_dir)
    return report
