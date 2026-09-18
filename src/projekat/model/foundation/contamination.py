"""Provera kontaminacije predtreniranjem -- PRVOKLASAN, ZAPISAN ARTEFAKT.

Nijedan ttm_* broj se ne prijavljuje bez ovog artefakta. Poredi objavljeni
datum odsecanja podataka za predtreniranje PRIKOVANOG checkpoint-a
(config.TTM_PRETRAIN_CUTOFF, vezan za config.TTM_REVISION) sa test
periodom, i ZAPISUJE nalaz u results/ pored prognoza -- bez obzira na
ishod.

Politika kada JESTE nađena kontaminacija: prijavi preklapanje eksplicitno,
kvantifikuj pogođeni period, i ZADRŽI model u analizi, jasno obeležen kao
kontaminiran. Nikad tiho ne izbacuj model i nikad tiho ne nastavljaj kao
da je čist -- oboje bi sakrilo baš ono što ova provera postoji da pokaže.
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
    overlap_days: int
    overlap_share_of_test: float
    note: str


def check_contamination(test_start, test_end) -> ContaminationReport:
    """Preklapanje = deo test perioda koji pada PRE datuma odsecanja
    predtreniranja prikovanog checkpoint-a."""
    cutoff = pd.Timestamp(TTM_PRETRAIN_CUTOFF)
    start = pd.Timestamp(test_start)
    end = pd.Timestamp(test_end)

    overlap_start = start
    overlap_end = min(end, cutoff - pd.Timedelta(days=1))
    contaminated = overlap_end >= overlap_start

    if contaminated:
        overlap_days = int((overlap_end - overlap_start).days) + 1
        total_days = int((end - start).days) + 1
        share = overlap_days / total_days if total_days > 0 else 0.0
        note = (
            f"CONTAMINATED: {overlap_days} of {total_days} test days "
            f"({share:.1%}) fall before the pinned checkpoint's pretraining "
            f"cutoff {TTM_PRETRAIN_CUTOFF}. The model is KEPT IN the analysis "
            f"and must be reported labelled as contaminated over this period."
        )
        return ContaminationReport(
            model_id=TTM_MODEL_ID,
            revision=TTM_REVISION,
            pretrain_cutoff=TTM_PRETRAIN_CUTOFF,
            test_start=str(start.date()),
            test_end=str(end.date()),
            contaminated=True,
            overlap_start=str(overlap_start.date()),
            overlap_end=str(overlap_end.date()),
            overlap_days=overlap_days,
            overlap_share_of_test=share,
            note=note,
        )

    return ContaminationReport(
        model_id=TTM_MODEL_ID,
        revision=TTM_REVISION,
        pretrain_cutoff=TTM_PRETRAIN_CUTOFF,
        test_start=str(start.date()),
        test_end=str(end.date()),
        contaminated=False,
        overlap_start=None,
        overlap_end=None,
        overlap_days=0,
        overlap_share_of_test=0.0,
        note=(
            f"CLEAN: the whole test period starts on or after the pinned "
            f"checkpoint's pretraining cutoff {TTM_PRETRAIN_CUTOFF}."
        ),
    )


def write_contamination_report(report: ContaminationReport, *, out_dir: Path | None = None) -> Path:
    """Zapiši nalaz u results/ -- uvek, i kad je čisto i kad nije."""
    directory = out_dir or RESULTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / CONTAMINATION_FILENAME
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path


def check_and_write(test_start, test_end, *, out_dir: Path | None = None) -> ContaminationReport:
    report = check_contamination(test_start, test_end)
    write_contamination_report(report, out_dir=out_dir)
    return report
