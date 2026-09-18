"""Obrada korporativnih akcija (splitova i dividendi).

Splitovi: preuzimaju se i koriste kao striktna provera (hard gate). Provereno
je da vault već isporučuje cene prilagođene splitovima (AAPL 7:1 u 2014,
NVDA 10:1 u 2024, AMZN 20:1 u 2022 -- svi pokazuju uobičajene log-prinose
<5% preko datuma splita; neprilagođen split 20:1 bi pokazao otprilike -3.0).
Ovde se ne primenjuje nikakvo prilagođavanje; umesto toga, pipeline proverava
(assert) da ovo svojstvo dobavljača i dalje važi, jer bi se moglo tiho
promeniti.

Dividende: preuzimaju se i arhiviraju samo radi evidencije porekla podataka,
i namerno se nikad ne koriste ni u jednoj transformaciji. RV (realizovana
varijansa) se gradi isključivo iz intradnevnih prinosa unutar trgovinske
sesije; pad cene na dan bez prava na dividendu (ex-dividend) dešava se na
otvaranju, unutar noćnog jaza (overnight gap) koji je već svuda isključen u
ovom pipeline-u. Provereno: AAPL-ov ex-dividendni dan 2015-11-05 je pokazao
log-prinos od -0.0069, što nije upadljivo u odnosu na susedne dane od
+0.0115 i -0.0053.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from projekat.config import RAW_DIR


@dataclass(frozen=True)
class SplitEvent:
    symbol: str
    effective_date: str
    split_from: int
    split_to: int

    @property
    def log_ratio(self) -> float:
        return math.log(self.split_to / self.split_from)


class UnadjustedDataError(RuntimeError):
    """Baca se kada veličina prinosa na dan splita odgovara odnosu splita,
    što ukazuje da je dobavljač počeo da isporučuje neprilagođene cene."""


def fetch_corporate_actions(symbol: str, start: str, end: str) -> tuple[list[SplitEvent], list[dict]]:
    """Preuzmi splitove (koriste se) i dividende (samo arhiviraju) za jedan simbol."""
    from lse import LSE
    import os
    from dotenv import load_dotenv

    load_dotenv()
    client = LSE(api_key=os.getenv("LSE_KEY"))

    raw_splits = client.splits(symbol, start=start, end=end)
    splits = [
        SplitEvent(
            symbol=symbol,
            effective_date=s["effective_date"][:10],
            split_from=int(s["split_from"]),
            split_to=int(s["split_to"]),
        )
        for s in raw_splits
    ]

    dividends = client.dividends(symbol, start=start, end=end)  # samo se arhiviraju, nikad ne koriste

    out_dir = RAW_DIR / "corporate"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    (out_dir / f"{symbol}_splits.json").write_text(json.dumps(raw_splits, indent=2))
    (out_dir / f"{symbol}_dividends.json").write_text(json.dumps(dividends, indent=2))

    return splits, dividends


def verify_split_adjustment(daily_close: dict[str, float], splits: list[SplitEvent], *, tolerance: float = 0.15) -> None:
    """Striktna provera: za svaki split, uporedi stvarni log-prinos preko
    datuma splita sa log(odnos_splita). Poklapanje znači da su podaci postali
    neprilagođeni (npr. dobavljač je promenio politiku) -- treba glasno
    prijaviti grešku, a ne tiho računati statistiku skokova na šumu iz
    korporativnog računovodstva.

    Prinos uobičajene veličine na dan splita je u redu (splitovi se mogu
    legitimno poklopiti sa skokovima usled objave zarade -- vidi
    check_jump_on_splits za deo koji uzima u obzir veličinu i nije fatalan).
    """
    dates = sorted(daily_close)
    for split in splits:
        if split.effective_date not in dates:
            continue
        idx = dates.index(split.effective_date)
        if idx == 0:
            continue
        prev_date = dates[idx - 1]
        prev_close = daily_close[prev_date]
        this_close = daily_close[split.effective_date]
        if prev_close <= 0 or this_close <= 0:
            continue
        actual_logret = math.log(this_close / prev_close)
        expected_logret = -split.log_ratio  # cena pada za odnos splita ako podaci nisu prilagođeni
        if abs(actual_logret - expected_logret) < tolerance:
            raise UnadjustedDataError(
                f"{split.symbol}: log return {actual_logret:.3f} on {split.effective_date} "
                f"matches unadjusted split ratio {split.split_from}:{split.split_to} "
                f"(expected ~{expected_logret:.3f}). Data appears to be unadjusted."
            )


def check_jump_on_splits(jump_days: set[str], splits: list[SplitEvent]) -> list[str]:
    """Nefatalna dijagnostika: datumi splitova na kojima je detektovan skok
    uobičajene veličine (verify_split_adjustment je već isključila slučaj
    neprilagođenih podataka). Vraća se radi ručne provere, ne kao greška
    pipeline-a -- splitovi se često poklapaju sa objavama zarade."""
    return [s.effective_date for s in splits if s.effective_date in jump_days]
