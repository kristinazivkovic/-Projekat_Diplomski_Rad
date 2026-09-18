"""Masovno preuzimanje 1-minutnih svećica iz LSE vault-a, sa lokalnim kešom i
manifestom porekla podataka za svaki preuzeti fajl.

Koristi se ``client.history()`` (masovni Parquet izvoz), nikada
``client.candles()``: potonji je ograničen na 5000 redova po pozivu čak i
preko višegodišnjih perioda (empirijski provereno), što ga čini
neupotrebljivim za 16 godina 1-minutnih podataka.
"""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from projekat.config import RAW_DIR

load_dotenv()

# Windows skladište sertifikata na ovoj mašini sadrži sertifikat koji
# ssl.enum_certificates() označava kao neispravan. To nikad ne utiče na
# stvarno TLS rukovanje sa vault-om (dijagnostikovano u Projekat/bad_cert.py)
# -- upozorenje se ovde isključuje da ne zatrpava logove preuzimanja.
warnings.filterwarnings(
    "ignore", message="Bad certificate in Windows certificate store", category=UserWarning
)


@dataclass(frozen=True)
class Manifest:
    symbol: str
    timeframe: str
    start: str
    end: str
    fetched_at: str
    row_count: int
    sha256: str
    path: str


def _client():
    from lse import LSE

    api_key = os.getenv("LSE_KEY")
    if not api_key:
        raise RuntimeError("LSE_KEY is not set (expected in .env)")
    return LSE(api_key=api_key)


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(parquet_path.suffix + ".manifest.json")


def fetch_1m(symbol: str, start: str, end: str, *, force: bool = False) -> Path:
    """Preuzmi 1-minutne svećice za ``symbol`` u opsegu [start, end) preko
    masovnog izvoza iz vault-a, čuvajući Parquet fajl i manifest porekla
    podataka u RAW_DIR. Idempotentno: preskače preuzimanje ako manifest već
    odgovara traženom opsegu, osim ako je ``force=True``.
    """
    dest_dir = RAW_DIR / "1m"
    dest_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = dest_dir / f"{symbol}_1m_{start}_{end}.parquet"
    manifest_path = _manifest_path(parquet_path)

    if not force and parquet_path.exists() and manifest_path.exists():
        return parquet_path

    client = _client()
    saved_path = client.history(
        symbol,
        timeframe="1m",
        start=start,
        end=end,
        dest=str(dest_dir),
        dataframe=False,
    )
    saved_path = Path(saved_path)
    if saved_path != parquet_path:
        saved_path.replace(parquet_path)

    import pyarrow.parquet as pq

    row_count = pq.read_metadata(parquet_path).num_rows

    manifest = Manifest(
        symbol=symbol,
        timeframe="1m",
        start=start,
        end=end,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        row_count=row_count,
        sha256=_sha256_of(parquet_path),
        path=str(parquet_path),
    )
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2))
    return parquet_path


def load_manifest(symbol: str, start: str, end: str) -> Manifest | None:
    dest_dir = RAW_DIR / "1m"
    manifest_path = _manifest_path(dest_dir / f"{symbol}_1m_{start}_{end}.parquet")
    if not manifest_path.exists():
        return None
    return Manifest(**json.loads(manifest_path.read_text()))


# ── izvozni limit reda: tiha skraćivanja ─────────────────────────────────
# Vault-ov /export vraća NAJVIŠE ~2.5M redova po poslu i NE prijavljuje
# grešku kad odseče -- fajl je validan parquet, manifest je tačan, a
# podaci prosto prestaju usred perioda. Izmereno: AAPL 2010-2025 stigao
# je kao tačno 2.500.000 redova koji se završavaju 2023-07-06, dakle bez
# 2.5 godine test perioda; INTC, sa 2.398.142 reda, prošao je ceo.
#
# Zato se duži rasponi MORAJU preuzimati u komadima. 8-godišnji komad je
# ~1.25M redova -- udobno ispod limita -- i traži samo 2 izvozna posla po
# simbolu, što je bitno jer je izvozna kvota (429) ograničena po BROJU
# poslova, pa sitniji komadi troše kvotu bez potrebe.
EXPORT_ROW_CAP = 2_500_000
SAFE_CHUNK_YEARS = 8


def is_truncated(path: Path, *, expected_end: str) -> bool:
    """Da li je izvoz tiho odsečen: broj redova je na limitu I poslednja
    svećica je znatno pre traženog kraja."""
    import pandas as pd
    import pyarrow.parquet as pq

    if pq.read_metadata(path).num_rows < EXPORT_ROW_CAP:
        return False
    df = pq.read_table(path).to_pandas()
    col = "ts" if "ts" in df.columns else "timestamp"
    last = pd.to_datetime(df[col], utc=True).max()
    want = pd.Timestamp(expected_end, tz="UTC")
    return last < want - pd.Timedelta("30D")


def _chunk_ranges(start: str, end: str, years: int = SAFE_CHUNK_YEARS) -> list[tuple[str, str]]:
    """Podeli [start, end) na komade od `years` godina."""
    import pandas as pd

    out = []
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    while lo < hi:
        nxt = min(lo + pd.DateOffset(years=years), hi)
        out.append((lo.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        lo = nxt
    return out


def fetch_1m_chunked(symbol: str, start: str, end: str, *, force: bool = False,
                     years: int = SAFE_CHUNK_YEARS) -> list[Path]:
    """Preuzmi [start, end) u komadima koji sigurno staju ispod izvoznog
    limita, i vrati putanje svih komada.

    Svaki komad je zaseban keš unos (sopstveni manifest), pa se prekinuto
    preuzimanje nastavlja po komadu, a ne od početka simbola."""
    return [
        fetch_1m(symbol, chunk_start, chunk_end, force=force)
        for chunk_start, chunk_end in _chunk_ranges(start, end, years)
    ]


def load_1m_range(symbol: str, start: str, end: str) -> "pd.DataFrame":
    """Učitaj sve komade za [start, end) kao JEDAN okvir, sortiran po
    vremenu i bez duplikata na granicama komada.

    Ovo je tačka na kojoj ostatak pipeline-a ne mora da zna da li je
    simbol preuzet u jednom ili u više komada."""
    import pandas as pd
    import pyarrow.parquet as pq

    frames = []
    for chunk_start, chunk_end in _chunk_ranges(start, end):
        path = RAW_DIR / "1m" / f"{symbol}_1m_{chunk_start}_{chunk_end}.parquet"
        if path.exists():
            frames.append(pq.read_table(path).to_pandas())

    # Rezervno: jedan fajl za ceo raspon (stariji keš, pre deljenja na
    # komade). Prihvata se SAMO ako nije tiho odsečen na izvoznom limitu --
    # odsečen fajl je validan parquet sa tačnim manifestom, pa bi inače
    # nečujno izbacio simbol iz kasnih prozora.
    whole = RAW_DIR / "1m" / f"{symbol}_1m_{start}_{end}.parquet"
    if not frames and whole.exists():
        if is_truncated(whole, expected_end=end):
            raise ValueError(
                f"{whole.name} je tiho odsečen na izvoznom limitu vault-a "
                f"(~{EXPORT_ROW_CAP:,} redova) i ne pokriva {end}. Preuzmi "
                f"{symbol} ponovo preko scripts/fetch_data.py (deli na komade), "
                f"pa obriši stari fajl."
            )
        frames.append(pq.read_table(whole).to_pandas())

    if not frames:
        raise FileNotFoundError(
            f"nema keširanih 1m svećica za {symbol} u {start}..{end}; "
            f"pokreni scripts/fetch_data.py"
        )

    df = pd.concat(frames, ignore_index=True)
    col = "ts" if "ts" in df.columns else "timestamp"
    df[col] = pd.to_datetime(df[col], utc=True)
    return df.drop_duplicates(subset=[col]).sort_values(col).reset_index(drop=True)
