"""Euro-area and Italian macro data from the ECB Data Portal and Eurostat.

Both APIs are free and keyless, so European macro context is available without
the FRED key the US macro tool needs. The series are chosen for analysing
Italian and euro-area equities: ECB policy rates, Euribor, euro-area and Italian
HICP inflation, Italian growth, labour, industry and sentiment, and the BTP-Bund
spread that prices Italian sovereign risk.

Point-in-time: neither API serves vintages (FRED's ALFRED has no counterpart
here), so the values are the latest published revision. To keep a historical
run from seeing a figure before it was released, every observation is held back
until its period has ended and the series' usual publication lag has passed.
The report says so, and flags the series whose figures are often revised, so a
backtest reader knows which numbers may differ from what was known at the time.
"""

from __future__ import annotations

import calendar
import csv
import functools
import io
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import requests

from tradingagents.dataflows.errors import VendorUnavailableError

logger = logging.getLogger(__name__)

ECB_API_BASE = "https://data-api.ecb.europa.eu/service/data"
EUROSTAT_API_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"

REQUEST_TIMEOUT = 30
DEFAULT_LOOKBACK_DAYS = 365
MAX_ROWS = 40


@dataclass(frozen=True)
class EuroSeries:
    """One series, where it lives, and when each observation becomes public."""

    title: str
    source: str  # "ecb" or "eurostat"
    dataset: str  # ECB flow (e.g. "FM") or Eurostat dataset code
    units: str
    frequency: str
    # Days after the end of an observation's period before it is treated as
    # published. Conservative: a figure arriving a little later than its release
    # costs nothing, one arriving before it would leak the future into a backtest.
    publication_lag_days: int
    key: str = ""  # ECB series key
    filters: dict = field(default_factory=dict)  # Eurostat dimension filters
    revised: bool = False  # figures are commonly revised after first release


def _eurostat(dataset, title, units, frequency, lag, revised=False, **filters):
    return EuroSeries(title=title, source="eurostat", dataset=dataset, units=units,
                      frequency=frequency, publication_lag_days=lag,
                      filters=filters, revised=revised)


def _ecb(flow, key, title, units, frequency, lag):
    return EuroSeries(title=title, source="ecb", dataset=flow, key=key, units=units,
                      frequency=frequency, publication_lag_days=lag)


# Friendly alias -> series. Formats and filter codes were checked against the
# live APIs (Eurostat's HICP moved to prc_hicp_minr / COICOP 2018 in 2026; the
# older prc_hicp_manr stopped at 2025-12).
EURO_SERIES: dict[str, EuroSeries] = {
    # ECB policy and money markets
    "ecb_deposit_rate": _ecb(
        "FM", "D.U2.EUR.4F.KR.DFR.LEV", "ECB deposit facility rate", "%", "Daily", 1),
    "ecb_main_refi_rate": _ecb(
        "FM", "D.U2.EUR.4F.KR.MRR_FR.LEV", "ECB main refinancing operations rate", "%", "Daily", 1),
    "eur_usd": _ecb(
        "EXR", "D.USD.EUR.SP00.A", "ECB reference rate, US dollars per euro", "USD", "Daily", 1),
    "euribor_3m": _eurostat(
        "irt_st_m", "Euribor 3-month (monthly average), euro area", "%", "Monthly", 3,
        int_rt="IRT_M3", geo="EA"),
    # Inflation (HICP, annual rate of change)
    "euro_hicp": _eurostat(
        "prc_hicp_minr", "HICP inflation, euro area", "% y/y", "Monthly", 20,
        unit="RCH_A", coicop18="TOTAL", geo="EA"),
    "euro_core_hicp": _eurostat(
        "prc_hicp_minr", "Core HICP inflation (excl. energy, food, alcohol, tobacco), euro area",
        "% y/y", "Monthly", 20, unit="RCH_A", coicop18="TOT_X_NRG_FOOD", geo="EA"),
    "italy_hicp": _eurostat(
        "prc_hicp_minr", "HICP inflation, Italy", "% y/y", "Monthly", 20,
        unit="RCH_A", coicop18="TOTAL", geo="IT"),
    "italy_core_hicp": _eurostat(
        "prc_hicp_minr", "Core HICP inflation (excl. energy, food, alcohol, tobacco), Italy",
        "% y/y", "Monthly", 20, unit="RCH_A", coicop18="TOT_X_NRG_FOOD", geo="IT"),
    # Sovereign yields (EMU convergence criterion, monthly average)
    "italy_10y": _eurostat(
        "irt_lt_mcby_m", "10-year government bond yield (BTP), Italy", "%", "Monthly", 10,
        geo="IT"),
    "germany_10y": _eurostat(
        "irt_lt_mcby_m", "10-year government bond yield (Bund), Germany", "%", "Monthly", 10,
        geo="DE"),
    # Italian real economy
    "italy_gdp_growth": _eurostat(
        "namq_10_gdp", "Real GDP growth q/q (chain-linked, seasonally and calendar adjusted), Italy",
        "% q/q", "Quarterly", 60, revised=True,
        unit="CLV_PCH_PRE", s_adj="SCA", na_item="B1GQ", geo="IT"),
    "italy_unemployment": _eurostat(
        "une_rt_m", "Unemployment rate (seasonally adjusted), Italy", "% of labour force",
        "Monthly", 35, revised=True,
        s_adj="SA", age="TOTAL", sex="T", unit="PC_ACT", geo="IT"),
    "italy_industrial_production": _eurostat(
        "sts_inpr_m", "Industrial production (B-D) y/y, calendar adjusted, Italy", "% y/y",
        "Monthly", 45, revised=True,
        indic_bt="PRD", nace_r2="B-D", s_adj="CA", unit="PCH_SM", geo="IT"),
    "italy_economic_sentiment": _eurostat(
        "ei_bssi_m_r2", "Economic sentiment indicator (long-run average = 100), Italy", "index",
        "Monthly", 0, indic="BS-ESI-I", s_adj="SA", geo="IT"),
}

# Common spellings the model may use for the same series.
_ALIASES = {
    "deposit_rate": "ecb_deposit_rate",
    "ecb_rate": "ecb_deposit_rate",
    "main_refi_rate": "ecb_main_refi_rate",
    "eurusd": "eur_usd",
    "euribor": "euribor_3m",
    "hicp": "euro_hicp",
    "euro_inflation": "euro_hicp",
    "euro_core_inflation": "euro_core_hicp",
    "italy_inflation": "italy_hicp",
    "btp_10y": "italy_10y",
    "btp": "italy_10y",
    "bund_10y": "germany_10y",
    "bund": "germany_10y",
    "italy_gdp": "italy_gdp_growth",
    "italy_industrial": "italy_industrial_production",
    "italy_esi": "italy_economic_sentiment",
    "spread": "btp_bund_spread",
    "btp_bund": "btp_bund_spread",
}

# Derived series: computed from two series above rather than fetched.
SPREAD_ALIAS = "btp_bund_spread"

AVAILABLE = sorted([*EURO_SERIES, SPREAD_ALIAS])


def _canonical(indicator: str) -> str | None:
    key = indicator.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")
    key = _ALIASES.get(key, key)
    return key if key in EURO_SERIES or key == SPREAD_ALIAS else None


# ---------------------------------------------------------------- periods


def period_end(period: str) -> date:
    """The last day of an SDMX time period: 2026-08-14, 2026-08, 2026-Q2, 2026-S1, 2026."""
    p = period.strip()
    try:
        if len(p) == 10:
            return datetime.strptime(p, "%Y-%m-%d").date()
        if "-Q" in p:
            year, quarter = p.split("-Q")
            month = int(quarter) * 3
            return date(int(year), month, calendar.monthrange(int(year), month)[1])
        if "-S" in p:
            year, half = p.split("-S")
            return date(int(year), 6, 30) if half == "1" else date(int(year), 12, 31)
        if len(p) == 7:
            year, month = int(p[:4]), int(p[5:7])
            return date(year, month, calendar.monthrange(year, month)[1])
        if len(p) == 4:
            return date(int(p), 12, 31)
    except ValueError:
        pass
    raise ValueError(f"unrecognised time period {period!r}")


def published_by(period: str, lag_days: int, as_of: date) -> bool:
    """Whether an observation for ``period`` was public on ``as_of``."""
    return period_end(period) + timedelta(days=lag_days) <= as_of


# ---------------------------------------------------------------- fetching


def _get(url: str, params: dict, accept: str | None = None) -> requests.Response:
    headers = {"Accept": accept} if accept else {}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        raise VendorUnavailableError(f"ECB/Eurostat request failed: {exc}") from None
    if response.status_code == 429 or response.status_code >= 500:
        raise VendorUnavailableError(
            f"ECB/Eurostat answered HTTP {response.status_code} for {url}"
        )
    return response


@functools.lru_cache(maxsize=64)
def _fetch_ecb(flow: str, key: str, start: str, end: str) -> tuple[tuple[str, float], ...]:
    """(period, value) pairs from the ECB Data Portal's SDMX-CSV, oldest first."""
    response = _get(
        f"{ECB_API_BASE}/{flow}/{key}",
        {"startPeriod": start, "endPeriod": end, "format": "csvdata"},
    )
    # The ECB answers 404 when a valid series has no observations in the window.
    if response.status_code == 404:
        return ()
    if response.status_code >= 400:
        raise ValueError(f"ECB request for {flow}/{key} failed: HTTP {response.status_code}")
    return parse_ecb_csv(response.text)


def parse_ecb_csv(text: str) -> tuple[tuple[str, float], ...]:
    rows = csv.DictReader(io.StringIO(text))
    if rows.fieldnames is None or not {"TIME_PERIOD", "OBS_VALUE"} <= set(rows.fieldnames):
        raise VendorUnavailableError("ECB response is not SDMX-CSV with TIME_PERIOD and OBS_VALUE")
    points = []
    for row in rows:
        value = (row.get("OBS_VALUE") or "").strip()
        if value in ("", "NaN"):
            continue
        try:
            points.append((row["TIME_PERIOD"].strip(), float(value)))
        except ValueError:
            continue
    return tuple(sorted(points, key=lambda p: period_end(p[0])))


@functools.lru_cache(maxsize=64)
def _fetch_eurostat(dataset: str, filters: tuple[tuple[str, str], ...]) -> tuple[tuple[str, float], ...]:
    """(period, value) pairs for one Eurostat series, oldest first.

    The whole history of the single series is fetched (a few hundred points)
    and windowed locally, which keeps the request independent of the series'
    period format and lets one fetch serve every window in a run.
    """
    response = _get(
        f"{EUROSTAT_API_BASE}/{dataset}",
        {"format": "JSON", "lang": "EN", **dict(filters)},
    )
    if response.status_code >= 400:
        try:
            detail = response.json().get("error", {})
            message = detail.get("label") if isinstance(detail, dict) else detail
        except ValueError:
            message = response.text[:200]
        raise ValueError(f"Eurostat request for {dataset} failed: {message or response.status_code}")
    return parse_jsonstat(response.json())


def parse_jsonstat(payload: dict) -> tuple[tuple[str, float], ...]:
    """(period, value) pairs from a Eurostat JSON-stat 2.0 dataset.

    Every non-time dimension is expected to be filtered to one category; the
    flat value index is computed from the dimension sizes all the same, so a
    dimension listed after "time" cannot shift the values onto wrong periods.
    """
    ids = payload.get("id") or []
    sizes = payload.get("size") or []
    if "time" not in ids or len(ids) != len(sizes):
        raise VendorUnavailableError("Eurostat response is not a JSON-stat dataset with a time dimension")
    extra = [d for d, n in zip(ids, sizes, strict=True) if d != "time" and n != 1]
    if extra:
        raise ValueError(f"Eurostat series is ambiguous: dimensions {extra} have several categories")
    time_index = payload["dimension"]["time"]["category"]["index"]
    position = ids.index("time")
    stride = 1
    for n in sizes[position + 1:]:
        stride *= n
    values = payload.get("value") or {}
    if isinstance(values, list):
        values = {str(i): v for i, v in enumerate(values)}
    points = []
    for period, i in time_index.items():
        value = values.get(str(i * stride))
        if value is not None:
            points.append((period, float(value)))
    return tuple(sorted(points, key=lambda p: period_end(p[0])))


def fetch_series(series: EuroSeries, start: date, end: date) -> list[tuple[str, float]]:
    """The series' observations whose period ends inside [start, end]."""
    if series.source == "ecb":
        points = _fetch_ecb(series.dataset, series.key, start.isoformat(), end.isoformat())
    else:
        points = _fetch_eurostat(series.dataset, tuple(sorted(series.filters.items())))
    return [(p, v) for p, v in points if start <= period_end(p) <= end]


# ---------------------------------------------------------------- report


def _point_in_time_note(lag_days: int, revised: bool) -> str:
    note = (
        f"- Point-in-time: an observation is shown only once its period has ended and "
        f"{lag_days} days have passed (the usual publication lag); values are the latest "
        f"revision, since ECB and Eurostat serve no vintages."
    )
    if revised:
        note += " This series is often revised: a historical figure may differ from the first release."
    return note + "\n"


def _render(title: str, ident: str, units: str, frequency: str, start: date, as_of: date,
            points: list[tuple[str, float]], note: str) -> str:
    header = (
        f"## {title} ({ident})\n"
        f"- Units: {units}\n"
        f"- Frequency: {frequency}\n"
        f"- Window: {start.isoformat()} to {as_of.isoformat()}\n"
        f"{note}"
    )
    if not points:
        return header + (
            "\nNo observation published in this window. The series may be released less "
            "often than the window covers; try a longer look_back_days.\n"
        )
    first_period, first = points[0]
    last_period, last = points[-1]
    delta = last - first
    summary = (
        f"\n**Latest:** {last:g} ({last_period}) | **Change from {first_period} to "
        f"{last_period}:** {delta:+.2f}, from {first:g}\n"
    )
    shown, more = points, ""
    if len(points) > MAX_ROWS:
        shown = points[-MAX_ROWS:]
        more = f"\n_(showing the most recent {MAX_ROWS} of {len(points)} observations)_\n"
    table = "\n| Period | Value |\n| --- | --- |\n" + "\n".join(
        f"| {p} | {v:g} |" for p, v in shown
    ) + "\n"
    return header + summary + more + table


def _spread_report(start: date, as_of: date) -> str:
    italy, germany = EURO_SERIES["italy_10y"], EURO_SERIES["germany_10y"]
    lag = max(italy.publication_lag_days, germany.publication_lag_days)
    it_points = dict(fetch_series(italy, start, as_of))
    de_points = dict(fetch_series(germany, start, as_of))
    points = [
        (p, round((it_points[p] - de_points[p]) * 100, 1))
        for p in sorted(set(it_points) & set(de_points), key=period_end)
        if published_by(p, lag, as_of)
    ]
    return _render(
        "BTP-Bund 10-year spread (Italy minus Germany, monthly average yields)",
        SPREAD_ALIAS, "basis points", "Monthly", start, as_of, points,
        _point_in_time_note(lag, revised=False),
    )


def get_euro_macro_data(indicator: str, as_of_date: str, look_back_days: int | None = None) -> str:
    """A euro-area or Italian macro series as a markdown report, as of ``as_of_date``.

    Args:
        indicator: an alias from ``AVAILABLE`` (e.g. "italy_hicp", "btp_bund_spread").
        as_of_date: yyyy-mm-dd; nothing published after it is shown.
        look_back_days: trailing window; ``None`` uses one year.
    """
    look_back_days = DEFAULT_LOOKBACK_DAYS if look_back_days is None else max(1, int(look_back_days))
    as_of = datetime.strptime(as_of_date, "%Y-%m-%d").date()
    start = as_of - timedelta(days=look_back_days)

    alias = _canonical(indicator)
    if alias is None:
        return (
            f"ECB/Eurostat: '{indicator}' is not a known European macro series. "
            f"Use one of: {', '.join(AVAILABLE)}."
        )
    if alias == SPREAD_ALIAS:
        return _spread_report(start, as_of)

    series = EURO_SERIES[alias]
    points = [
        (p, v) for p, v in fetch_series(series, start, as_of)
        if published_by(p, series.publication_lag_days, as_of)
    ]
    source = "ECB" if series.source == "ecb" else "Eurostat"
    ident = f"{source} {series.dataset}" + (f" {series.key}" if series.key else "")
    return _render(
        series.title, f"{alias}; {ident}", series.units, series.frequency, start, as_of, points,
        _point_in_time_note(series.publication_lag_days, series.revised),
    )
