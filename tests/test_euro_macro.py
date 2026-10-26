"""The ECB/Eurostat vendor, the European Macro Analyst's wiring, and Italian tickers.

API access is mocked with payloads in the shapes the live APIs return (checked
against Eurostat's JSON-stat and the ECB's SDMX-CSV), so the unit tests run
offline. The ``integration`` tests at the end call the live APIs and run only
with ``pytest -m integration``.
"""

from datetime import date
from unittest import mock

import pytest
import requests

from tradingagents.agents.context import euro_macro_line
from tradingagents.dataflows import router
from tradingagents.dataflows.errors import VendorUnavailableError
from tradingagents.dataflows.symbols import apply_default_exchange
from tradingagents.dataflows.vendors import euro_macro as em
from tradingagents.graph.analyst_execution import ANALYST_NODE_SPECS


def _jsonstat(points: dict[str, float], ids=("freq", "unit", "geo", "time")) -> dict:
    """A Eurostat JSON-stat 2.0 payload with every non-time dimension filtered to one."""
    return {
        "id": list(ids),
        "size": [1] * (len(ids) - 1) + [len(points)],
        "dimension": {"time": {"category": {"index": {p: i for i, p in enumerate(points)}}}},
        "value": {str(i): v for i, v in enumerate(points.values())},
    }


_ECB_CSV = (
    "KEY,FREQ,REF_AREA,CURRENCY,PROVIDER_FM,INSTRUMENT_FM,PROVIDER_FM_ID,DATA_TYPE_FM,"
    "TIME_PERIOD,OBS_VALUE,OBS_STATUS,TITLE\n"
    "FM.D.U2.EUR.4F.KR.DFR.LEV,D,U2,EUR,4F,KR,DFR,LEV,2026-09-10,2.25,A,Deposit facility\n"
    "FM.D.U2.EUR.4F.KR.DFR.LEV,D,U2,EUR,4F,KR,DFR,LEV,2026-09-11,2.00,A,Deposit facility\n"
    "FM.D.U2.EUR.4F.KR.DFR.LEV,D,U2,EUR,4F,KR,DFR,LEV,2026-09-12,NaN,M,Deposit facility\n"
)


@pytest.fixture(autouse=True)
def _fresh_cache():
    em._fetch_ecb.cache_clear()
    em._fetch_eurostat.cache_clear()
    yield
    em._fetch_ecb.cache_clear()
    em._fetch_eurostat.cache_clear()


def _response(status=200, json_body=None, text=""):
    response = mock.Mock(status_code=status, text=text)
    response.json.return_value = json_body
    return response


# ---------------------------------------------------------------- parsing


@pytest.mark.unit
class TestParsing:
    def test_period_ends(self):
        assert em.period_end("2026-08-14") == date(2026, 8, 14)
        assert em.period_end("2026-02") == date(2026, 2, 28)
        assert em.period_end("2026-Q2") == date(2026, 6, 30)
        assert em.period_end("2026-S1") == date(2026, 6, 30)
        assert em.period_end("2026") == date(2026, 12, 31)
        with pytest.raises(ValueError):
            em.period_end("August")

    def test_jsonstat_reads_values_in_period_order(self):
        payload = _jsonstat({"2026-08": 6.2, "2026-06": 6.0, "2026-07": 6.1})
        assert em.parse_jsonstat(payload) == (("2026-06", 6.0), ("2026-07", 6.1), ("2026-08", 6.2))

    def test_jsonstat_skips_missing_values_in_a_sparse_payload(self):
        payload = _jsonstat({"2026-06": 1.0, "2026-07": 2.0})
        del payload["value"]["0"]
        assert em.parse_jsonstat(payload) == (("2026-07", 2.0),)

    def test_jsonstat_with_a_dimension_after_time_uses_its_stride(self):
        payload = {
            "id": ["time", "geo"], "size": [2, 1],
            "dimension": {"time": {"category": {"index": {"2026-07": 0, "2026-08": 1}}}},
            "value": [1.5, 2.5],
        }
        assert em.parse_jsonstat(payload) == (("2026-07", 1.5), ("2026-08", 2.5))

    def test_jsonstat_refuses_an_unfiltered_dimension(self):
        payload = _jsonstat({"2026-07": 1.0})
        payload["size"][1] = 5
        with pytest.raises(ValueError, match="ambiguous"):
            em.parse_jsonstat(payload)

    def test_ecb_csv_skips_missing_observations(self):
        assert em.parse_ecb_csv(_ECB_CSV) == (("2026-09-10", 2.25), ("2026-09-11", 2.0))

    def test_unexpected_ecb_format_is_a_vendor_problem_not_a_crash(self):
        with pytest.raises(VendorUnavailableError):
            em.parse_ecb_csv("<html>maintenance</html>")


# ---------------------------------------------------------------- point in time


@pytest.mark.unit
class TestPointInTime:
    def test_an_observation_waits_for_its_period_end_and_publication_lag(self):
        # August HICP (lag 20 days) is public from 20 September, not before.
        assert not em.published_by("2026-08", 20, date(2026, 9, 19))
        assert em.published_by("2026-08", 20, date(2026, 9, 20))
        # A month in progress is never shown.
        assert not em.published_by("2026-09", 0, date(2026, 9, 15))

    def test_report_hides_figures_not_yet_published_on_the_analysis_date(self):
        payload = _jsonstat({"2026-06": 1.6, "2026-07": 1.7, "2026-08": 1.9})
        with mock.patch.object(em.requests, "get", return_value=_response(json_body=payload)):
            report = em.get_euro_macro_data("italy_hicp", "2026-09-15", 120)
        assert "| 2026-07 | 1.7 |" in report
        assert "2026-08" not in report  # released after 15 September
        assert "**Latest:** 1.7 (2026-07)" in report
        assert "no vintages" in report

    def test_revised_series_say_so(self):
        payload = _jsonstat({"2026-Q1": 0.3, "2026-Q2": 0.2})
        with mock.patch.object(em.requests, "get", return_value=_response(json_body=payload)):
            report = em.get_euro_macro_data("italy_gdp", "2026-10-26")
        assert "often revised" in report
        assert "| 2026-Q2 | 0.2 |" in report


# ---------------------------------------------------------------- report content


@pytest.mark.unit
class TestReports:
    def test_eurostat_request_carries_the_series_filters(self):
        payload = _jsonstat({"2026-07": 99.6})
        with mock.patch.object(em.requests, "get", return_value=_response(json_body=payload)) as get:
            em.get_euro_macro_data("italy_economic_sentiment", "2026-10-26")
        url, = get.call_args.args
        params = get.call_args.kwargs["params"]
        assert url.endswith("/ei_bssi_m_r2")
        assert params["geo"] == "IT" and params["indic"] == "BS-ESI-I" and params["s_adj"] == "SA"

    def test_ecb_series(self):
        with mock.patch.object(em.requests, "get", return_value=_response(text=_ECB_CSV)) as get:
            report = em.get_euro_macro_data("deposit rate", "2026-09-20", 30)
        assert get.call_args.args[0].endswith("/FM/D.U2.EUR.4F.KR.DFR.LEV")
        assert get.call_args.kwargs["params"]["endPeriod"] == "2026-09-20"
        assert "**Latest:** 2 (2026-09-11)" in report

    def test_btp_bund_spread_in_basis_points(self):
        italy = _jsonstat({"2026-06": 3.734, "2026-07": 3.70})
        germany = _jsonstat({"2026-06": 2.60, "2026-07": 2.65})

        def get(url, params, **kw):
            return _response(json_body=italy if params["geo"] == "IT" else germany)

        with mock.patch.object(em.requests, "get", side_effect=get):
            report = em.get_euro_macro_data("btp_bund_spread", "2026-10-26")
        assert "| 2026-06 | 113.4 |" in report
        assert "| 2026-07 | 105 |" in report
        assert "basis points" in report

    def test_unknown_indicator_lists_the_choices_without_a_request(self):
        with mock.patch.object(em.requests, "get") as get:
            report = em.get_euro_macro_data("bank of japan rate", "2026-10-26")
        get.assert_not_called()
        assert "not a known European macro series" in report and "btp_bund_spread" in report

    def test_empty_window_says_so(self):
        with mock.patch.object(em.requests, "get", return_value=_response(json_body=_jsonstat({"2020-01": 1.0}))):
            report = em.get_euro_macro_data("italy_unemployment", "2026-10-26", 30)
        assert "No observation published in this window" in report

    def test_every_alias_resolves(self):
        for alias in em.AVAILABLE:
            assert em._canonical(alias) == alias


# ---------------------------------------------------------------- failure handling


@pytest.mark.unit
class TestFailures:
    def test_network_error_is_vendor_unavailable(self):
        with (mock.patch.object(em.requests, "get", side_effect=requests.ConnectionError("down")),
              pytest.raises(VendorUnavailableError)):
            em.get_euro_macro_data("italy_hicp", "2026-10-26")

    def test_server_error_is_vendor_unavailable(self):
        with (mock.patch.object(em.requests, "get", return_value=_response(status=503)),
              pytest.raises(VendorUnavailableError)):
            em.get_euro_macro_data("italy_hicp", "2026-10-26")

    def test_router_degrades_to_a_sentinel_instead_of_ending_the_run(self):
        with mock.patch.object(em.requests, "get", side_effect=requests.ConnectionError("down")):
            out = router.route_to_vendor("get_euro_macro_indicators", "italy_hicp", "2026-10-26", None)
        assert out.startswith("DATA_UNAVAILABLE")

    def test_the_tool_never_serves_data_past_the_analysis_date(self):
        from tradingagents.agents import tools

        with mock.patch.object(tools, "route_to_vendor", return_value="ok") as route:
            tools.get_euro_macro_indicators.func("italy_hicp", "2030-01-01", None, trade_date="2026-09-15")
        assert route.call_args.args[2] == "2026-09-15"


# ---------------------------------------------------------------- analyst wiring


@pytest.mark.unit
class TestAnalystWiring:
    def test_spec(self):
        spec = ANALYST_NODE_SPECS["euro_macro"]
        assert spec.agent_node == "European Macro Analyst"
        assert spec.report_key == "euro_macro_report"
        assert {t.name for t in spec.tools} == {"get_euro_macro_indicators", "get_global_news"}

    def test_report_reaches_the_debate_only_when_there_is_one(self):
        assert euro_macro_line({}) == ""
        assert euro_macro_line({"euro_macro_report": "  "}) == ""
        line = euro_macro_line({"euro_macro_report": "Spread at 105bp"})
        assert line.startswith("European macro report") and "Spread at 105bp" in line

    def test_crypto_runs_leave_it_out(self):
        from cli.models import AnalystType, AssetType
        from cli.prompts import filter_analysts_for_asset_type

        assert AnalystType.EURO_MACRO not in filter_analysts_for_asset_type(list(AnalystType), AssetType.CRYPTO)
        assert AnalystType.EURO_MACRO in filter_analysts_for_asset_type(list(AnalystType), AssetType.STOCK)

    def test_saved_report_has_a_section_for_it(self):
        from tradingagents.reporting import _report_parts

        parts = _report_parts({"euro_macro_report": "BCE ferma al 2%"})
        assert any(agent == "European Macro Analyst" for _, _, written in parts for agent, _, _ in written)


# ---------------------------------------------------------------- tickers


@pytest.mark.unit
class TestItalianTickers:
    @pytest.mark.parametrize("raw, expected", [
        ("ENI", "ENI.MI"),
        ("isp", "ISP.MI"),
        ("UCG.MI", "UCG.MI"),
        ("AAPL.US", "AAPL"),
        ("SAP.DE", "SAP.DE"),
        ("FTSEMIB", "FTSEMIB.MI"),
        ("IT40", "FTSEMIB.MI"),
        ("^GSPC", "^GSPC"),
        ("GC=F", "GC=F"),
        ("EURUSD", "EURUSD=X"),
        ("BTCUSD", "BTC-USD"),
        ("0700.HK", "0700.HK"),
    ])
    def test_bare_tickers_take_the_milan_listing(self, raw, expected):
        assert apply_default_exchange(raw, ".MI") == expected

    @pytest.mark.parametrize("suffix", [None, "", "none", "off"])
    def test_the_suffix_can_be_turned_off(self, suffix):
        assert apply_default_exchange("AAPL", suffix) == "AAPL"

    def test_milan_tickers_benchmark_against_the_ftse_mib(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["benchmark_map"][".MI"] == "FTSEMIB.MI"

    def test_fork_defaults(self):
        from tradingagents.default_config import build_default_config

        config = build_default_config()
        assert config["output_language"] == "Italian"
        assert config["default_exchange_suffix"] == ".MI"
        assert config["data_vendors"]["euro_macro_data"] == "ecb_eurostat"


# ---------------------------------------------------------------- live APIs


@pytest.mark.integration
@pytest.mark.parametrize("alias", em.AVAILABLE)
def test_live_series_returns_observations(alias):
    report = em.get_euro_macro_data(alias, date.today().isoformat(), 400)
    assert "**Latest:**" in report, report
