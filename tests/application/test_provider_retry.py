"""Provider failures must not change the requested source or universe."""
import pandas as pd
import pytest

from app import marketdata


@pytest.mark.parametrize("message", ["HTTP 429", "YFRateLimitError", "Too Many Requests", "throttled"])
def test_rate_limit_detection(message):
    assert marketdata._is_rate_limited(RuntimeError(message))


def test_retry_and_known_metadata(monkeypatch, tmp_path):
    class Ticker:
        calls = 0

        def history(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("HTTP 429")
            return pd.DataFrame({"Close": [100., 101.]}, index=pd.date_range("2025-01-01", periods=2))

        def get_info(self):
            pytest.fail("Known securities must not request a second profile")

    ticker = Ticker()
    class Provider:
        Ticker = staticmethod(lambda symbol: ticker)

    monkeypatch.setattr(marketdata, "require_provider", lambda: Provider)
    monkeypatch.setattr(marketdata, "CACHE", tmp_path)
    monkeypatch.setattr(marketdata.time, "sleep", lambda delay: None)
    prices, info = marketdata.download("RY.TO", "2025-01-01", "2025-01-02")
    assert ticker.calls == 3
    assert len(prices) == 2
    assert info["currency"] == "CAD"


@pytest.mark.parametrize("source", ["auto", "yahoo"])
def test_refusal_never_substitutes_bundled_prices(monkeypatch, source):
    monkeypatch.setattr(marketdata, "require_provider", lambda: object())
    def refuse(*args):
        raise RuntimeError("HTTP 429")
    monkeypatch.setattr(marketdata, "download", refuse)
    monkeypatch.setattr(marketdata, "load_bundled", lambda *args: pytest.fail("Unexpected frozen fallback"))
    with pytest.raises(ValueError, match="429"):
        marketdata.load(["RY.TO"], "2025-01-01", "2025-12-31", source)


def test_non_rate_errors_are_not_retryable():
    assert not marketdata._is_rate_limited(ValueError("404 symbol not found"))
