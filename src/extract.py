import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd


def extract(root, config, source):
    securities = pd.read_csv(root / "data/securities.csv")
    holdings = pd.read_csv(root / "data/holdings.csv")
    path = root / "data" / ("synthetic_prices.csv" if source == "synthetic" else "cached_prices.csv")
    metadata_path = path.with_suffix(".json")
    if source == "download":
        import yfinance as yf
        frames = []
        for ticker in securities.ticker:
            data = yf.download(ticker, start=config["start"], end=config["end"],
                               auto_adjust=False, progress=False, threads=False)
            if data.empty:
                raise ValueError(f"No market data returned for {ticker}; use --source synthetic explicitly for a demo.")
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            frames.append(pd.DataFrame({"date": data.index.strftime("%Y-%m-%d"),
                                        "ticker": ticker, "adjusted_price": data["Adj Close"].to_numpy(),
                                        "volume": data.Volume.to_numpy()}))
        prices = pd.concat(frames, ignore_index=True)
    elif source == "synthetic":
        rng = np.random.default_rng(config["seed"])
        dates = pd.bdate_range(config["start"], pd.Timestamp(config["end"]) - pd.Timedelta(days=1))
        market = rng.normal(0.00025, 0.01, len(dates))
        frames = []
        for row in securities.itertuples():
            scale = {"Technology": 0.018, "Utilities": 0.006}.get(row.sector, 0.009)
            r = market * (0.65 if row.sector == "Utilities" else 1) + rng.normal(0, scale, len(dates))
            frames.append(pd.DataFrame({"date": dates, "ticker": row.ticker,
                                        "adjusted_price": 100 * np.cumprod(1 + r),
                                        "volume": rng.integers(100000, 3000000, len(dates))}))
        prices = pd.concat(frames, ignore_index=True)
    else:
        prices = pd.read_csv(path)
        metadata = json.loads(metadata_path.read_text())
        if metadata["sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Cached price checksum does not match provenance metadata")
        return prices, holdings, securities, metadata
    prices.to_csv(path, index=False)
    metadata = {"source": "Yahoo Finance via yfinance" if source == "download" else "SYNTHETIC DEMONSTRATION",
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "start": config["start"], "end_exclusive": config["end"],
                "price_basis": "Dividend/split adjusted close; CAD"}
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return prices, holdings, securities, metadata
