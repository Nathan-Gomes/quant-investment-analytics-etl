-- Latest allocations, with sector names obtained from the security dimension.
CREATE VIEW latest_sector_exposure AS
WITH latest AS (SELECT MAX(date) AS date FROM portfolio_positions)
SELECT p.portfolio_id, s.sector, SUM(p.market_value) AS market_value,
       SUM(p.weight) AS weight
FROM portfolio_positions p
JOIN securities s ON s.ticker = p.ticker
JOIN latest l ON l.date = p.date
GROUP BY p.portfolio_id, s.sector;

-- Compound monthly returns from month-end NAV using a window function.
CREATE VIEW monthly_portfolio_returns AS
WITH ranked AS (
 SELECT *, SUBSTR(date, 1, 7) AS month,
 ROW_NUMBER() OVER (PARTITION BY portfolio_id, SUBSTR(date, 1, 7) ORDER BY date DESC) AS rn
 FROM portfolio_daily_summary
), endpoints AS (
 SELECT portfolio_id, month, nav FROM ranked WHERE rn = 1
)
SELECT portfolio_id, month, nav,
 nav / LAG(nav) OVER (PARTITION BY portfolio_id ORDER BY month) - 1 AS monthly_return
FROM endpoints;

-- Daily cross-sectional ranking by trailing annualized volatility.
CREATE VIEW volatility_ranking AS
SELECT date, ticker, rolling_volatility_20,
 DENSE_RANK() OVER (PARTITION BY date ORDER BY rolling_volatility_20) AS volatility_rank
FROM security_daily_analytics WHERE rolling_volatility_20 IS NOT NULL;

CREATE VIEW rolling_portfolio_returns AS
SELECT date, portfolio_id,
 nav / LAG(nav, 20) OVER (PARTITION BY portfolio_id ORDER BY date) - 1 AS return_20_sessions
FROM portfolio_daily_summary;
