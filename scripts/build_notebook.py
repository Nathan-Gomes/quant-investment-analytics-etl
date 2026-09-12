"""Generate and execute a portable research notebook with embedded static figures."""
from pathlib import Path
import sys
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

root = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.cells = [
    nbf.v4.new_markdown_cell("# Quantitative Investment Analytics ETL Pipeline\n\nFour Canadian equity portfolios, auditable SQL analytics, and purged volatility prediction. This notebook reads the last completed pipeline run. Charts are embedded as PNG outputs for notebook viewers. See the source modules for the executable implementation."),
    nbf.v4.new_code_cell("from pathlib import Path\nimport json, sqlite3\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom IPython.display import display\n%matplotlib inline\nROOT = Path.cwd()\nif not (ROOT / 'output').exists():\n    ROOT = ROOT.parent\nmanifest = json.loads((ROOT / 'output/run_manifest.json').read_text())\nprint(manifest['source']['source'])\nprint(manifest['config'])\nconn = sqlite3.connect(ROOT / 'output/investment_analytics.db')\nsummary = pd.read_sql('SELECT * FROM portfolio_summary', conn)\ndisplay(summary)"),
    nbf.v4.new_markdown_cell("## Portfolio performance and drawdown\nPrevious-close weights earn daily returns. Month-start rebalancing deducts costs. These are selected equity allocations, not an optimized investment recommendation."),
    nbf.v4.new_code_cell("daily = pd.read_sql('SELECT * FROM portfolio_daily_summary ORDER BY date', conn, parse_dates=['date'])\nfig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)\nfor name, group in daily.groupby('portfolio_id'):\n    axes[0].plot(group.date, group.nav, label=name)\n    axes[1].plot(group.date, group.drawdown * 100, label=name)\naxes[0].set(title='Portfolio value after transaction costs', ylabel='CAD')\naxes[1].set(title='Drawdown', ylabel='% below peak')\naxes[0].legend(ncol=3)\nfor ax in axes: ax.grid(alpha=.2)\nplt.tight_layout()\nplt.show()"),
    nbf.v4.new_markdown_cell("## SQL data mart\nThe view joins position facts to the security dimension. Monthly returns use a CTE and LAG over month-end NAV."),
    nbf.v4.new_code_cell("print((ROOT / 'sql/analysis_queries.sql').read_text())\nexposure = pd.read_sql('SELECT * FROM latest_sector_exposure', conn)\ndisplay(exposure)\nexposure[exposure.portfolio_id != 'Benchmark'].pivot(index='portfolio_id', columns='sector', values='weight').plot.bar(stacked=True, figsize=(10, 5))\nplt.title('Latest sector exposure')\nplt.ylabel('Portfolio weight')\nplt.xticks(rotation=0)\nplt.tight_layout()\nplt.show()"),
    nbf.v4.new_markdown_cell("## Machine learning validation\nPredict next-20-session volatility from past returns, volatility and market volume. All scaling is inside cross-validation. Training labels end before validation begins. Compare held-out RMSE with persistence; ML may lose."),
    nbf.v4.new_code_cell("scores = pd.read_sql('SELECT * FROM model_scores', conn)\ndisplay(scores[['portfolio_id', 'model', 'rmse', 'r2', 'selected_by_cv']])\nscores.pivot(index='portfolio_id', columns='model', values='rmse').plot.bar(figsize=(12, 5))\nplt.title('Held-out volatility prediction RMSE: lower is better')\nplt.xticks(rotation=0)\nplt.tight_layout()\nplt.show()\ndisplay(pd.read_sql('SELECT * FROM validation_splits', conn))"),
    nbf.v4.new_markdown_cell("## Implementation\nThe following cells display the actual pipeline source used to generate these results."),
]
nb.cells[4:4] = [
    nbf.v4.new_markdown_cell("## Equal-capital five-year strategy comparison\nAll portfolios and XIC begin with CAD 100,000 on the same date and share sampled 20-session blocks. The main chart, uncertainty panels and SQL summary use equal capital. The final, secondary chart continues historical wealth using each portfolio's prior ending balance. Scenario medians depend on the historical sample; strong past growth is carried forward, not independently forecast. Ranges omit parameter uncertainty and unseen regimes. The horizon is exactly five calendar years with 252 modeled steps per year, not exchange calendar dates."),
    nbf.v4.new_code_cell("from IPython.display import Image\nfor filename in ['forward_indexed.png', 'forward_ranges.png']:\n    display(Image(filename=str(ROOT / 'output' / filename)))\ndisplay(pd.read_sql('SELECT * FROM forward_projection_summary', conn))\ndisplay(Image(filename=str(ROOT / 'output' / 'forward_scenarios.png')))"),
    nbf.v4.new_markdown_cell("## Backtesting and look-ahead controls\nReturns use previous-close weights; low-volatility targets use only the initial calibration window. Later prices cannot affect earlier weights. Regression labels are purged at chronological split boundaries and scaling stays within training folds. These timing controls do not remove retrospectively chosen securities, manual allocations or survivorship bias. Five-year scenarios use completed history at the final as-of date, never feed into earlier decisions, and have not been validated as five-year forecasts."),
]
for name in ("extract", "validation", "analytics", "forward", "models", "load", "pipeline"):
    source = (root / "src" / f"{name}.py").read_text()
    nb.cells.append(nbf.v4.new_markdown_cell(f"### {name}.py\n```python\n{source}\n```"))
nb.cells.append(nbf.v4.new_code_cell("conn.close()"))
ExecutePreprocessor(timeout=180, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(root)}})
directory = root / "notebooks"
directory.mkdir(exist_ok=True)
nbf.write(nb, directory / "portfolio_exploration.ipynb")
template_root = Path(sys.base_prefix) / "share/jupyter/nbconvert/templates"
body, _ = HTMLExporter(extra_template_basedirs=[str(template_root)], extra_template_paths=[str(template_root)]).from_notebook_node(nb)
(root / "output/notebook.html").write_text(body)
print(directory / "portfolio_exploration.ipynb")
