import { useEffect, useMemo, useState } from "react";

const fmtPct = (value) =>
  typeof value === "number" && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
const fmtSignedPct = (value) =>
  typeof value === "number" && Number.isFinite(value) ? `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)}%` : "—";
const fmtFloat = (value, digits = 2) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "—";
const fmtInr = (value) =>
  typeof value === "number" && Number.isFinite(value)
    ? new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 }).format(value)
    : "—";
const asObject = (value) => (value && typeof value === "object" && !Array.isArray(value) ? value : {});
const asArray = (value) => (Array.isArray(value) ? value : []);
const assetLabel = (value) => (value === "CASH" ? "LIQUIDBEES" : value === "US" ? "MON100" : value);
const rowDate = (row) => row?.date || row?.as_of || null;
const pendingCount = (value, fallback = 0) => (Array.isArray(value) ? value.length : Number.isFinite(Number(value)) ? Number(value) : fallback);
const SERIES_COLORS = {
  PORTFOLIO_V9: "#0f766e",
  EQWT_RISKY: "#b45309",
  NIFTY: "#2563eb",
  MIDCAP: "#7c3aed",
  SMALLCAP: "#db2777",
  GOLD: "#ca8a04",
  SILVER: "#64748b",
  US: "#0ea5e9",
  CASH: "#6b7280",
};

function Card({ title, subtitle, children, className = "" }) {
  return (
    <section className={`card ${className}`}>
      <div className="card-head">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p className="card-subtitle">{subtitle}</p> : null}
        </div>
      </div>
      {children}
    </section>
  );
}

function Metric({ label, value, hint }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {hint ? <div className="metric-hint">{hint}</div> : null}
    </div>
  );
}

function Table({ rows, columns, empty = "No data yet." }) {
  if (!rows || rows.length === 0) return <div className="empty">{empty}</div>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={row.id || row.signal_date || row.as_of || row.asset || idx}>
              {columns.map((column) => (
                <td key={column.key}>{column.render ? column.render(row) : row[column.key] ?? "—"}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AllocationBars({ data }) {
  const entries = Object.entries(data || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return <div className="empty">No weights yet.</div>;
  return (
    <div className="bars">
      {entries.map(([asset, weight]) => (
        <div key={asset} className="bar-row">
          <div className="bar-label">
            <span>{assetLabel(asset)}</span>
            <span>{fmtPct(weight)}</span>
          </div>
          <div className="bar-track">
            <div className="bar-fill" style={{ width: `${Math.max(weight * 100, 2)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function LineChart({ rows, keys, colors = SERIES_COLORS, height = 260, valueFormatter = fmtFloat }) {
  const safeRows = asArray(rows).filter((row) => keys.some((key) => typeof row?.[key] === "number" && Number.isFinite(row[key])));
  if (!safeRows.length || !keys.length) return <div className="empty">No chart data yet.</div>;

  const width = 920;
  const padding = { top: 20, right: 20, bottom: 28, left: 44 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  const values = safeRows.flatMap((row) => keys.map((key) => row[key]).filter((value) => typeof value === "number" && Number.isFinite(value)));
  const minY = Math.min(...values);
  const maxY = Math.max(...values);
  const span = maxY - minY || 1;

  const xAt = (index) => padding.left + (safeRows.length === 1 ? innerWidth / 2 : (index / (safeRows.length - 1)) * innerWidth);
  const yAt = (value) => padding.top + innerHeight - ((value - minY) / span) * innerHeight;

  const ticks = [0, 0.5, 1].map((step) => minY + span * step);

  return (
    <div className="chart-wrap">
      <div className="chart-legend">
        {keys.map((key) => (
          <div key={key} className="legend-item">
            <span className="legend-swatch" style={{ background: colors[key] || "#0f766e" }} />
            <span>{assetLabel(key)}</span>
          </div>
        ))}
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg" role="img" aria-label="Line chart">
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={padding.left} x2={width - padding.right} y1={yAt(tick)} y2={yAt(tick)} className="chart-grid" />
            <text x={8} y={yAt(tick) + 4} className="chart-axis-label">
              {valueFormatter(tick)}
            </text>
          </g>
        ))}
        {keys.map((key) => {
          const points = safeRows
            .map((row, index) => (typeof row?.[key] === "number" && Number.isFinite(row[key]) ? `${xAt(index)},${yAt(row[key])}` : null))
            .filter(Boolean)
            .join(" ");
          if (!points) return null;
          return <polyline key={key} fill="none" stroke={colors[key] || "#0f766e"} strokeWidth="2.5" points={points} />;
        })}
      </svg>
      <div className="chart-footer">
        <span>{safeRows[0]?.date || "—"}</span>
        <span>{safeRows[safeRows.length - 1]?.date || "—"}</span>
      </div>
    </div>
  );
}

function StackedAreaChart({ rows, keys, colors = SERIES_COLORS, height = 260 }) {
  const safeRows = asArray(rows).filter((row) => keys.some((key) => typeof row?.[key] === "number"));
  if (!safeRows.length || !keys.length) return <div className="empty">No allocation history yet.</div>;

  const width = 920;
  const padding = { top: 18, right: 18, bottom: 28, left: 44 };
  const innerWidth = width - padding.left - padding.right;
  const innerHeight = height - padding.top - padding.bottom;

  const xAt = (index) => padding.left + (safeRows.length === 1 ? innerWidth / 2 : (index / (safeRows.length - 1)) * innerWidth);
  const yAt = (value) => padding.top + innerHeight - value * innerHeight;

  const layerPaths = [];
  let previous = new Array(safeRows.length).fill(0);
  for (const key of keys) {
    const current = safeRows.map((row, idx) => previous[idx] + (Number(row[key]) || 0));
    const upper = current.map((value, idx) => `${xAt(idx)},${yAt(value)}`).join(" ");
    const lower = previous
      .map((value, idx) => `${xAt(safeRows.length - 1 - idx)},${yAt(previous[safeRows.length - 1 - idx])}`)
      .join(" ");
    layerPaths.push({ key, points: `${upper} ${lower}`, color: colors[key] || "#0f766e" });
    previous = current;
  }

  return (
    <div className="chart-wrap">
      <div className="chart-legend">
        {keys.map((key) => (
          <div key={key} className="legend-item">
            <span className="legend-swatch" style={{ background: colors[key] || "#0f766e" }} />
            <span>{assetLabel(key)}</span>
          </div>
        ))}
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg" role="img" aria-label="Allocation split chart">
        {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
          <g key={tick}>
            <line x1={padding.left} x2={width - padding.right} y1={yAt(tick)} y2={yAt(tick)} className="chart-grid" />
            <text x={8} y={yAt(tick) + 4} className="chart-axis-label">
              {fmtPct(tick)}
            </text>
          </g>
        ))}
        {layerPaths.map((layer) => (
          <polygon key={layer.key} points={layer.points} fill={layer.color} opacity="0.8" />
        ))}
      </svg>
      <div className="chart-footer">
        <span>{safeRows[0]?.date || "—"}</span>
        <span>{safeRows[safeRows.length - 1]?.date || "—"}</span>
      </div>
    </div>
  );
}

function TabButton({ id, active, onClick, children }) {
  return (
    <button type="button" className={`tab-button ${active ? "is-active" : ""}`} onClick={() => onClick(id)}>
      {children}
    </button>
  );
}

function CodeBlock({ children }) {
  return (
    <pre className="code-block">
      <code>{children}</code>
    </pre>
  );
}

function StatusNote({ children, tone = "neutral" }) {
  return <div className={`status-note status-${tone}`}>{children}</div>;
}

function mergePaperCurves(backfillRows, liveRows) {
  const merged = new Map();

  for (const row of [...asArray(backfillRows), ...asArray(liveRows)]) {
    const date = rowDate(row);
    if (!date) continue;
    const normalized = {
      ...row,
      date,
      as_of: date,
    };
    merged.set(date, {
      ...(merged.get(date) || {}),
      ...normalized,
    });
  }

  return Array.from(merged.values()).sort((a, b) => String(rowDate(a)).localeCompare(String(rowDate(b))));
}

export default function App() {
  const [dashboard, setDashboard] = useState(null);
  const [error, setError] = useState("");
  const [tab, setTab] = useState("overview");
  const [overviewMode, setOverviewMode] = useState("simple");
  const [selectedSeries, setSelectedSeries] = useState("PORTFOLIO_V9");
  const [researchTab, setResearchTab] = useState("stats");

  useEffect(() => {
    let isMounted = true;

    const loadDashboard = async () => {
      const candidates = ["/api/dashboard", "/data/dashboard.json"];
      let lastError = null;

      for (const url of candidates) {
        try {
          const response = await fetch(url, { cache: "no-store" });
          if (!response.ok) throw new Error(`Failed to load dashboard snapshot from ${url}: ${response.status}`);
          const text = await response.text();
          const payload = JSON.parse(text);
          if (isMounted) {
            setDashboard(payload);
            setError("");
          }
          return;
        } catch (err) {
          lastError = err;
        }
      }

      if (isMounted) {
        setError(lastError?.message || "Failed to load dashboard snapshot.");
      }
    };

    loadDashboard();
    const intervalId = window.setInterval(loadDashboard, 60_000);
    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, []);

  const live = asObject(dashboard?.live_signal?.models);
  const v9 = asObject(live.v9);
  const v9Rows = asArray(v9.rows);
  const latestV9Signal = v9Rows.length ? v9Rows[v9Rows.length - 1] : null;
  const recentV9Rows = v9Rows.slice(-5).reverse();

  const marketClock = asObject(dashboard?.daily_cycle?.market_clock || dashboard?.live_signal?.market_clock);
  const execution = asObject(dashboard?.execution_plan);
  const executionBase = asObject(dashboard?.execution_plan_base);
  const executionOrders = asArray(execution.orders);
  const paperTrading = asObject(dashboard?.paper_trading);
  const paperBase = asObject(dashboard?.paper_base);
  const paperComparison = asObject(dashboard?.paper_comparison);
  const paperBackfill = asObject(dashboard?.paper_backfill);
  const auditRuns = asArray(dashboard?.audit_runs);
  const latestAudit = auditRuns.length ? auditRuns[auditRuns.length - 1] : {};
  const v9Weights = asObject(dashboard?.model_allocation || latestAudit.v9_final_weights || dashboard?.current_allocation || execution.target_weights);
  const paperSummary = asObject(latestAudit.paper_summary || paperTrading);
  const paperPositions = asArray(paperTrading.positions).filter((row) => Number(row?.quantity || 0) > 0);
  const paperCurve = useMemo(
    () => mergePaperCurves(paperBackfill.calendar_curve, paperTrading.equity_curve || paperTrading.history),
    [paperBackfill.calendar_curve, paperTrading.equity_curve, paperTrading.history],
  );
  const paperStartDate = paperCurve[0]?.as_of || paperBackfill.start || "—";
  const paperEndDate = paperCurve[paperCurve.length - 1]?.as_of || paperTrading.as_of || paperBackfill.end || "—";
  const paperInitialEquity = paperCurve[0]?.equity ?? paperBackfill.initial_cash;
  const paperFinalEquity = paperCurve[paperCurve.length - 1]?.equity ?? paperBackfill.final_equity ?? paperTrading.total_equity;
  const paperCurveRows = paperCurve.map((row) => ({
    ...row,
    date: rowDate(row),
    PAPER_EQUITY: Number(row.equity),
  }));
  const validation = asObject(dashboard?.validation);
  const significance = asObject(dashboard?.significance);
  const analytics = asObject(validation.asset_portfolio);
  const overlay = asObject(dashboard?.overlay);
  const overlayRaw = asObject(dashboard?.overlay_raw);
  const overlayActive = asObject(dashboard?.overlay_active);
  const overlayCurrent = dashboard?.overlay_latest || null;
  const learningState = asObject(dashboard?.learning_state);
  const overlayPolicy = asObject(learningState.overlay_policy);
  const benchmarkRows = useMemo(() => asArray(dashboard?.benchmark_rows), [dashboard]);
  const otherModels = Object.values(live).filter((model) => model?.model && model.model !== "v9");
  const diagnosticsRows = asArray(analytics.diagnostics_rows);
  const expandingSeriesRows = asArray(analytics.expanding_sharpe_series);
  const rollingSeriesRows = asArray(analytics.rolling_sharpe_series);
  const equityOverlayRows = asArray(analytics.equity_curve_overlay);
  const allocationHistoryRows = asArray(analytics.allocation_history);
  const diagnosticNames = diagnosticsRows.map((row) => row.name).filter(Boolean);
  const isLoading = !dashboard && !error;

  const tabs = [
    { id: "overview", label: "V9" },
    { id: "paper", label: "Paper" },
    { id: "orders", label: "Orders" },
    { id: "research", label: "Research" },
    { id: "overlay", label: "LLM" },
    { id: "setup", label: "Setup" },
  ];

  return (
    <div className="app-shell">
      <main className="layout">
        <header className="hero">
          <div className="hero-copy">
            <div className="kicker">Trader Dashboard</div>
            <h1>V9 first. Everything else second.</h1>
            <p>
              The main tab is just the current `v9` read: weights, value, recent signals, and paper account. Research,
              validation, LLM, and setup live in separate tabs so the page stays readable.
            </p>
          </div>
          <div className="hero-strip">
            <div className="hero-chip">As of {dashboard?.as_of || "—"}</div>
            <div className="hero-chip">Market {marketClock.session || "—"}</div>
            <div className="hero-chip">Next {marketClock.next_trading_day || "—"}</div>
          </div>
        </header>

        {error ? <div className="error-banner">{error}</div> : null}
        {isLoading ? <div className="empty">Loading dashboard snapshot...</div> : null}

        {!isLoading ? (
          <>
            <nav className="tabs" aria-label="Dashboard sections">
              {tabs.map((item) => (
                <TabButton key={item.id} id={item.id} active={tab === item.id} onClick={setTab}>
                  {item.label}
                </TabButton>
              ))}
            </nav>

            {tab === "overview" ? (
              <div className="stack">
                <div className="subtabs">
                  <TabButton id="simple" active={overviewMode === "simple"} onClick={setOverviewMode}>
                    Simple
                  </TabButton>
                  <TabButton id="detailed" active={overviewMode === "detailed"} onClick={setOverviewMode}>
                    Detailed
                  </TabButton>
                </div>

                <div className="split split-main">
                  <Card title="Current V9 read" subtitle={v9.label || "Current production model"}>
                    <div className="metric-grid metric-grid-main">
                      <Metric label="Latest bar" value={v9.latest_completed_bar || dashboard?.as_of || "—"} />
                      <Metric label="Next session" value={v9.actionable_for_next_session || marketClock.next_trading_day || "—"} />
                      <Metric label="Market" value={marketClock.session || "—"} hint={marketClock.holiday_name || ""} />
                      <Metric label="Paper equity" value={`₹${fmtInr(paperSummary.total_equity || paperTrading.total_equity)}`} />
                      <Metric label="Net PnL" value={fmtSignedPct(paperSummary.return_since_start)} hint={`₹${fmtInr(paperSummary.net_pnl)}`} />
                      <Metric label="Queued paper fills" value={String(pendingCount(paperSummary.pending_orders, executionOrders.length || 0))} />
                    </div>
                    {dashboard?.daily_cycle?.market_closed_skip ? (
                      <StatusNote tone="neutral">
                        No order generation today. Reason: {dashboard.daily_cycle.skip_reason}. Next trading day is{" "}
                        {marketClock.next_trading_day}.
                      </StatusNote>
                    ) : null}
                    <StatusNote tone={latestV9Signal?.changed ? "accent" : "neutral"}>
                      Latest signal: {latestV9Signal?.changed ? "portfolio changed" : "no material change"} for{" "}
                      {latestV9Signal?.action_date || marketClock.next_trading_day || "the next session"}.
                    </StatusNote>
                    {overviewMode === "simple" ? (
                      <StatusNote tone="neutral">
                        Simple view shows only the current stance, value, and allocation. Switch to Detailed if you want
                        recent signal history, paper positions, and historical charts.
                      </StatusNote>
                    ) : null}
                  </Card>

                  <Card title="Allocation" subtitle={v9.current_allocation || "Current target weights"}>
                    <AllocationBars data={v9Weights} />
                  </Card>
                </div>

                {overviewMode === "detailed" ? (
                  <>
                    <div className="split">
                      <Card title="Recent V9 signals" subtitle="What the model did over the last few completed bars">
                        <Table
                          rows={recentV9Rows}
                          columns={[
                            { key: "signal_date", label: "Signal" },
                            { key: "action_date", label: "Action" },
                            { key: "changed", label: "Changed", render: (row) => (row.changed ? "Yes" : "No") },
                            { key: "next_day_return", label: "Next day", render: (row) => fmtPct(row.next_day_return) },
                            { key: "allocation", label: "Allocation" },
                          ]}
                          empty="No v9 signal history yet."
                        />
                      </Card>

                      <Card title="Paper account" subtitle="Current paper positions from the April 1 start">
                        <div className="metric-grid">
                          <Metric label="Start cash" value={`₹${fmtInr(paperBackfill.initial_cash || paperInitialEquity)}`} />
                          <Metric label="Current cash" value={`₹${fmtInr(paperTrading.cash)}`} />
                          <Metric label="Start date" value={paperStartDate} />
                          <Metric label="Last mark" value={paperEndDate} />
                        </div>
                        <Table
                          rows={paperPositions}
                          columns={[
                            { key: "asset", label: "Asset" },
                            { key: "quantity", label: "Qty" },
                            { key: "price", label: "Price", render: (row) => fmtFloat(row.price, 2) },
                            { key: "market_value", label: "Value", render: (row) => `₹${fmtInr(row.market_value)}` },
                          ]}
                          empty="No paper positions yet."
                        />
                      </Card>
                    </div>

                    <div className="split">
                      <Card title="Portfolio value over time" subtitle={`V9 vs ${analytics.benchmark_overlay || "fixed benchmark"} from the honest analysis sample`}>
                        <LineChart
                          rows={equityOverlayRows}
                          keys={["PORTFOLIO_V9", analytics.benchmark_overlay || "EQWT_RISKY"]}
                          valueFormatter={(value) => `₹${fmtInr(value)}`}
                        />
                      </Card>

                      <Card title="Allocation split over time" subtitle="How v9 moved money across sleeves through time">
                        <StackedAreaChart rows={allocationHistoryRows} keys={["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US", "CASH"]} />
                      </Card>
                    </div>
                  </>
                ) : null}
              </div>
            ) : null}

            {tab === "paper" ? (
              <div className="stack">
                <Card title="Paper equity path" subtitle="Day-by-day carry from the configured paper start">
                  <div className="metric-grid">
                    <Metric label="Start" value={paperStartDate} />
                    <Metric label="End" value={paperEndDate} />
                    <Metric label="Initial" value={`₹${fmtInr(paperInitialEquity)}`} />
                    <Metric label="Final" value={`₹${fmtInr(paperFinalEquity)}`} />
                  </div>
                  <LineChart rows={paperCurveRows} keys={["PAPER_EQUITY"]} colors={{ PAPER_EQUITY: "#0f766e" }} valueFormatter={(value) => `₹${fmtInr(value)}`} />
                </Card>

                <Card title="Paper ledger" subtitle="Daily marks, holidays, and queued paper fills from April 1 onward">
                  <Table
                    rows={paperCurve.slice(-15).reverse()}
                    columns={[
                      { key: "as_of", label: "Date" },
                      { key: "trading_day", label: "Trading", render: (row) => (row.trading_day === false ? "No" : "Yes") },
                      { key: "holiday_name", label: "Holiday" },
                      { key: "equity", label: "Equity", render: (row) => `₹${fmtInr(row.equity)}` },
                      { key: "cash", label: "Cash", render: (row) => `₹${fmtInr(row.cash)}` },
                      { key: "pending_orders", label: "Queued fills", render: (row) => (Array.isArray(row.pending_orders) ? row.pending_orders.length : row.pending_orders ?? "—") },
                    ]}
                    empty="No paper curve yet."
                  />
                </Card>

                <Card title="Paper positions" subtitle="Current held sleeves in the paper book">
                  <Table
                    rows={paperPositions}
                    columns={[
                      { key: "asset", label: "Asset" },
                      { key: "quantity", label: "Qty" },
                      { key: "price", label: "Price", render: (row) => fmtFloat(row.price, 2) },
                      { key: "market_value", label: "Value", render: (row) => `₹${fmtInr(row.market_value)}` },
                    ]}
                    empty="No paper positions yet."
                  />
                </Card>

                <Card title="Base vs LLM shadow book" subtitle="Same model day, two separate paper paths: raw v9 vs v9 plus LLM overlay">
                  <div className="metric-grid">
                    <Metric label="Base equity" value={`₹${fmtInr(paperBase.total_equity)}`} />
                    <Metric label="LLM equity" value={`₹${fmtInr(paperTrading.total_equity)}`} />
                    <Metric label="Equity delta" value={`₹${fmtInr(paperComparison.equity_delta)}`} />
                    <Metric label="Base return" value={fmtSignedPct(paperBase.return_since_start)} />
                    <Metric label="LLM return" value={fmtSignedPct(paperTrading.return_since_start)} />
                    <Metric label="Return delta" value={fmtSignedPct(paperComparison.return_delta)} />
                  </div>
                  <StatusNote tone="neutral">
                    This is the honest comparison path for deciding whether the LLM is helping. The live paper book is
                    the LLM-adjusted side; the base book is raw `v9` without narrative changes.
                  </StatusNote>
                </Card>
              </div>
            ) : null}

            {tab === "orders" ? (
              <div className="stack">
                <Card title="Execution plan" subtitle="Dry-run order plan from the latest cached target">
                  <div className="metric-grid">
                    <Metric label="Portfolio value" value={`₹${fmtInr(execution.portfolio_value)}`} />
                    <Metric label="Model cash sleeve" value={fmtPct(v9Weights.CASH)} />
                    <Metric label="Execution cash sleeve" value={fmtPct(execution.target_weights?.CASH)} />
                    <Metric label="Orders" value={String(executionOrders.length)} />
                  </div>
                  <StatusNote tone="neutral">
                    The V9 tab shows the model allocation. This tab shows the broker-facing execution target after lot sizes,
                    rounding, reserve cash, and tradable proxy constraints.
                  </StatusNote>
                  {dashboard?.daily_cycle?.market_closed_skip ? (
                    <StatusNote tone="neutral">
                      Orders are intentionally blank on Indian market holidays and stale-bar days.
                    </StatusNote>
                  ) : null}
                  <Table
                    rows={executionOrders}
                    columns={[
                      { key: "side", label: "Side" },
                      { key: "asset", label: "Asset" },
                      { key: "quantity", label: "Qty" },
                      { key: "reference_price", label: "Ref px", render: (row) => fmtFloat(row.reference_price, 2) },
                      { key: "delta_weight", label: "Weight delta", render: (row) => fmtSignedPct(row.delta_weight) },
                      { key: "delta_value", label: "Value delta", render: (row) => `₹${fmtInr(row.delta_value)}` },
                    ]}
                    empty="No orders queued for the current cycle."
                  />
                </Card>
              </div>
            ) : null}

            {tab === "research" ? (
              <div className="stack">
                <div className="subtabs">
                  <TabButton id="stats" active={researchTab === "stats"} onClick={setResearchTab}>
                    Stats
                  </TabButton>
                  <TabButton id="stability" active={researchTab === "stability"} onClick={setResearchTab}>
                    Stability
                  </TabButton>
                  <TabButton id="charts" active={researchTab === "charts"} onClick={setResearchTab}>
                    Charts
                  </TabButton>
                </div>

                {researchTab === "stats" ? (
                  <>
                    <Card title="Benchmark scorecard" subtitle="Where v9 sits against the main benchmark set">
                      <Table
                        rows={benchmarkRows}
                        columns={[
                          { key: "name", label: "Model" },
                          { key: "source", label: "Slice" },
                          { key: "cagr", label: "CAGR", render: (row) => fmtPct(row.cagr) },
                          { key: "sharpe", label: "Sharpe", render: (row) => fmtFloat(row.sharpe) },
                          { key: "mdd", label: "MaxDD", render: (row) => fmtPct(row.mdd) },
                          { key: "turnover", label: "Turn", render: (row) => fmtPct(row.turnover) },
                        ]}
                      />
                    </Card>

                    <div className="split">
                      <Card title="Significance" subtitle="Statistical read on the strategy quality">
                        <div className="metric-grid">
                          <Metric label="v9 HAC p" value={fmtFloat(significance.models?.v9?.hac_p_value, 4)} />
                          <Metric label="EqWt HAC p" value={fmtFloat(significance.models?.eqwt?.hac_p_value, 4)} />
                          <Metric label="Holm v9" value={fmtFloat(validation.holm_models?.v9, 4)} />
                          <Metric label="Boot Sharpe" value={fmtFloat(validation.models?.v9?.bootstrap_sharpe?.p50, 2)} />
                        </div>
                      </Card>

                      <Card title="Other model branches" subtitle="Kept for research, not the default run">
                        <Table
                          rows={otherModels}
                          columns={[
                            { key: "model", label: "Model" },
                            { key: "label", label: "Label" },
                            { key: "current_allocation", label: "Current read" },
                          ]}
                          empty="No alternate models found."
                        />
                      </Card>
                    </div>

                    <Card title="Sharpe t-test and distribution diagnostics" subtitle="Per asset, plus the v9 portfolio and the fixed benchmark">
                      <Table
                        rows={diagnosticsRows}
                        columns={[
                          { key: "name", label: "Series" },
                          { key: "kind", label: "Kind" },
                          { key: "annualized_return", label: "CAGR", render: (row) => fmtPct(row.annualized_return) },
                          { key: "annualized_vol", label: "Vol", render: (row) => fmtPct(row.annualized_vol) },
                          { key: "sharpe", label: "Sharpe", render: (row) => fmtFloat(row.sharpe) },
                          { key: "sharpe_t_stat", label: "t-stat", render: (row) => fmtFloat(row.sharpe_t_stat) },
                          { key: "sharpe_p_value", label: "p", render: (row) => fmtFloat(row.sharpe_p_value, 4) },
                          { key: "skewness", label: "Skew", render: (row) => fmtFloat(row.skewness) },
                          { key: "kurtosis", label: "Kurtosis", render: (row) => fmtFloat(row.kurtosis) },
                          { key: "fat_tails", label: "Fat tails", render: (row) => (row.fat_tails ? "Yes" : "No") },
                        ]}
                        empty="No diagnostics yet."
                      />
                    </Card>
                  </>
                ) : null}

                {researchTab === "stability" ? (
                  <Card title="Sharpe stability" subtitle="Pick any asset or the portfolio to see expanding and rolling Sharpe over time">
                    <div className="series-picker">
                      {diagnosticNames.map((name) => (
                        <TabButton key={name} id={name} active={selectedSeries === name} onClick={setSelectedSeries}>
                          {name}
                        </TabButton>
                      ))}
                    </div>
                    <div className="split">
                      <Card title="Expanding Sharpe" subtitle={`Starts after ${analytics.expanding_start_days || 30} trading days`}>
                        <LineChart rows={expandingSeriesRows} keys={[selectedSeries]} />
                      </Card>
                      <Card title="Rolling Sharpe" subtitle={`${analytics.rolling_window_days || 63}-day window`}>
                        <LineChart rows={rollingSeriesRows} keys={[selectedSeries]} />
                      </Card>
                    </div>
                  </Card>
                ) : null}

                {researchTab === "charts" ? (
                  <div className="split">
                    <Card title="Portfolio value over time" subtitle={`V9 vs ${analytics.benchmark_overlay || "fixed benchmark"} on the honest stitched OOS sample`}>
                      <LineChart
                        rows={equityOverlayRows}
                        keys={["PORTFOLIO_V9", analytics.benchmark_overlay || "EQWT_RISKY"]}
                        valueFormatter={(value) => `₹${fmtInr(value)}`}
                      />
                    </Card>

                    <Card title="Allocation split over time" subtitle="How v9 moved capital across sleeves through time">
                      <StackedAreaChart rows={allocationHistoryRows} keys={["NIFTY", "MIDCAP", "SMALLCAP", "GOLD", "SILVER", "US", "CASH"]} />
                    </Card>
                  </div>
                ) : null}
              </div>
            ) : null}

            {tab === "overlay" ? (
              <div className="stack">
                <Card title="LLM overlay state" subtitle="The LLM is a risk governor, not the main allocator">
                  <div className="metric-grid">
                    <Metric label="Raw risk-off" value={fmtFloat(overlayRaw.default_risk_off_override, 2)} />
                    <Metric label="Active risk-off" value={fmtFloat(overlayActive.default_risk_off_override, 2)} />
                    <Metric label="Trust" value={fmtFloat(overlayPolicy.trust_multiplier, 2)} />
                    <Metric label="Risk cap" value={fmtFloat(overlayPolicy.max_risk_off_cap, 2)} />
                    <Metric label="Cooldown" value={String(overlayPolicy.cooldown_days_remaining ?? 0)} />
                    <Metric label="Biases" value={overlayPolicy.allow_asset_bias ? "On" : "Off"} />
                  </div>
                  <StatusNote tone={overlayCurrent ? "accent" : "neutral"}>
                    {overlayCurrent?.rationale || "No active narrative override right now."}
                  </StatusNote>
                </Card>

                <Card title="Counterfactual check" subtitle="We now keep both the raw model and the LLM-adjusted path so the overlay has to earn its keep">
                  <div className="metric-grid">
                    <Metric label="Base orders" value={String(asArray(executionBase.orders).length)} />
                    <Metric label="LLM orders" value={String(executionOrders.length)} />
                    <Metric label="Base cash" value={fmtPct(paperComparison.base_weights?.CASH)} />
                    <Metric label="LLM cash" value={fmtPct(paperComparison.llm_weights?.CASH)} />
                    <Metric label="Equity delta" value={`₹${fmtInr(paperComparison.equity_delta)}`} />
                    <Metric label="Return delta" value={fmtSignedPct(paperComparison.return_delta)} />
                  </div>
                  <StatusNote tone="neutral">
                    If the LLM keeps hurting the shadow comparison, we should scale it down or turn it off. This gives
                    us a concrete way to improve it instead of relying on vibes.
                  </StatusNote>
                </Card>

                <Card title="Learning state" subtitle="Daily trust adaptation, kept intentionally conservative">
                  <div className="metric-grid">
                    <Metric label="Updated" value={learningState.updated_at || "—"} />
                    <Metric label="Override dates" value={String(asArray(overlay.dates).length)} />
                    <Metric label="Confidence" value={fmtFloat(overlayCurrent?.confidence, 2)} />
                    <Metric label="Hold days" value={String(overlayCurrent?.holding_days || 0)} />
                  </div>
                  <Table
                    rows={auditRuns.slice(-8).reverse()}
                    columns={[
                      { key: "ran_at", label: "Run" },
                      { key: "signal_as_of", label: "As of" },
                      { key: "order_count", label: "Orders" },
                      { key: "paper_trading", label: "Paper", render: (row) => (row.paper_trading ? "Yes" : "No") },
                      { key: "overlay", label: "Trust", render: (row) => fmtFloat(row.overlay?.trust_multiplier, 2) },
                    ]}
                    empty="No audit history yet."
                  />
                </Card>
              </div>
            ) : null}

            {tab === "setup" ? (
              <div className="stack">
                <Card title="How this runs" subtitle="Simple operating model">
                  <div className="metric-grid">
                    <Metric label="Dashboard" value="Always-on local server" />
                    <Metric label="Daily cycle" value="16:05 IST" />
                    <Metric label="Market logic" value="India holiday aware" />
                    <Metric label="Machine need" value="Mac awake or always-on host" />
                  </div>
                  <StatusNote tone="neutral">
                    If your laptop sleeps or shuts down, scheduled jobs stop. For true 24/7 operation, run this exact
                    setup on a Mac mini, home server, or VPS.
                  </StatusNote>
                </Card>

                <div className="split">
                  <Card title="macOS background install" subtitle="One-time install for launchd">
                    <CodeBlock>{`cd /Users/mananagarwal/Desktop/2nd\\ brain/plant\\ to\\ image/trader\nchmod +x deploy/bin/install_launchd.sh deploy/bin/uninstall_launchd.sh\n./deploy/bin/install_launchd.sh`}</CodeBlock>
                    <StatusNote tone="neutral">
                      This starts the local dashboard on `127.0.0.1:8050` and schedules the daily cycle.
                    </StatusNote>
                    <CodeBlock>{`launchctl list | rg 'com\\.trader\\.(dashboard|daily-cycle)'`}</CodeBlock>
                  </Card>

                  <Card title="Linux background install" subtitle="User-level systemd services">
                    <CodeBlock>{`mkdir -p ~/.config/systemd/user\ncp deploy/systemd/* ~/.config/systemd/user/\nsystemctl --user daemon-reload\nsystemctl --user enable --now trader-dashboard.service\nsystemctl --user enable --now trader-daily-cycle.timer`}</CodeBlock>
                    <StatusNote tone="neutral">
                      The Linux timer runs after market close; the dashboard service stays up continuously.
                    </StatusNote>
                  </Card>
                </div>

                <Card title="Useful places to look" subtitle="If you want to inspect what the system is doing">
                  <CodeBlock>{`UI:        http://127.0.0.1:8050/\nAPI:       http://127.0.0.1:8050/api\nWeights:   ./.venv/bin/python -m runtime.weights_on_date --model v9 --date 2026-04-02\nBackfill:  ./.venv/bin/python -m runtime.paper_backfill --start 2026-04-01 --end 2026-04-03 --initial-cash 1000000 --reset --apply\nLogs:      cache/logs/launchd_dashboard.out.log\n           cache/logs/launchd_daily_cycle.out.log`}</CodeBlock>
                </Card>
              </div>
            ) : null}
          </>
        ) : null}
      </main>
    </div>
  );
}
