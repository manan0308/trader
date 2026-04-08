import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const dashboardDir = path.resolve(__dirname, "..");
const repoRoot = path.resolve(dashboardDir, "..");
const cacheDir = path.join(repoRoot, "cache");
const publicDataDir = path.join(dashboardDir, "public", "data");

function readJsonIfExists(filePath, fallback = null) {
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8"));
  } catch {
    return fallback;
  }
}

function overlayHasEffect(payload) {
  if (!payload || typeof payload !== "object") return false;
  const defaultRisk = Number(payload.default_risk_off_override || 0);
  const dates = Array.isArray(payload.dates) ? payload.dates : [];
  return defaultRisk > 0 || dates.length > 0;
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function toRows(obj) {
  return Object.entries(obj || {})
    .filter(([key, value]) => value && typeof value === "object")
    .map(([key, value]) => ({ name: key, ...value }));
}

function buildSignalRows(liveSignal) {
  const models = liveSignal?.models || {};
  return Object.values(models).flatMap((model) =>
    (model.rows || []).map((row) => ({
      model: model.model,
      label: model.label,
      ...row,
    })),
  );
}

const alphaV9 = readJsonIfExists(path.join(cacheDir, "alpha_v9_results.json"), {});
const validation = readJsonIfExists(path.join(cacheDir, "validation_pack.json"), {});
const significance = readJsonIfExists(path.join(cacheDir, "performance_significance.json"), {});
const executionPlan = readJsonIfExists(path.join(cacheDir, "execution_plan_latest.json"), {});
const executionPlanBase = readJsonIfExists(path.join(cacheDir, "execution_plan_base_latest.json"), {});
const overlayRaw = readJsonIfExists(path.join(cacheDir, "anthropic_overlay_latest.json"), {});
const overlayActive = readJsonIfExists(path.join(cacheDir, "active_overlay_latest.json"), {});
const liveSignal = readJsonIfExists(path.join(cacheDir, "live_signal_latest.json"), {});
const paperTrading = readJsonIfExists(path.join(cacheDir, "paper_trading_latest.json"), {});
const paperBase = readJsonIfExists(path.join(cacheDir, "paper_base_latest.json"), {});
const paperComparison = readJsonIfExists(path.join(cacheDir, "paper_comparison_latest.json"), {});
const paperBenchmark = readJsonIfExists(path.join(cacheDir, "paper_benchmark_latest.json"), {});
const paperBackfill = readJsonIfExists(path.join(cacheDir, "paper_backfill_latest.json"), {});
const auditRuns = readJsonIfExists(path.join(cacheDir, "audit_runs_latest.json"), []);
const learningState = readJsonIfExists(path.join(cacheDir, "learning_state.json"), {});
const dailyCycle = readJsonIfExists(path.join(cacheDir, "daily_cycle_latest.json"), {});
const growwAuth = readJsonIfExists(path.join(cacheDir, "groww_auth_status.json"), {});
const executionSubmissions = readJsonIfExists(path.join(cacheDir, "execution_submissions_latest.json"), []);
const reconciliation = readJsonIfExists(path.join(cacheDir, "reconciliation_latest.json"), {});
const executionConfirmation = readJsonIfExists(path.join(cacheDir, "execution_confirmation_latest.json"), {});
const fullSample = alphaV9.full_sample || {};
const stitched = alphaV9.wfo?.stitched || {};
const latestAudit = Array.isArray(auditRuns) && auditRuns.length ? auditRuns[auditRuns.length - 1] : {};
const effectiveOverlay = overlayHasEffect(overlayActive) ? overlayActive : overlayRaw;
const overlayDates = Array.isArray(effectiveOverlay?.dates) ? effectiveOverlay.dates : [];
const latestOverlay = overlayDates.length ? overlayDates[overlayDates.length - 1] : null;

const benchmarkRows = [
  fullSample["weekly_core85_tilt15"]
    ? { name: "v9 Full Sample", source: "Full sample", ...fullSample["weekly_core85_tilt15"] }
    : null,
  Object.keys(stitched).length
    ? { name: "v9 Stitched OOS", source: "Walk-forward", ...stitched, turnover: null, avg_cash: null }
    : null,
  fullSample["EqWt Risky"]
    ? { name: "EqWt Risky", source: "Benchmark", ...fullSample["EqWt Risky"] }
    : null,
  fullSample["Nifty B&H"]
    ? { name: "Nifty B&H", source: "Benchmark", ...fullSample["Nifty B&H"] }
    : null,
  fullSample["60/40 Nifty/Cash"]
    ? { name: "60/40 Nifty/Cash", source: "Benchmark", ...fullSample["60/40 Nifty/Cash"] }
    : null,
].filter(Boolean);

const dashboard = {
  generated_at: new Date().toISOString(),
  as_of: dailyCycle.as_of || liveSignal.as_of || executionPlan.as_of || null,
  model_allocation: latestAudit?.v9_final_weights || executionPlan.plan?.current_weights || {},
  execution_target_allocation: executionPlan.plan?.target_weights || {},
  current_allocation: latestAudit?.v9_final_weights || executionPlan.plan?.target_weights || executionPlan.plan?.current_weights || {},
  execution_plan: executionPlan.plan || {},
  execution_plan_base: executionPlanBase.plan || {},
  live_signal: liveSignal,
  signal_rows: buildSignalRows(liveSignal),
  benchmarks: {
    full_sample: fullSample,
    walk_forward: stitched,
  },
  benchmark_rows: benchmarkRows,
  validation,
  significance,
  overlay: effectiveOverlay,
  overlay_raw: overlayRaw,
  overlay_active: overlayActive,
  overlay_latest: latestOverlay,
  paper_trading: paperTrading,
  paper_base: paperBase,
  paper_comparison: paperComparison,
  paper_benchmark: paperBenchmark,
  paper_backfill: paperBackfill,
  audit_runs: auditRuns,
  learning_state: learningState,
  daily_cycle: dailyCycle,
  groww_auth: growwAuth,
  execution_submissions: executionSubmissions,
  reconciliation,
  execution_confirmation: executionConfirmation,
  sections: {
    full_sample_rows: toRows(fullSample),
  },
};

ensureDir(publicDataDir);
fs.writeFileSync(path.join(publicDataDir, "dashboard.json"), JSON.stringify(dashboard, null, 2));
console.log(`Wrote ${path.join(publicDataDir, "dashboard.json")}`);
