import { t, initI18n, onLangChange, localeTag, localizeWarning, localizeSource } from "./i18n.js";

const DATA_URL = "data/predictions.json";
const BOARD_STORAGE_KEY = "ti2026_board_strategy";
const ANALYST_THRESHOLD = 5;

const BOARD_COLUMN_KEYS = [
  { key: "undefeated", titleKey: "col.undefeated", tone: "ok" },
  { key: "one_loss", titleKey: "col.one_loss", tone: "ok" },
  { key: "advance", titleKey: "col.advance", tone: "warn" },
  { key: "eliminate", titleKey: "col.eliminate", tone: "warn" },
  { key: "one_win", titleKey: "col.one_win", tone: "bad" },
  { key: "winless", titleKey: "col.winless", tone: "bad" },
];

const BOARD_ROWS = [
  ["undefeated", "one_loss", "advance"],
  ["eliminate", "one_win", "winless"],
];

/** Only fusion UI is exposed; model/analyst boards remain in data for presets. */
const MODE_GROUPS = [
  {
    id: "fusion",
    titleKey: "mode.group.fusion",
    descKey: "mode.group.fusion_desc",
    strategies: [
      "fusion",
      "fusion_model_heavy",
      "fusion_balanced",
      "fusion_market_lean",
      "fusion_analyst_lean",
    ],
  },
];

const LEGACY_STRATEGIES = new Set([
  "points_optimal",
  "qualify_rank",
  "analyst_consensus",
]);

const VALVE_POINTS_TABLE = [
  [0, 0],
  [1, 30],
  [2, 60],
  [3, 120],
  [4, 360],
  [5, 720],
  [6, 1200],
  [7, 1800],
  [8, 2520],
  [9, 3360],
  [10, 4320],
  [11, 5400],
  [12, 6600],
  [13, 7920],
  [14, 9360],
  [15, 10920],
  [16, 12000],
];

/** @type {any} */
let DATA = null;

/** Escape text for safe insertion into HTML. */
function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/** Local team emblem path (PNG preferred; SVG monogram as last resort). */
function teamLogoSrc(teamId) {
  const id = encodeURIComponent(String(teamId || "unknown"));
  return `assets/img/teams/${id}.png`;
}

/** Team emblem <img>; falls back to monogram SVG, then letter chip. */
function teamLogoHtml(teamId, shortName, { size = "" } = {}) {
  const id = String(teamId || shortName || "unknown");
  const mon = escapeHtml(String(shortName || teamId || "?").slice(0, 3).toUpperCase());
  const cls = size === "sm" ? "team-logo sm" : "team-logo";
  const png = `assets/img/teams/${encodeURIComponent(id)}.png`;
  const svg = `assets/img/teams/${encodeURIComponent(id)}.svg`;
  // Try PNG → SVG monogram → text chip
  return `<img class="${cls}" src="${png}" alt="" width="64" height="64" loading="lazy" data-monogram="${mon}" data-fallback-src="${svg}" onerror="if(this.dataset.fallbackSrc){const s=this.dataset.fallbackSrc;delete this.dataset.fallbackSrc;this.src=s;return;}this.onerror=null;this.replaceWith(Object.assign(document.createElement('span'),{className:this.className+' logo-fallback',textContent:this.dataset.monogram}))" />`;
}

async function loadData() {
  const res = await fetch(DATA_URL);
  if (!res.ok) throw new Error(t("error.fetch", { status: res.status }));
  return res.json();
}

function fmtPct(v) {
  return `${Number(v).toFixed(1)}%`;
}

function fmtNum(v) {
  return Number(v).toLocaleString(localeTag());
}

function valvePointsTooltip() {
  const rows = VALVE_POINTS_TABLE.map(([k, pts]) => `${k}/16 → ${pts.toLocaleString(localeTag())}`);
  return `${t("valve.points_title")}\n${rows.join("\n")}`;
}

function getBoardStrategy() {
  const sel = document.getElementById("board-strategy");
  let strategy = sel?.value || localStorage.getItem(BOARD_STORAGE_KEY) || "fusion";
  if (LEGACY_STRATEGIES.has(strategy) || !String(strategy).startsWith("fusion")) {
    strategy = "fusion";
  }
  return strategy;
}

function activeBoard(data) {
  const strategy = getBoardStrategy();
  if (data.boards && data.boards[strategy]) return data.boards[strategy];
  return data.board;
}

function fusionScenarioKey(strategy) {
  if (strategy === "fusion") return null;
  if (strategy && strategy.startsWith("fusion_")) {
    return strategy.replace(/^fusion_/, "");
  }
  return null;
}

function modeGroupForStrategy(strategy) {
  return MODE_GROUPS.find((g) => g.strategies.includes(strategy)) || MODE_GROUPS[0];
}

const FUSION_WEIGHT_KEYS = [
  "model_weight",
  "analyst_weight",
  "market_weight",
  "ranking_weight",
  "expert_weight",
];

const FUSION_WEIGHT_I18N = {
  model_weight: "fusion.weight.model",
  analyst_weight: "fusion.weight.analyst",
  market_weight: "fusion.weight.market",
  ranking_weight: "fusion.weight.ranking",
  expert_weight: "fusion.weight.expert",
};

const FUSION_PRESET_KEYS = [
  "fusion",
  "fusion_model_heavy",
  "fusion_balanced",
  "fusion_market_lean",
  "fusion_analyst_lean",
];

/** Effective shares after renormalizing raw soft weights (matches server fuse). */
function effectiveWeightShares(weights) {
  const raw = FUSION_WEIGHT_KEYS.map((k) => Math.max(0, Number(weights[k] || 0)));
  const total = raw.reduce((s, v) => s + v, 0);
  if (total <= 0) {
    return Object.fromEntries(FUSION_WEIGHT_KEYS.map((k) => [k, k === "model_weight" ? 1 : 0]));
  }
  return Object.fromEntries(FUSION_WEIGHT_KEYS.map((k, i) => [k, raw[i] / total]));
}

function scenarioWeightsMap(meta) {
  const scenarios = meta.fusion_weight_scenarios || {};
  const out = {};
  for (const [name, w] of Object.entries(scenarios)) {
    const model = Number(w.model_weight || 0);
    const market = Number(w.market_weight || 0);
    const ranking = Number(w.ranking_weight || 0);
    const expert = Number(w.expert_weight || 0);
    const analyst =
      w.analyst_weight != null
        ? Number(w.analyst_weight)
        : Math.max(0, 1 - model - market - ranking - expert);
    out[`fusion_${name}`] = {
      model_weight: model,
      analyst_weight: analyst,
      market_weight: market,
      ranking_weight: ranking,
      expert_weight: expert,
    };
  }
  const model = Number(meta.fusion_model_weight ?? 0.65);
  const market = Number(meta.fusion_market_weight ?? 0.1);
  const ranking = Number(meta.fusion_ranking_weight ?? 0.05);
  const analyst =
    meta.fusion_analyst_weight != null
      ? Number(meta.fusion_analyst_weight)
      : Math.max(0, 1 - model - market - ranking);
  out.fusion = {
    model_weight: model,
    analyst_weight: analyst,
    market_weight: market,
    ranking_weight: ranking,
    expert_weight: 0,
  };
  return out;
}

function weightsDistance(a, b) {
  return FUSION_WEIGHT_KEYS.reduce(
    (s, k) => s + Math.abs(Number(a[k] || 0) - Number(b[k] || 0)),
    0
  );
}

/** Pick precomputed fusion_* board closest to draft weights. */
function nearestFusionStrategy(meta, draftWeights) {
  const map = scenarioWeightsMap(meta);
  let best = "fusion_balanced";
  let bestDist = Infinity;
  for (const [strategy, w] of Object.entries(map)) {
    if (strategy === "fusion") continue;
    const d = weightsDistance(draftWeights, w);
    if (d < bestDist) {
      bestDist = d;
      best = strategy;
    }
  }
  if (weightsDistance(draftWeights, map.fusion) <= bestDist + 0.02) {
    return "fusion";
  }
  return best;
}

function expectedPointsForStrategy(data, strategy) {
  const meta = data.meta || {};
  const compare = meta.board_compare || {};
  if (compare[strategy]?.expected_points != null) {
    return Number(compare[strategy].expected_points);
  }
  if (strategy === "analyst_consensus" && data.analyst?.expected_points != null) {
    return Number(data.analyst.expected_points);
  }
  const key = fusionScenarioKey(strategy);
  const scores = meta.fusion_scenario_scores || {};
  if (key && scores[key]?.expected_points != null) {
    return Number(scores[key].expected_points);
  }
  if (strategy.startsWith("fusion") && meta.fusion_expected_points != null) {
    return Number(meta.fusion_expected_points);
  }
  return meta.expected_compendium_points != null
    ? Number(meta.expected_compendium_points)
    : null;
}

function setBoardStrategy(strategy, data, { persist = true } = {}) {
  const sel = document.getElementById("board-strategy");
  if (sel && sel.querySelector(`option[value="${strategy}"]`)) {
    sel.value = strategy;
  }
  if (persist) localStorage.setItem(BOARD_STORAGE_KEY, strategy);
  renderHero(data);
  renderModePanel(data);
  renderFusionWeights(data);
  renderBoard(data);
}

function renderModePanel(data) {
  const groupsEl = document.getElementById("mode-groups");
  const variantsEl = document.getElementById("mode-variants");
  const summaryEl = document.getElementById("mode-summary");
  // Source tiles removed — only fusion mix UI remains.
  if (groupsEl) {
    groupsEl.hidden = true;
    groupsEl.innerHTML = "";
  }
  if (variantsEl) {
    variantsEl.hidden = true;
    variantsEl.innerHTML = "";
  }

  const strategy = getBoardStrategy();
  if (!String(strategy).startsWith("fusion")) {
    setBoardStrategy("fusion", data);
    return;
  }
  const pts = expectedPointsForStrategy(data, strategy);

  if (summaryEl) {
    const ptsText =
      pts != null
        ? ` · ${t("mode.points_approx")}: <strong>${Math.round(pts).toLocaleString(localeTag())}</strong>`
        : "";
    summaryEl.innerHTML = `${escapeHtml(t("mode.current"))}: <strong>${escapeHtml(t(`strategy.${strategy}`))}</strong>${ptsText}`;
  }
}

function renderFusionWeights(data) {
  const root = document.getElementById("fusion-weights");
  const sliders = document.getElementById("fusion-sliders");
  const presets = document.getElementById("fusion-presets");
  const scoreEl = document.getElementById("fusion-score");
  const disc = document.getElementById("market-disclaimer");
  if (!root || !sliders) return;
  const meta = data.meta || {};
  const strategy = getBoardStrategy();
  const isFusion = strategy === "fusion" || strategy.startsWith("fusion_");
  root.hidden = !isFusion;
  if (disc) {
    disc.textContent = t("fusion.market_fallback");
  }
  if (!isFusion) {
    sliders.innerHTML = "";
    if (presets) presets.innerHTML = "";
    if (scoreEl) scoreEl.textContent = "";
    return;
  }

  const map = scenarioWeightsMap(meta);
  const weights = map[strategy] || map.fusion;
  const shares = effectiveWeightShares(weights);
  const rawSum = FUSION_WEIGHT_KEYS.reduce((s, k) => s + Math.max(0, Number(weights[k] || 0)), 0);

  if (presets) {
    presets.innerHTML = FUSION_PRESET_KEYS.filter((k) => map[k] || k === "fusion")
      .map((k) => {
        const active = k === strategy ? " active" : "";
        const hintKey = `strategy.${k}_hint`;
        const hint = t(hintKey);
        const titleAttr =
          hint && !hint.startsWith("strategy.")
            ? ` title="${escapeHtml(hint)}"`
            : "";
        // Same human labels as strategy.* (single mix chooser lives here)
        return `<button type="button" class="fusion-preset${active}" data-strategy="${k}"${titleAttr}>${escapeHtml(t(`strategy.${k}`))}</button>`;
      })
      .join("");
    presets.querySelectorAll("[data-strategy]").forEach((btn) => {
      btn.addEventListener("click", () => {
        setBoardStrategy(btn.getAttribute("data-strategy"), data);
      });
    });
  }

  sliders.innerHTML =
    FUSION_WEIGHT_KEYS.map((k) => {
      const v = Math.round(Number(weights[k] ?? 0) * 100);
      const eff = Math.round(Number(shares[k] || 0) * 100);
      return `
      <label class="fusion-slider-row">
        <span class="fusion-slider-head">
          <span>${escapeHtml(t(FUSION_WEIGHT_I18N[k]))}: <strong data-wlabel="${k}">${v}</strong></span>
          <span class="fusion-slider-eff" data-weff="${k}">${escapeHtml(t("fusion.weight_in_mix"))}: ${eff}%</span>
        </span>
        <input type="range" min="0" max="100" step="5" value="${v}" data-weight="${k}" aria-valuetext="${v}" />
      </label>`;
    }).join("") +
    `<p class="fusion-weights-help" style="margin:0.35rem 0 0;grid-column:1/-1">${escapeHtml(
      t("fusion.weights_raw_sum", { sum: Math.round(rawSum * 100) })
    )}</p>`;

  const refreshEff = (draft) => {
    const nextShares = effectiveWeightShares(draft);
    const nextSum = FUSION_WEIGHT_KEYS.reduce(
      (s, k) => s + Math.max(0, Number(draft[k] || 0)),
      0
    );
    sliders.querySelectorAll("[data-weff]").forEach((el) => {
      const key = el.getAttribute("data-weff");
      const eff = Math.round(Number(nextShares[key] || 0) * 100);
      el.textContent = `${t("fusion.weight_in_mix")}: ${eff}%`;
    });
    const sumEl = sliders.querySelector(".fusion-weights-help");
    if (sumEl) {
      sumEl.textContent = t("fusion.weights_raw_sum", { sum: Math.round(nextSum * 100) });
    }
  };

  sliders.querySelectorAll("input[data-weight]").forEach((input) => {
    input.addEventListener("input", () => {
      const draft = { ...weights };
      sliders.querySelectorAll("input[data-weight]").forEach((el) => {
        draft[el.getAttribute("data-weight")] = Number(el.value) / 100;
        const lab = sliders.querySelector(`[data-wlabel="${el.getAttribute("data-weight")}"]`);
        if (lab) lab.textContent = `${el.value}`;
      });
      refreshEff(draft);
      const next = nearestFusionStrategy(meta, draft);
      if (next !== strategy) setBoardStrategy(next, data);
    });
  });

  if (scoreEl) {
    const key = fusionScenarioKey(strategy);
    const scores = meta.fusion_scenario_scores || {};
    const pts =
      (key && scores[key]?.expected_points) ?? meta.fusion_expected_points ?? null;
    const correct = (key && scores[key]?.expected_correct) ?? null;
    scoreEl.innerHTML =
      pts != null
        ? `${escapeHtml(t("fusion.score"))}: <strong>${Math.round(pts).toLocaleString(localeTag())}</strong>` +
          (correct != null
            ? ` · ${escapeHtml(t("fusion.correct_slots"))} ${Number(correct).toFixed(2)}`
            : "")
        : "";
  }
}

function renderFallbackBanner(data) {
  const el = document.getElementById("fallback-banner");
  if (!el) return;
  const meta = data.meta || {};
  const isFallback =
    meta.is_power_ranking_fallback === true ||
    String(meta.model || "").includes("power_ranking");
  if (!isFallback) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.textContent = t("fallback.banner");
}

function renderHero(data) {
  const meta = data.meta;
  document.getElementById("disclaimer-text").textContent = t("disclaimer.body");
  renderFallbackBanner(data);
  const warnEl = document.getElementById("meta-warnings");
  if (warnEl) {
    const warnings = Array.isArray(meta.warnings) ? meta.warnings : [];
    const localized = warnings.map(localizeWarning).filter(Boolean);
    if (localized.length) {
      warnEl.hidden = false;
      warnEl.innerHTML = localized.map((w) => `<li>${escapeHtml(w)}</li>`).join("");
    } else {
      warnEl.hidden = true;
      warnEl.innerHTML = "";
    }
  }
  const strategy = getBoardStrategy();
  const ptsVal = expectedPointsForStrategy(data, strategy);
  const pts =
    ptsVal != null
      ? `<span class="chip" title="${escapeHtml(valvePointsTooltip())}">${escapeHtml(t("chip.points"))}: <strong>${Math.round(ptsVal).toLocaleString(localeTag())}</strong></span>`
      : "";
  const modelPts = Math.round(Number(ptsVal));
  const analystPts =
    data.analyst?.expected_points != null
      ? Math.round(Number(data.analyst.expected_points))
      : null;
  const fusionPts =
    meta.fusion_expected_points != null
      ? Math.round(Number(meta.fusion_expected_points))
      : null;
  let extraPts = "";
  if (analystPts != null && analystPts !== modelPts) {
    extraPts += `<span class="chip" title="${escapeHtml(t("title.analyst_pts"))}">${escapeHtml(t("chip.analysts"))}: <strong>${analystPts.toLocaleString(localeTag())}</strong></span>`;
  }
  if (
    fusionPts != null &&
    fusionPts !== modelPts &&
    fusionPts !== analystPts
  ) {
    extraPts += `<span class="chip" title="${escapeHtml(t("title.fusion_pts"))}">${escapeHtml(t("chip.fusion"))}: <strong>${fusionPts.toLocaleString(localeTag())}</strong></span>`;
  } else if (fusionPts != null && fusionPts !== modelPts && analystPts == null) {
    extraPts += `<span class="chip" title="${escapeHtml(t("title.fusion_pts"))}">${escapeHtml(t("chip.fusion"))}: <strong>${fusionPts.toLocaleString(localeTag())}</strong></span>`;
  } else if (
    fusionPts != null &&
    analystPts != null &&
    fusionPts === analystPts &&
    fusionPts !== modelPts &&
    !extraPts
  ) {
    extraPts += `<span class="chip" title="${escapeHtml(t("title.both_pts"))}">${escapeHtml(t("chip.analysts_fusion"))}: <strong>${fusionPts.toLocaleString(localeTag())}</strong></span>`;
  }
  const modelChipLabel = friendlyModelLabel(meta);
  document.getElementById("hero-meta").innerHTML = `
    <span class="chip"><strong>${escapeHtml(modelChipLabel)}</strong></span>
    <span class="chip">${escapeHtml(t("chip.sims"))}: <strong>${fmtNum(meta.n_simulations)}</strong></span>
    ${pts}
    ${extraPts}
    <span class="chip">${escapeHtml(friendlyFormatLabel(meta.format))}</span>
    <span class="chip">v${escapeHtml(meta.version)}</span>
  `;
  document.getElementById("footer-stamp").textContent =
    `${t("footer.updated")}: ${new Date(meta.generated_at).toLocaleString(localeTag())}`;
}

function friendlyModelLabel(meta) {
  const key = String(meta.model || "").toLowerCase();
  if (key.includes("blend")) return t("chip.model_blend");
  if (key.includes("power")) return t("chip.model_power");
  if (key.includes("catboost")) return t("chip.model_catboost");
  if (key.includes("xgb")) return t("chip.model_xgb");
  return t("chip.model_default");
}

function friendlyFormatLabel(fmt) {
  if (!fmt) return t("format.swiss_short");
  if (/16-team Swiss/i.test(fmt) || /Swiss to 4/i.test(fmt)) {
    return t("format.swiss");
  }
  return fmt;
}

function renderTeamMark(tRow, nAnalysts) {
  const agree = tRow.analyst_agreement ?? 0;
  const names = (tRow.analyst_names || []).join(", ");
  const denom = nAnalysts != null ? String(nAnalysts) : "?";
  const chip =
    agree >= ANALYST_THRESHOLD
      ? `<span class="chip mini" title="${escapeHtml(names)}">${agree}/${denom}</span>`
      : "";
  const slotPct = tRow.slot_pct ?? 0;
  const displayName = escapeHtml(tRow.short || tRow.name);
  const title = escapeHtml(`${tRow.name || ""}${names ? " · " + names : ""}`);
  const teamId = tRow.id || tRow.short || tRow.name;
  return `
    <article class="team-mark" title="${title}">
      ${teamLogoHtml(teamId, tRow.short || tRow.name)}
      <div class="name">${displayName}${chip}</div>
      <div class="meta">${escapeHtml(tRow.record)} · ${fmtPct(slotPct)}</div>
      <div class="prob">${fmtPct(tRow.qualify_pct)}</div>
    </article>`;
}

function renderBoard(data) {
  const root = document.getElementById("swiss-board");
  const board = activeBoard(data);
  const nAnalysts = data.analyst?.n_analysts;
  const colMap = Object.fromEntries(BOARD_COLUMN_KEYS.map((c) => [c.key, c]));

  root.innerHTML = BOARD_ROWS.map((rowKeys, rowIdx) => {
    const rowClass = rowIdx === 0 ? "swiss-row advance-row" : "swiss-row elim-row";
    const cols = rowKeys
      .map((key) => {
        const col = colMap[key];
        const teams = board[col.key] || [];
        const marks = teams.length
          ? `<div class="team-slots">${teams.map((tRow) => renderTeamMark(tRow, nAnalysts)).join("")}</div>`
          : `<p class="meta" style="color:var(--muted);margin:0;font-size:0.85rem;text-align:center">${escapeHtml(t("board.empty"))}</p>`;
        return `
      <div class="board-col ${col.tone}">
        <h4>${escapeHtml(t(col.titleKey))}</h4>
        ${marks}
      </div>`;
      })
      .join("");
    return `<div class="${rowClass}">${cols}</div>`;
  }).join("");
}

function sortTeams(teams, mode) {
  const copy = [...teams];
  if (mode === "power") copy.sort((a, b) => a.power_rank - b.power_rank);
  else if (mode === "elim") copy.sort((a, b) => b.eliminated_pct - a.eliminated_pct);
  else if (mode === "alpha") copy.sort((a, b) => a.name.localeCompare(b.name, localeTag()));
  else copy.sort((a, b) => b.qualify_pct - a.qualify_pct);
  return copy;
}

function renderStandings(data) {
  const mode = document.getElementById("sort-by").value;
  const q = document.getElementById("search").value.trim().toLowerCase();
  let teams = sortTeams(data.teams, mode);
  if (q) {
    teams = teams.filter(
      (row) =>
        row.name.toLowerCase().includes(q) ||
        row.short.toLowerCase().includes(q) ||
        row.region.toLowerCase().includes(q)
    );
  }
  const tbody = document.querySelector("#standings-table tbody");
  tbody.innerHTML = teams
    .map((row, i) => {
      const strength =
        row.strength_label ||
        (row.strength_mu != null
          ? `${Math.round(row.strength_mu)} ± ${Math.round(row.strength_sigma || 0)}`
          : "—");
      const home =
        row.home_lan_elo > 0 ? ` · ${t("rank.home")}${row.home_lan_elo}` : "";
      return `
      <tr>
        <td>${i + 1}</td>
        <td class="team-cell">
          <div class="team-cell-with-logo">
            ${teamLogoHtml(row.id || row.short, row.short || row.name, { size: "sm" })}
            <div>
              ${escapeHtml(row.name)}
              <span class="sub">${escapeHtml(localizeSource(row.source))} · ${escapeHtml(t("standings.power_rank"))} ${escapeHtml(row.power_rank)}${escapeHtml(home)}</span>
            </div>
          </div>
        </td>
        <td>${escapeHtml(row.region)}</td>
        <td class="strength-cell" title="${escapeHtml(t("title.strength"))}">${escapeHtml(strength)}</td>
        <td>
          <div class="bar">
            <span>${fmtPct(row.qualify_pct)}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${Number(row.qualify_pct) || 0}%"></div></div>
          </div>
        </td>
        <td>
          <div class="bar">
            <span>${fmtPct(row.eliminated_pct)}</span>
            <div class="bar-track"><div class="bar-fill danger" style="width:${Number(row.eliminated_pct) || 0}%"></div></div>
          </div>
        </td>
        <td>${Number(row.expected_wins).toFixed(2)}</td>
        <td>${escapeHtml(row.most_likely_record)}</td>
      </tr>`;
    })
    .join("");
}

function heatColor(pct) {
  const v = Math.max(0, Math.min(100, Number(pct) || 0));
  const alpha = 0.08 + (v / 100) * 0.55;
  return `rgba(197, 160, 89, ${alpha.toFixed(3)})`;
}

function renderHeatmap(data) {
  const root = document.getElementById("slot-heatmap");
  if (!root) return;
  const hm = data.slot_heatmap;
  if (!hm || !hm.matrix) {
    root.innerHTML = `<tbody><tr><td>${escapeHtml(t("heatmap.empty"))}</td></tr></tbody>`;
    return;
  }
  const head = `<thead><tr><th>${escapeHtml(t("heatmap.team"))}</th>${hm.slots
    .map((s) => `<th>${escapeHtml(s)}</th>`)
    .join("")}</tr></thead>`;
  const body = hm.matrix
    .map((row, i) => {
      const team = hm.teams[i];
      const cells = row
        .map(
          (v) =>
            `<td style="background:${heatColor(v)}" title="${escapeHtml(team.name)} · ${Number(v)}%">${Number(v).toFixed(1)}</td>`
        )
        .join("");
      const logo = teamLogoHtml(team.id || team.short, team.short || team.name, {
        size: "sm",
      });
      return `<tr><th scope="row"><span class="team-cell-with-logo">${logo}<span>${escapeHtml(team.short || team.name)}</span></span></th>${cells}</tr>`;
    })
    .join("");
  root.innerHTML = `${head}<tbody>${body}</tbody>`;
}

function fillMatchupSelects(data) {
  const opts = data.teams
    .slice()
    .sort((a, b) => a.name.localeCompare(b.name, localeTag()))
    .map((row) => `<option value="${escapeHtml(row.id)}">${escapeHtml(row.name)}</option>`)
    .join("");
  const a = document.getElementById("team-a");
  const b = document.getElementById("team-b");
  const prevA = a.value;
  const prevB = b.value;
  a.innerHTML = opts;
  b.innerHTML = opts;
  a.value = prevA || data.teams[0]?.id || "";
  b.value = prevB || data.teams[1]?.id || data.teams[0]?.id || "";
}

function findMatchup(data, idA, idB) {
  if (idA === idB) return { p_a: 0.5, p_b: 0.5 };
  const direct = data.matchups.find((m) => m.a === idA && m.b === idB);
  if (direct) return direct;
  const rev = data.matchups.find((m) => m.a === idB && m.b === idA);
  if (rev) return { p_a: rev.p_b, p_b: rev.p_a };
  return { p_a: 0.5, p_b: 0.5 };
}

function renderMatchup(data) {
  const idA = document.getElementById("team-a").value;
  const idB = document.getElementById("team-b").value;
  const teamA = data.teams.find((row) => row.id === idA);
  const teamB = data.teams.find((row) => row.id === idB);
  const m = findMatchup(data, idA, idB);
  const pa = Math.round(m.p_a * 1000) / 10;
  const pb = Math.round(m.p_b * 1000) / 10;
  document.getElementById("matchup-result").innerHTML = `
    <div class="duel">
      <div class="duel-side">
        ${teamLogoHtml(teamA?.id || idA, teamA?.short || teamA?.name)}
        <div class="name">${escapeHtml(teamA?.name || idA)}</div>
        <div class="pct" style="color:var(--gold)">${pa}%</div>
      </div>
      <div class="meta">${escapeHtml(t("matchup.series_win"))}</div>
      <div class="duel-side">
        ${teamLogoHtml(teamB?.id || idB, teamB?.short || teamB?.name)}
        <div class="name">${escapeHtml(teamB?.name || idB)}</div>
        <div class="pct" style="color:var(--cta-hi)">${pb}%</div>
      </div>
    </div>
    <div class="duel-bar" aria-hidden="true">
      <span style="width:${pa}%"></span>
      <span style="width:${pb}%"></span>
    </div>`;
}

function renderMetrics(data) {
  const m = data.model_metrics || {};
  const meta = data.meta || {};
  const meth = document.getElementById("methodology-text");
  if (meth) {
    meth.textContent = t("model.methodology");
  }
  const cov = m.player_coverage;
  const covLabel =
    cov && cov.coverage != null
      ? `${(Number(cov.coverage) * 100).toFixed(1).replace(/\.0$/, "")}%`
      : "—";
  const missing =
    cov && cov.n_matches != null && cov.n_matches_with_players != null
      ? Number(cov.n_matches) - Number(cov.n_matches_with_players)
      : null;
  const blendAuc = m.blend_leave_one_ti_avg_auc;
  const compare = data.meta?.board_compare || {};
  const covHint = cov
    ? [
        `${cov.n_matches_with_players || 0}/${cov.n_matches || "?"} ${t("metric.coverage_with")}`,
        t("metric.coverage_ok"),
        missing != null && missing > 0
          ? t("metric.coverage_left", { n: missing })
          : null,
      ]
        .filter(Boolean)
        .join(" · ")
    : t("metric.coverage_hint_base");
  const covTitle = t("metric.coverage_title");
  const cards = [
    {
      label: t("metric.loo"),
      value:
        m.leave_one_ti_recent_avg_auc != null
          ? Number(m.leave_one_ti_recent_avg_auc).toFixed(3)
          : blendAuc != null
            ? Number(blendAuc).toFixed(3)
            : m.leave_one_ti_avg_auc != null
              ? Number(m.leave_one_ti_avg_auc).toFixed(3)
              : "—",
      hint: t("metric.loo_hint"),
    },
    {
      label: t("metric.wf"),
      value:
        m.walk_forward_avg_auc != null
          ? Number(m.walk_forward_avg_auc).toFixed(3)
          : "—",
      hint: t("metric.wf_hint"),
    },
    {
      label: t("metric.corpus"),
      value: meta.n_maps != null ? `${meta.n_maps}` : (cov?.n_matches ?? "—"),
      hint:
        meta.n_leagues != null
          ? `${meta.n_leagues} ${t("metric.corpus_hint")}`
          : t("metric.corpus_hint_fallback"),
    },
    {
      label: t("metric.points"),
      value: compare.points_optimal?.expected_points?.toFixed(0) ?? "—",
      hint: t("metric.points_hint"),
    },
    {
      label: t("metric.coverage"),
      value: covLabel,
      hint: covHint,
      title: covTitle,
      note:
        missing != null && missing > 0
          ? t("metric.coverage_note", { n: missing })
          : null,
    },
  ];
  document.getElementById("metrics-grid").innerHTML = cards
    .map(
      (c) => `
    <article class="metric"${c.title ? ` title="${escapeHtml(c.title)}"` : ""}>
      <p class="label">${escapeHtml(c.label)}</p>
      <p class="value">${escapeHtml(String(c.value))}</p>
      <p class="hint">${escapeHtml(c.hint)}</p>
      ${c.note ? `<p class="note">${escapeHtml(c.note)}</p>` : ""}
    </article>`
    )
    .join("");
}

function bindNavSpy() {
  const links = [...document.querySelectorAll(".side-nav a[href^='#']")];
  const sections = links
    .map((a) => document.querySelector(a.getAttribute("href")))
    .filter(Boolean);
  if (!sections.length || !("IntersectionObserver" in window)) return;

  const io = new IntersectionObserver(
    (entries) => {
      const visible = entries
        .filter((e) => e.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      const id = `#${visible.target.id}`;
      links.forEach((a) => {
        a.classList.toggle("is-active", a.getAttribute("href") === id);
      });
    },
    { rootMargin: "-20% 0px -55% 0px", threshold: [0.1, 0.35, 0.6] }
  );
  sections.forEach((sec) => io.observe(sec));
}

function rerenderAll(data) {
  if (!data) return;
  renderHero(data);
  renderModePanel(data);
  renderBoard(data);
  renderFusionWeights(data);
  renderStandings(data);
  renderHeatmap(data);
  renderMatchup(data);
  renderMetrics(data);
}

function bind(data) {
  document.getElementById("sort-by").addEventListener("change", () => renderStandings(data));
  document.getElementById("search").addEventListener("input", () => renderStandings(data));
  document.getElementById("team-a").addEventListener("change", () => renderMatchup(data));
  document.getElementById("team-b").addEventListener("change", () => renderMatchup(data));
  const boardSel = document.getElementById("board-strategy");
  if (boardSel) {
    const saved = localStorage.getItem(BOARD_STORAGE_KEY);
    if (saved && boardSel.querySelector(`option[value="${saved}"]`)) {
      boardSel.value = saved;
    }
    boardSel.addEventListener("change", () => {
      localStorage.setItem(BOARD_STORAGE_KEY, boardSel.value);
      renderHero(data);
      renderModePanel(data);
      renderFusionWeights(data);
      renderBoard(data);
    });
  }
  bindNavSpy();
  onLangChange(() => rerenderAll(data));
}

async function main() {
  initI18n();
  try {
    DATA = await loadData();
    fillMatchupSelects(DATA);
    const boardSel = document.getElementById("board-strategy");
    const saved = localStorage.getItem(BOARD_STORAGE_KEY);
    if (boardSel && saved && boardSel.querySelector(`option[value="${saved}"]`)) {
      boardSel.value = saved;
    }
    renderHero(DATA);
    renderModePanel(DATA);
    renderBoard(DATA);
    renderFusionWeights(DATA);
    renderStandings(DATA);
    renderHeatmap(DATA);
    renderMatchup(DATA);
    renderMetrics(DATA);
    bind(DATA);
  } catch (err) {
    document.getElementById("disclaimer-text").textContent = t("error.load", {
      msg: err.message,
    });
    console.error(err);
  }
}

main();
