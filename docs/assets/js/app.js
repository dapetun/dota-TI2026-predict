import { t, initI18n, onLangChange, localeTag, localizeWarning, localizeSource } from "./i18n.js";

const DATA_URL = "data/predictions.json";
const BOARD_STORAGE_KEY = "ti2026_board_strategy";
const CUSTOM_WEIGHTS_KEY = "ti2026_custom_weights";
const ANALYST_THRESHOLD = 5;

const SLOT_KEYS = [
  "undefeated",
  "one_loss",
  "advance",
  "eliminate",
  "one_win",
  "winless",
];

const SLOT_CAPS = {
  undefeated: 1,
  one_loss: 2,
  advance: 5,
  eliminate: 5,
  one_win: 2,
  winless: 1,
};

const SLOT_RECORD = {
  undefeated: "4–0",
  one_loss: "4–1",
  advance: "Проход",
  eliminate: "Выбывание",
  one_win: "1–4",
  winless: "0–4",
};

const SLOT_TO_PROB = {
  undefeated: "prob_4_0",
  one_loss: "prob_4_1",
  advance: "prob_advance",
  eliminate: "prob_eliminate",
  one_win: "prob_1_4",
  winless: "prob_0_4",
};

const SOURCE_TO_WEIGHT = {
  model: "model_weight",
  analyst: "analyst_weight",
  market: "market_weight",
  ranking: "ranking_weight",
  expert: "expert_weight",
};

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
      "fusion_custom",
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
  if (strategy === "fusion_custom") {
    ensureCustomBoard(data);
  }
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
  "fusion_custom",
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

function defaultCustomWeights(meta) {
  return {
    model_weight: Number(meta?.fusion_model_weight ?? 0.65),
    analyst_weight: Number(meta?.fusion_analyst_weight ?? 0.2),
    market_weight: Number(meta?.fusion_market_weight ?? 0.1),
    ranking_weight: Number(meta?.fusion_ranking_weight ?? 0.05),
    expert_weight: 0,
  };
}

function loadCustomWeights(meta) {
  try {
    const raw = localStorage.getItem(CUSTOM_WEIGHTS_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      const out = defaultCustomWeights(meta);
      for (const k of FUSION_WEIGHT_KEYS) {
        if (parsed[k] != null) out[k] = Math.max(0, Math.min(1, Number(parsed[k]) || 0));
      }
      return out;
    }
  } catch {
    /* ignore */
  }
  return defaultCustomWeights(meta);
}

function saveCustomWeights(weights) {
  localStorage.setItem(CUSTOM_WEIGHTS_KEY, JSON.stringify(weights));
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
  out.fusion_custom = loadCustomWeights(meta);
  return out;
}

/** Blend slot sources client-side (same renormalize as server fuse). */
function fuseTeamSlotProbs(teamId, shares, slotSources) {
  const src = slotSources?.[teamId];
  if (!src) return null;
  const out = {};
  for (const slot of SLOT_KEYS) {
    let p = 0;
    for (const [source, weightKey] of Object.entries(SOURCE_TO_WEIGHT)) {
      const w = Number(shares[weightKey] || 0);
      if (w <= 0) continue;
      p += w * Number(src[source]?.[slot] ?? 0);
    }
    out[slot] = p;
  }
  const sum = SLOT_KEYS.reduce((s, k) => s + out[k], 0);
  if (sum > 0) {
    for (const slot of SLOT_KEYS) out[slot] /= sum;
  }
  return out;
}

function slotProbFromTeam(teamLike, slot) {
  if (teamLike?.slots && teamLike.slots[slot] != null) return Number(teamLike.slots[slot]);
  const key = SLOT_TO_PROB[slot];
  const raw = Number(teamLike?.[key] ?? 0);
  return raw > 1 ? raw / 100 : raw;
}

function expectedCorrectCount(assignment, fusedTeams) {
  const byId = Object.fromEntries(fusedTeams.map((t) => [t.id, t]));
  return assignment.reduce((s, { id, slot }) => s + slotProbFromTeam(byId[id], slot), 0);
}

function expectedValvePointsFromCorrectDist(assignment, fusedTeams) {
  let dist = { 0: 1 };
  for (const { id, slot } of assignment) {
    const team = fusedTeams.find((t) => t.id === id);
    const pHit = Math.max(0, Math.min(1, slotProbFromTeam(team, slot)));
    const next = {};
    for (const [kStr, pr] of Object.entries(dist)) {
      const k = Number(kStr);
      next[k] = (next[k] || 0) + pr * (1 - pHit);
      next[k + 1] = (next[k + 1] || 0) + pr * pHit;
    }
    dist = next;
  }
  let pts = 0;
  for (const [kStr, pr] of Object.entries(dist)) {
    const k = Number(kStr);
    const row = VALVE_POINTS_TABLE.find(([n]) => n === k);
    pts += pr * (row ? row[1] : 0);
  }
  return pts;
}

function greedyAssignment(fusedTeams) {
  const remaining = { ...SLOT_CAPS };
  const assignment = new Map();
  const edges = [];
  for (const team of fusedTeams) {
    for (const slot of SLOT_KEYS) {
      edges.push({ p: slotProbFromTeam(team, slot), id: team.id, slot });
    }
  }
  edges.sort((a, b) => b.p - a.p);
  for (const e of edges) {
    if (assignment.has(e.id) || remaining[e.slot] <= 0) continue;
    assignment.set(e.id, e.slot);
    remaining[e.slot] -= 1;
  }
  for (const team of fusedTeams) {
    if (assignment.has(team.id)) continue;
    let best = null;
    let bestP = -1;
    for (const slot of SLOT_KEYS) {
      if (remaining[slot] <= 0) continue;
      const p = slotProbFromTeam(team, slot);
      if (p > bestP) {
        bestP = p;
        best = slot;
      }
    }
    if (best) {
      assignment.set(team.id, best);
      remaining[best] -= 1;
    }
  }
  return assignment;
}

function improveAssignmentBySwaps(assignment, fusedTeams, maxRounds = 80) {
  const ids = [...assignment.keys()];
  const asList = () => ids.map((id) => ({ id, slot: assignment.get(id) }));
  let bestScore = expectedValvePointsFromCorrectDist(asList(), fusedTeams);
  for (let round = 0; round < maxRounds; round += 1) {
    let improved = false;
    for (let i = 0; i < ids.length; i += 1) {
      for (let j = i + 1; j < ids.length; j += 1) {
        const a = ids[i];
        const b = ids[j];
        if (assignment.get(a) === assignment.get(b)) continue;
        const sa = assignment.get(a);
        const sb = assignment.get(b);
        assignment.set(a, sb);
        assignment.set(b, sa);
        const score = expectedValvePointsFromCorrectDist(asList(), fusedTeams);
        if (score > bestScore + 1e-9) {
          bestScore = score;
          improved = true;
        } else {
          assignment.set(a, sa);
          assignment.set(b, sb);
        }
      }
    }
    if (!improved) break;
  }
  return assignment;
}

function analystLookup(data) {
  const map = {};
  const board = data.boards?.points_optimal || data.board || {};
  for (const slot of SLOT_KEYS) {
    for (const row of board[slot] || []) {
      map[row.id] = {
        analyst_agreement: row.analyst_agreement,
        analyst_names: row.analyst_names,
      };
    }
  }
  return map;
}

/** Build live custom fusion board + score into data.boards.fusion_custom. */
function rebuildCustomFusionBoard(data, weights) {
  const sources = data.slot_sources;
  if (!sources || !data.teams?.length) return null;
  const shares = effectiveWeightShares(weights);
  const fusedTeams = data.teams.map((t) => {
    const slots = fuseTeamSlotProbs(t.id, shares, sources);
    if (!slots) return { ...t, slots: {} };
    const row = { ...t, slots };
    for (const [slot, probKey] of Object.entries(SLOT_TO_PROB)) {
      row[probKey] = Math.round(slots[slot] * 10000) / 100;
    }
    return row;
  });
  let assignment = greedyAssignment(fusedTeams);
  assignment = improveAssignmentBySwaps(assignment, fusedTeams);
  const lookup = analystLookup(data);
  const board = Object.fromEntries(SLOT_KEYS.map((s) => [s, []]));
  const assignList = [];
  for (const team of fusedTeams) {
    const slot = assignment.get(team.id);
    if (!slot) continue;
    assignList.push({ id: team.id, slot });
    const meta = lookup[team.id] || {};
    board[slot].push({
      id: team.id,
      name: team.name,
      short: team.short,
      record: SLOT_RECORD[slot] || slot,
      qualify_pct: team.qualify_pct,
      slot_pct: Math.round(slotProbFromTeam(team, slot) * 1000) / 10,
      analyst_agreement: meta.analyst_agreement,
      analyst_names: meta.analyst_names,
    });
  }
  const expected_correct = expectedCorrectCount(assignList, fusedTeams);
  const expected_points = expectedValvePointsFromCorrectDist(assignList, fusedTeams);
  if (!data.boards) data.boards = {};
  data.boards.fusion_custom = board;
  if (!data.meta) data.meta = {};
  if (!data.meta.board_compare) data.meta.board_compare = {};
  data.meta.board_compare.fusion_custom = {
    expected_correct: Math.round(expected_correct * 1000) / 1000,
    expected_points: Math.round(expected_points * 10) / 10,
  };
  return data.meta.board_compare.fusion_custom;
}

function ensureCustomBoard(data) {
  if (!data?.slot_sources) return;
  const weights = loadCustomWeights(data.meta || {});
  rebuildCustomFusionBoard(data, weights);
}

function expectedPointsForStrategy(data, strategy) {
  const meta = data.meta || {};
  const compare = meta.board_compare || {};
  if (strategy === "fusion_custom") {
    ensureCustomBoard(data);
  }
  if (compare[strategy]?.expected_points != null) {
    return Number(compare[strategy].expected_points);
  }
  if (strategy === "analyst_consensus" && data.analyst?.expected_points != null) {
    return Number(data.analyst.expected_points);
  }
  const key = fusionScenarioKey(strategy);
  const scores = meta.fusion_scenario_scores || {};
  if (key && key !== "custom" && scores[key]?.expected_points != null) {
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
  if (sel) {
    if (!sel.querySelector(`option[value="${strategy}"]`)) {
      const opt = document.createElement("option");
      opt.value = strategy;
      opt.textContent = t(`strategy.${strategy}`);
      sel.appendChild(opt);
    }
    sel.value = strategy;
  }
  if (persist) localStorage.setItem(BOARD_STORAGE_KEY, strategy);
  if (strategy === "fusion_custom") ensureCustomBoard(data);
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
  const customPanel = document.getElementById("fusion-custom-panel");
  const scoreEl = document.getElementById("fusion-score");
  const disc = document.getElementById("market-disclaimer");
  if (!root || !sliders) return;
  const meta = data.meta || {};
  const strategy = getBoardStrategy();
  const isFusion = strategy === "fusion" || strategy.startsWith("fusion_");
  const isCustom = strategy === "fusion_custom";
  root.hidden = !isFusion;
  if (disc) {
    const mmeta = meta.market_priors_meta || {};
    const derivation = String(mmeta.derivation || "");
    if (mmeta.seeded_from_ranking || mmeta.is_real_market === false) {
      disc.textContent = t("fusion.market_fallback");
    } else if (derivation === "derived_from_winner_odds") {
      disc.textContent = t("fusion.market_derived");
    } else if (derivation === "direct_slot") {
      disc.textContent = t("fusion.market_direct");
    } else {
      disc.textContent = t("fusion.market_fallback");
    }
  }
  if (!isFusion) {
    sliders.innerHTML = "";
    if (presets) presets.innerHTML = "";
    if (scoreEl) scoreEl.textContent = "";
    if (customPanel) customPanel.hidden = true;
    return;
  }

  const map = scenarioWeightsMap(meta);
  const weights = isCustom ? loadCustomWeights(meta) : map[strategy] || map.fusion;
  const shares = effectiveWeightShares(weights);
  const rawSum = FUSION_WEIGHT_KEYS.reduce((s, k) => s + Math.max(0, Number(weights[k] || 0)), 0);

  if (presets) {
    presets.innerHTML = FUSION_PRESET_KEYS.map((k) => {
      const active = k === strategy ? " active" : "";
      const hintKey = `strategy.${k}_hint`;
      const hint = t(hintKey);
      const titleAttr =
        hint && !hint.startsWith("strategy.")
          ? ` title="${escapeHtml(hint)}"`
          : "";
      return `<button type="button" class="fusion-preset${active}" data-strategy="${k}"${titleAttr}>${escapeHtml(t(`strategy.${k}`))}</button>`;
    }).join("");
    presets.querySelectorAll("[data-strategy]").forEach((btn) => {
      btn.addEventListener("click", () => {
        setBoardStrategy(btn.getAttribute("data-strategy"), data);
      });
    });
  }

  if (customPanel) customPanel.hidden = !isCustom;

  if (!isCustom) {
    sliders.innerHTML = "";
  } else {
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

    const applyDraft = (draft) => {
      saveCustomWeights(draft);
      rebuildCustomFusionBoard(data, draft);
      refreshEff(draft);
      renderBoard(data);
      renderModePanel(data);
      renderHero(data);
      const cmp = data.meta?.board_compare?.fusion_custom;
      if (scoreEl && cmp) {
        scoreEl.innerHTML =
          `${escapeHtml(t("fusion.score"))}: <strong>${Math.round(cmp.expected_points).toLocaleString(localeTag())}</strong>` +
          ` · ${escapeHtml(t("fusion.correct_slots"))} ${Number(cmp.expected_correct).toFixed(2)}`;
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
        applyDraft(draft);
      });
    });
  }

  if (scoreEl) {
    if (isCustom) {
      ensureCustomBoard(data);
      const cmp = meta.board_compare?.fusion_custom;
      scoreEl.innerHTML =
        cmp?.expected_points != null
          ? `${escapeHtml(t("fusion.score"))}: <strong>${Math.round(cmp.expected_points).toLocaleString(localeTag())}</strong>` +
            ` · ${escapeHtml(t("fusion.correct_slots"))} ${Number(cmp.expected_correct).toFixed(2)}`
          : `<span class="meta">${escapeHtml(t("fusion.custom_unavailable"))}</span>`;
    } else {
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
    // Hide internal/diagnostic export warnings from the product UI.
    const warnings = Array.isArray(meta.warnings) ? meta.warnings : [];
    const localized = warnings
      .filter(
        (w) =>
          !/In-sample tune|derived from Polymarket|seeded from POWER_RANKINGS/i.test(
            String(w)
          )
      )
      .map(localizeWarning)
      .filter(Boolean);
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
  const modelChipLabel = friendlyModelLabel(meta);
  const mmeta = meta.market_priors_meta || {};
  let marketChip = "";
  if (mmeta.seeded_from_ranking || mmeta.is_real_market === false) {
    marketChip = "";
  } else if (String(mmeta.derivation || "") === "derived_from_winner_odds") {
    marketChip = `<span class="chip" title="${escapeHtml(t("fusion.market_derived"))}">${escapeHtml(t("chip.market_derived"))}</span>`;
  } else if (String(mmeta.derivation || "") === "direct_slot") {
    marketChip = `<span class="chip" title="${escapeHtml(t("fusion.market_direct"))}">${escapeHtml(t("chip.market_direct"))}</span>`;
  } else if (mmeta.is_real_market) {
    marketChip = `<span class="chip">${escapeHtml(t("chip.market_live"))}</span>`;
  }
  const nAnalysts = data.analyst?.n_analysts;
  const analystsChip =
    nAnalysts != null
      ? `<span class="chip">${escapeHtml(t("chip.analysts"))}: <strong>${fmtNum(nAnalysts)}</strong></span>`
      : "";
  document.getElementById("hero-meta").innerHTML = `
    <span class="chip"><strong>${escapeHtml(modelChipLabel)}</strong></span>
    ${analystsChip}
    <span class="chip">${escapeHtml(t("chip.sims"))}: <strong>${fmtNum(meta.n_simulations)}</strong></span>
    ${pts}
    ${marketChip}
    <span class="chip">${escapeHtml(friendlyFormatLabel(meta.format))}</span>
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
      <div class="name"><span class="name-text">${displayName}</span>${chip}</div>
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
