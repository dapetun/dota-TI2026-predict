const DATA_URL = "data/predictions.json";
const BOARD_STORAGE_KEY = "ti2026_board_strategy";
const ANALYST_THRESHOLD = 5;

const BOARD_COLUMNS = [
  { key: "undefeated", title: "4–0 · Undefeated", tone: "ok" },
  { key: "one_loss", title: "4–1 · One loss", tone: "ok" },
  { key: "advance", title: "Проход · Advancing", tone: "warn" },
  { key: "eliminate", title: "Выбывание · Out", tone: "warn" },
  { key: "one_win", title: "1–4 · One win", tone: "bad" },
  { key: "winless", title: "0–4 · Winless", tone: "bad" },
];

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

async function loadData() {
  const res = await fetch(DATA_URL);
  if (!res.ok) throw new Error(`Не удалось загрузить predictions.json (${res.status})`);
  return res.json();
}

function fmtPct(v) {
  return `${Number(v).toFixed(1)}%`;
}

function valvePointsTooltip() {
  const rows = VALVE_POINTS_TABLE.map(([k, pts]) => `${k}/16 → ${pts.toLocaleString("ru-RU")}`);
  return `Очки Valve за точные слоты:\n${rows.join("\n")}`;
}

function getBoardStrategy() {
  const sel = document.getElementById("board-strategy");
  return sel?.value || localStorage.getItem(BOARD_STORAGE_KEY) || "points_optimal";
}

function activeBoard(data) {
  const strategy = getBoardStrategy();
  if (data.boards && data.boards[strategy]) return data.boards[strategy];
  return data.board;
}

function renderHero(data) {
  const meta = data.meta;
  document.getElementById("disclaimer-text").textContent = meta.disclaimer;
  const strategy = getBoardStrategy();
  const compare = meta.board_compare || {};
  const stratPts = compare[strategy]?.expected_points;
  const ptsVal = stratPts ?? meta.expected_compendium_points;
  const pts =
    ptsVal != null
      ? `<span class="chip" title="${valvePointsTooltip()}">E[очки] <strong>${Math.round(ptsVal).toLocaleString("ru-RU")}</strong></span>`
      : "";
  const analystPts = data.analyst?.expected_points;
  const analystChip =
    analystPts != null
      ? `<span class="chip">Консенсус <strong>${Math.round(analystPts).toLocaleString("ru-RU")}</strong></span>`
      : "";
  const fusion =
    meta.fusion_expected_points != null
      ? `<span class="chip">Fusion <strong>${Math.round(meta.fusion_expected_points).toLocaleString("ru-RU")}</strong></span>`
      : "";
  document.getElementById("hero-meta").innerHTML = `
    <span class="chip"><strong>${meta.model_label}</strong></span>
    <span class="chip">Симуляций <strong>${meta.n_simulations.toLocaleString("ru-RU")}</strong></span>
    ${pts}
    ${analystChip}
    ${fusion}
    <span class="chip">${meta.format}</span>
    <span class="chip">v${meta.version}</span>
  `;
  document.getElementById("footer-stamp").textContent =
    `Обновлено: ${new Date(meta.generated_at).toLocaleString("ru-RU")} · ${meta.model}`;
}

function renderBoard(data) {
  const root = document.getElementById("swiss-board");
  const board = activeBoard(data);
  const nAnalysts = data.analyst?.n_analysts || 11;
  root.innerHTML = BOARD_COLUMNS.map((col) => {
    const teams = board[col.key] || [];
    const pills = teams.length
      ? teams
          .map((t) => {
            const agree = t.analyst_agreement ?? 0;
            const names = (t.analyst_names || []).join(", ");
            const chip =
              agree >= ANALYST_THRESHOLD
                ? `<span class="chip mini" title="${names}">${agree}/${nAnalysts}</span>`
                : "";
            const slotPct = t.slot_pct ?? 0;
            return `
          <div class="team-pill" title="${t.name}${names ? " · " + names : ""}">
            <div>
              <div class="name">${t.name || t.short} ${chip}</div>
              <div class="meta">${t.record} · P(слот) ${fmtPct(slotPct)}</div>
            </div>
            <div class="prob">${fmtPct(t.qualify_pct)}</div>
          </div>`;
          })
          .join("")
      : `<p class="meta" style="color:var(--muted);margin:0;font-size:0.85rem">Пока пусто</p>`;
    return `
      <div class="board-col ${col.tone}">
        <h4>${col.title}</h4>
        ${pills}
      </div>`;
  }).join("");
}

function sortTeams(teams, mode) {
  const copy = [...teams];
  if (mode === "power") copy.sort((a, b) => a.power_rank - b.power_rank);
  else if (mode === "elim") copy.sort((a, b) => b.eliminated_pct - a.eliminated_pct);
  else if (mode === "alpha") copy.sort((a, b) => a.name.localeCompare(b.name, "ru"));
  else copy.sort((a, b) => b.qualify_pct - a.qualify_pct);
  return copy;
}

function renderStandings(data) {
  const mode = document.getElementById("sort-by").value;
  const q = document.getElementById("search").value.trim().toLowerCase();
  let teams = sortTeams(data.teams, mode);
  if (q) {
    teams = teams.filter(
      (t) =>
        t.name.toLowerCase().includes(q) ||
        t.short.toLowerCase().includes(q) ||
        t.region.toLowerCase().includes(q)
    );
  }
  const tbody = document.querySelector("#standings-table tbody");
  tbody.innerHTML = teams
    .map((t, i) => {
      const strength =
        t.strength_label ||
        (t.strength_mu != null
          ? `${Math.round(t.strength_mu)} ± ${Math.round(t.strength_sigma || 0)}`
          : "—");
      const home =
        t.home_lan_elo > 0 ? ` · home +${t.home_lan_elo}` : "";
      return `
      <tr>
        <td>${i + 1}</td>
        <td class="team-cell">
          ${t.name}
          <span class="sub">${t.source} · rank ${t.power_rank}${home}</span>
        </td>
        <td>${t.region}</td>
        <td class="strength-cell" title="Elo shrunk ± combined σ">${strength}</td>
        <td>
          <div class="bar">
            <span>${fmtPct(t.qualify_pct)}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${t.qualify_pct}%"></div></div>
          </div>
        </td>
        <td>
          <div class="bar">
            <span>${fmtPct(t.eliminated_pct)}</span>
            <div class="bar-track"><div class="bar-fill danger" style="width:${t.eliminated_pct}%"></div></div>
          </div>
        </td>
        <td>${t.expected_wins.toFixed(2)}</td>
        <td>${t.most_likely_record}</td>
      </tr>`;
    })
    .join("");
}

function heatColor(pct) {
  const v = Math.max(0, Math.min(100, Number(pct) || 0));
  const alpha = 0.08 + (v / 100) * 0.55;
  return `rgba(226, 179, 87, ${alpha.toFixed(3)})`;
}

function renderHeatmap(data) {
  const root = document.getElementById("slot-heatmap");
  if (!root) return;
  const hm = data.slot_heatmap;
  if (!hm || !hm.matrix) {
    root.innerHTML = "<tbody><tr><td>Нет данных heatmap</td></tr></tbody>";
    return;
  }
  const head = `<thead><tr><th>Команда</th>${hm.slots
    .map((s) => `<th>${s}</th>`)
    .join("")}</tr></thead>`;
  const body = hm.matrix
    .map((row, i) => {
      const team = hm.teams[i];
      const cells = row
        .map(
          (v) =>
            `<td style="background:${heatColor(v)}" title="${team.name} · ${v}%">${Number(v).toFixed(1)}</td>`
        )
        .join("");
      return `<tr><th scope="row">${team.short || team.name}</th>${cells}</tr>`;
    })
    .join("");
  root.innerHTML = `${head}<tbody>${body}</tbody>`;
}

function fillMatchupSelects(data) {
  const opts = data.teams
    .slice()
    .sort((a, b) => a.name.localeCompare(b.name, "ru"))
    .map((t) => `<option value="${t.id}">${t.name}</option>`)
    .join("");
  const a = document.getElementById("team-a");
  const b = document.getElementById("team-b");
  a.innerHTML = opts;
  b.innerHTML = opts;
  a.value = data.teams[0]?.id || "";
  b.value = data.teams[1]?.id || data.teams[0]?.id || "";
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
  const teamA = data.teams.find((t) => t.id === idA);
  const teamB = data.teams.find((t) => t.id === idB);
  const m = findMatchup(data, idA, idB);
  const pa = Math.round(m.p_a * 1000) / 10;
  const pb = Math.round(m.p_b * 1000) / 10;
  document.getElementById("matchup-result").innerHTML = `
    <div class="duel">
      <div class="duel-side">
        <div class="name">${teamA?.name || idA}</div>
        <div class="pct" style="color:var(--gold)">${pa}%</div>
      </div>
      <div class="meta">вероятность победы серии</div>
      <div class="duel-side">
        <div class="name">${teamB?.name || idB}</div>
        <div class="pct" style="color:var(--teal)">${pb}%</div>
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
  if (meth && meta.methodology) {
    meth.textContent = meta.methodology;
  }
  const cov = m.player_coverage;
  const covLabel =
    cov && cov.coverage != null
      ? `${Math.round(cov.coverage * 100)}%`
      : "—";
  const blendAuc = m.blend_leave_one_ti_avg_auc;
  const compare = data.meta?.board_compare || {};
  const cards = [
    {
      label: "Leave-One-TI AUC",
      value: blendAuc ?? m.leave_one_ti_avg_auc ?? "—",
      hint: blendAuc != null ? "Blend XGB+CatBoost" : "XGBoost team+player",
    },
    {
      label: "Корпус",
      value:
        meta.n_maps != null
          ? `${meta.n_maps}`
          : cov?.n_matches ?? "—",
      hint: meta.n_leagues != null ? `${meta.n_leagues} лиг` : "maps",
    },
    {
      label: "E[очки] model",
      value: compare.points_optimal?.expected_points?.toFixed(0) ?? "—",
      hint: "Points-optimal board",
    },
    {
      label: "Player coverage",
      value: covLabel,
      hint: cov
        ? `${cov.n_matches_with_players || 0} / ${cov.n_matches || "?"} матчей`
        : "OpenDota details",
    },
  ];
  document.getElementById("metrics-grid").innerHTML = cards
    .map(
      (c) => `
    <article class="metric">
      <p class="label">${c.label}</p>
      <p class="value">${c.value}</p>
      <p class="hint">${c.hint}</p>
    </article>`
    )
    .join("");
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
      renderBoard(data);
    });
  }
}

async function main() {
  try {
    DATA = await loadData();
    renderHero(DATA);
    renderBoard(DATA);
    fillMatchupSelects(DATA);
    renderStandings(DATA);
    renderHeatmap(DATA);
    renderMatchup(DATA);
    renderMetrics(DATA);
    bind(DATA);
  } catch (err) {
    document.getElementById("disclaimer-text").textContent =
      `Ошибка загрузки данных: ${err.message}. Открой через локальный сервер или GitHub Pages.`;
    console.error(err);
  }
}

main();
