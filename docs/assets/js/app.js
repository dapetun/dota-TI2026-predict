const DATA_URL = "data/predictions.json";

const BOARD_COLUMNS = [
  { key: "undefeated", title: "3–0 · Undefeated", tone: "ok" },
  { key: "one_loss", title: "3–1 · One loss", tone: "ok" },
  { key: "advancing", title: "3–2 · Advancing", tone: "warn" },
  { key: "borderline", title: "2–3 · Borderline", tone: "warn" },
  { key: "one_win", title: "1–3 · One win", tone: "bad" },
  { key: "winless", title: "0–3 · Winless", tone: "bad" },
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

function renderHero(data) {
  const meta = data.meta;
  document.getElementById("disclaimer-text").textContent = meta.disclaimer;
  document.getElementById("hero-meta").innerHTML = `
    <span class="chip"><strong>${meta.model_label}</strong></span>
    <span class="chip">Симуляций <strong>${meta.n_simulations.toLocaleString("ru-RU")}</strong></span>
    <span class="chip">${meta.format}</span>
    <span class="chip">v${meta.version}</span>
  `;
  document.getElementById("footer-stamp").textContent =
    `Обновлено: ${new Date(meta.generated_at).toLocaleString("ru-RU")} · ${meta.model}`;
}

function renderBoard(data) {
  const root = document.getElementById("swiss-board");
  root.innerHTML = BOARD_COLUMNS.map((col) => {
    const teams = data.board[col.key] || [];
    const pills = teams.length
      ? teams
          .map(
            (t) => `
          <div class="team-pill" title="${t.name}">
            <div>
              <div class="name">${t.short}</div>
              <div class="meta">${t.record}</div>
            </div>
            <div class="prob">${fmtPct(t.qualify_pct)}</div>
          </div>`
          )
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
      return `
      <tr>
        <td>${i + 1}</td>
        <td class="team-cell">
          ${t.name}
          <span class="sub">${t.source} · rank ${t.power_rank}</span>
        </td>
        <td>${t.region}</td>
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
  const cards = [
    {
      label: "Walk-forward AUC",
      value: m.walk_forward_avg_auc ?? "—",
      hint: "XGBoost v0.1",
    },
    {
      label: "Leave-One-TI AUC",
      value: m.leave_one_ti_avg_auc ?? "—",
      hint: "Среднее по прошлым TI",
    },
    {
      label: "Walk-forward LogLoss",
      value: m.walk_forward_avg_logloss ?? "—",
      hint: "Чем ниже, тем лучше",
    },
    {
      label: "UI source",
      value: "Baseline",
      hint: data.meta.model_label,
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
}

async function main() {
  try {
    DATA = await loadData();
    renderHero(DATA);
    renderBoard(DATA);
    fillMatchupSelects(DATA);
    renderStandings(DATA);
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
