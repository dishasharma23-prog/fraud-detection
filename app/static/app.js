const FIELDS = [
  ["TransactionAmt", "Amount (USD)", "number"],
  ["ProductCD", "Product type", ["W", "C", "H", "R", "S"]],
  ["card4", "Card network", ["visa", "mastercard", "american express", "discover"]],
  ["card6", "Card type", ["debit", "credit"]],
  ["P_emaildomain", "Buyer email domain", "text"],
  ["DeviceType", "Device type", ["desktop", "mobile"]],
];

let samples = [];
let current = null;
let filter = "all";

const $ = (id) => document.getElementById(id);
const pct = (x) => (x * 100).toFixed(1) + "%";

const HINT = "Change any field and score it again to see how the model reacts.";

async function getJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let message = `Request failed (${res.status})`;
    try { message = (await res.json()).detail ?? message; } catch (e) {}
    throw new Error(typeof message === "string" ? message : "Some fields are not valid");
  }
  return res.json();
}

function showHint(text, isError) {
  $("hint").textContent = text;
  $("hint").classList.toggle("error", Boolean(isError));
}

function showMetrics(m) {
  const cards = [
    [m.pr_auc.toFixed(3), `PR-AUC (random guessing: ${m.pr_auc_random_baseline.toFixed(3)})`],
    [m.roc_auc.toFixed(3), "ROC-AUC"],
    [pct(m.at_threshold.recall), `of fraud caught, with ${pct(m.at_threshold.false_positive_rate)} of legit flagged`],
    [m.rows.test.toLocaleString(), `test transactions (${m.test_fraud_cases.toLocaleString()} fraud)`],
  ];
  $("metrics").innerHTML = cards.map(([v, l]) =>
    `<div class="metric"><div class="value num">${v}</div><div class="label">${l}</div></div>`).join("");
}

function showList() {
  const rows = samples.filter((s) => filter === "all" || String(s.is_fraud) === filter);
  $("list").innerHTML = rows.map((s) => `
    <div class="item ${current && current.id === s.id ? "sel" : ""}" data-id="${s.id}">
      <span class="num">#${s.id}</span>
      <span class="tag ${s.is_fraud ? "fraud" : "legit"}">${s.is_fraud ? "fraud" : "legit"}</span>
      <span class="sub">$${Number(s.amount).toFixed(2)} · product ${s.product ?? "?"} · ${s.card ?? "unknown card"}</span>
    </div>`).join("");
}

function field(key, label, kind, value) {
  const v = value ?? "";
  if (Array.isArray(kind)) {
    const opts = ["", ...kind].map((o) =>
      `<option value="${o}" ${String(o) === String(v) ? "selected" : ""}>${o || "missing"}</option>`).join("");
    return `<div><label>${label}<select data-key="${key}">${opts}</select></label></div>`;
  }
  const shown = kind === "number" && v !== "" ? Number(v).toFixed(2) : v;
  return `<div><label>${label}<input data-key="${key}" type="${kind}" step="0.01" value="${shown}"></label></div>`;
}

async function pick(id) {
  current = await getJSON(`/api/samples/${id}`);
  $("title").textContent = `Transaction #${current.id}`;
  $("fields").innerHTML = FIELDS.map(([k, l, kind]) => field(k, l, kind, current.values[k])).join("");
  showList();
  await scoreCurrent();
}

function edited() {
  const values = { ...current.values };
  let changed = false;
  document.querySelectorAll("[data-key]").forEach((el) => {
    const key = el.dataset.key;
    let v = el.value === "" ? null : el.value;
    if (el.type === "number" && v !== null) v = Number(v);
    const before = current.values[key] ?? null;
    const same = el.type === "number" ? Math.abs((before ?? 0) - (v ?? 0)) < 0.005 && (before === null) === (v === null) : before === v;
    if (!same) changed = true;
    if (v === null) delete values[key]; else values[key] = v;
  });
  return { values, changed };
}

async function scoreCurrent() {
  if (!current) return;
  const { values, changed } = edited();
  let r;
  try {
    r = await getJSON("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
    });
  } catch (err) {
    $("result").style.display = "none";
    showHint(`Can't score this: ${err.message}.`, true);
    return;
  }
  showHint(HINT, false);
  $("result").style.display = "block";
  $("prob").textContent = pct(r.fraud_probability);
  $("pill").textContent = r.flagged ? "Flag for review" : "Approve";
  $("pill").className = "pill " + (r.flagged ? "flag" : "ok");
  $("actual").textContent = changed ? "Edited transaction" : `Actually: ${current.is_fraud ? "fraud" : "legit"}`;
  $("fill").style.width = Math.max(1, r.fraud_probability * 100) + "%";
  $("fill").style.background = r.flagged ? "var(--danger)" : "var(--safe)";
  $("mark").style.left = r.threshold * 100 + "%";
  $("threshold-note").textContent = `Flagged above ${pct(r.threshold)} (the black line)`;
  const biggest = Math.max(...r.reasons.map((x) => Math.abs(x.impact)), 1e-9);
  $("reasons").innerHTML = r.reasons.map((x) => `
    <div class="reason">
      <span>${x.label}</span>
      <div class="rbar ${x.impact > 0 ? "up" : "down"}" style="width:${(Math.abs(x.impact) / biggest) * 100}%"></div>
    </div>`).join("");
}

document.querySelectorAll(".filters button").forEach((b) => b.addEventListener("click", () => {
  filter = b.dataset.f;
  document.querySelectorAll(".filters button").forEach((x) => x.classList.toggle("on", x === b));
  showList();
}));
$("list").addEventListener("click", (e) => {
  const item = e.target.closest(".item");
  if (item) pick(Number(item.dataset.id));
});
$("score").addEventListener("click", scoreCurrent);

(async () => {
  showMetrics(await getJSON("/api/metrics"));
  samples = await getJSON("/api/samples");
  showList();
  if (samples.length) pick(samples.find((s) => s.is_fraud)?.id ?? samples[0].id);
})();
