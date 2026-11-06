const LABELS = {
  TransactionAmt: "Amount (USD)",
  ProductCD: "Product type",
  card4: "Card network",
  card6: "Card type",
  P_emaildomain: "Buyer email domain",
  DeviceType: "Device type",
};

let transactions = [];
let current = null;
let filter = "all";

const $ = (id) => document.getElementById(id);
const pct = (x) => (x * 100).toFixed(1) + "%";

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
  const rows = transactions.filter((t) => filter === "all" || String(t.is_fraud) === filter);
  $("list").innerHTML = rows.map((t) => `
    <div class="item ${current && current.id === t.id ? "sel" : ""}" data-id="${t.id}">
      <span class="num">#${t.id}</span>
      <span class="tag ${t.is_fraud ? "fraud" : "legit"}">${t.is_fraud ? "fraud" : "legit"}</span>
      <span class="sub">$${Number(t.fields.TransactionAmt).toFixed(2)} · product ${t.fields.ProductCD ?? "?"} · ${t.fields.card4 ?? "unknown card"}</span>
    </div>`).join("");
}

function show(t) {
  current = t;
  const r = t.result;
  $("title").textContent = `Transaction #${t.id}`;
  $("fields").innerHTML = Object.entries(LABELS).map(([k, label]) => {
    let v = t.fields[k];
    if (k === "TransactionAmt" && v != null) v = "$" + Number(v).toFixed(2);
    return `<div><label>${label}</label><div class="fv">${v ?? "missing"}</div></div>`;
  }).join("");

  $("result").style.display = "block";
  $("prob").textContent = pct(r.fraud_probability);
  $("pill").textContent = r.flagged ? "Flag for review" : "Approve";
  $("pill").className = "pill " + (r.flagged ? "flag" : "ok");
  const right = r.flagged === Boolean(t.is_fraud);
  $("actual").textContent = `Actually ${t.is_fraud ? "fraud" : "legit"}: ${right ? "model got it right" : "model got it wrong"}`;
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
  showList();
}

document.querySelectorAll(".filters button").forEach((b) => b.addEventListener("click", () => {
  filter = b.dataset.f;
  document.querySelectorAll(".filters button").forEach((x) => x.classList.toggle("on", x === b));
  showList();
}));
$("list").addEventListener("click", (e) => {
  const item = e.target.closest(".item");
  if (item) show(transactions.find((t) => t.id === Number(item.dataset.id)));
});

fetch("data.json").then((r) => r.json()).then((data) => {
  showMetrics(data.metrics);
  transactions = data.transactions;
  showList();
  if (transactions.length) show(transactions.find((t) => t.is_fraud && t.result.flagged) ?? transactions[0]);
});
