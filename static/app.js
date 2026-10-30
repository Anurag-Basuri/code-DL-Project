/**
 * Hinglish Language Identifier — Client Controller
 * Clean, minimal, two-mode architecture: Single Sentence & Batch Analysis.
 */

// -----------------------------------------------------------------------------
// Presets & Constants
// -----------------------------------------------------------------------------
const SINGLE_PRESETS = [
  "bhai ye app bahut slow chal raha hai please check karo",
  "kal meeting ka time kya hai ?",
  "mujhe flight book karni hai Mumbai se Goa tak",
  "Flipkart pe sale kab start hogi ?",
  "server down hai isliye API response nahi de raha"
];

const BATCH_PRESETS = {
  conv: [
    "bhai ye app bahut slow chal raha hai please check karo",
    "kal meeting ka time kya hai ?",
    "mujhe flight book karni hai Mumbai se Goa tak",
    "Flipkart pe sale kab start hogi ?",
    "server down hai isliye API response nahi de raha",
    "can you mujhe ek doosra option dikha sakte ho ?",
    "Rahul ne mujhe email bheja tha kal sham ko",
    "ye movie toh bilkul bakwaas thi"
  ],
  tech: [
    "mera payment deduct ho gaya par order confirm nahi hua",
    "login issue aa raha hai OTP deliver nahi ho raha",
    "database connection timeout ho raha hai compute restart karo",
    "error 404 page not found dikha raha hai link expired hai kya",
    "Google Pay se transaction fail ho gaya refund kab aayega ?",
    "password reset link email pe send kar dijiye please"
  ],
  contrast: [
    "aaj ka mausam kafi achha hai aur dhoop nikli hai",
    "The quick brown fox jumps over the lazy dog.",
    "bhai main to bore ho gaya hu let us go for a movie",
    "kal Prime Video pe new series launch hone wali hai",
    "actually maine notice kiya ki server performance degrade ho rahi hai",
    "thank you so much aapki help ke liye"
  ]
};

const TAG_INFO = {
  "HI": {
    label: "Hindi",
    desc: "Hindi word",
    class: "token-hi",
    role: "Hindi vocabulary"
  },
  "EN": {
    label: "English",
    desc: "English word",
    class: "token-en",
    role: "English vocabulary"
  },
  "NE": {
    label: "Name / Brand",
    desc: "Named entity (person, place, brand)",
    class: "token-ne",
    role: "Proper noun"
  },
  "OTHER": {
    label: "Symbol",
    desc: "Punctuation or symbol",
    class: "token-other",
    role: "Punctuation / symbol"
  }
};

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------
let currentBatchData = null;
let activeBatchFilter = "all";
let batchSearchQuery = "";

// -----------------------------------------------------------------------------
// DOM References
// -----------------------------------------------------------------------------
const tabSingle = document.getElementById("tab-single");
const tabBatch = document.getElementById("tab-batch");
const viewSingle = document.getElementById("view-single");
const viewBatch = document.getElementById("view-batch");

const modelSelector = document.getElementById("model-selector");
const statusLabel = document.getElementById("status-label");
const systemStatus = document.getElementById("system-status");
const toastMsg = document.getElementById("toast-msg");

// Single Mode Elements
const textInput = document.getElementById("text-input");
const charCount = document.getElementById("char-count");
const btnAnalyze = document.getElementById("btn-analyze");
const btnClear = document.getElementById("btn-clear");
const presetsList = document.getElementById("presets-list");
const loadingSingle = document.getElementById("loading-single");
const resultSingle = document.getElementById("result-single");
const tokensDisplay = document.getElementById("tokens-display");
const singleCmiVal = document.getElementById("single-cmi-val");
const singleCmiBadge = document.getElementById("single-cmi-badge");
const singleLatency = document.getElementById("single-latency");

const inspHint = document.getElementById("insp-hint");
const inspContent = document.getElementById("insp-content");
const inspWord = document.getElementById("insp-word");
const inspTag = document.getElementById("insp-tag");
const inspConf = document.getElementById("insp-conf");
const inspRole = document.getElementById("insp-role");

const cntHi = document.getElementById("cnt-hi");
const cntEn = document.getElementById("cnt-en");
const cntNe = document.getElementById("cnt-ne");
const cntOther = document.getElementById("cnt-other");

// Batch Mode Elements
const batchInput = document.getElementById("batch-input");
const batchCounter = document.getElementById("batch-counter");
const btnAnalyzeBatch = document.getElementById("btn-analyze-batch");
const btnClearBatch = document.getElementById("btn-clear-batch");
const loadingBatch = document.getElementById("loading-batch");
const resultBatch = document.getElementById("result-batch");

const kpiSentences = document.getElementById("kpi-sentences");
const kpiTokens = document.getElementById("kpi-tokens");
const kpiAvgCmi = document.getElementById("kpi-avg-cmi");
const kpiCmiLevel = document.getElementById("kpi-cmi-level");
const kpiDominantLang = document.getElementById("kpi-dominant-lang");
const kpiLangRatio = document.getElementById("kpi-lang-ratio");
const btnExportCsv = document.getElementById("btn-export-csv");

const batchSearchInput = document.getElementById("batch-search-input");
const searchClearBtn = document.getElementById("search-clear-btn");
const batchListContainer = document.getElementById("batch-list-container");

const filterCntAll = document.getElementById("filter-cnt-all");
const filterCntMixed = document.getElementById("filter-cnt-mixed");
const filterCntMono = document.getElementById("filter-cnt-mono");
const filterCntHi = document.getElementById("filter-cnt-hi");
const filterCntEn = document.getElementById("filter-cnt-en");

// -----------------------------------------------------------------------------
// App Initialization
// -----------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  renderSinglePresets();
  setupEventListeners();
  checkHealth();
  updateSingleWordCount();
  updateBatchCounter();
});

// -----------------------------------------------------------------------------
// Mode Switching
// -----------------------------------------------------------------------------
function setupEventListeners() {
  // Top Tabs: Single vs Batch
  tabSingle.addEventListener("click", () => {
    tabSingle.classList.add("active");
    tabBatch.classList.remove("active");
    viewSingle.classList.remove("hidden");
    viewBatch.classList.add("hidden");
  });

  tabBatch.addEventListener("click", () => {
    tabBatch.classList.add("active");
    tabSingle.classList.remove("active");
    viewBatch.classList.remove("hidden");
    viewSingle.classList.add("hidden");
  });

  // Single sentence input events
  textInput.addEventListener("input", updateSingleWordCount);
  textInput.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") {
      e.preventDefault();
      analyzeSingleText();
    }
  });

  btnAnalyze.addEventListener("click", analyzeSingleText);
  btnClear.addEventListener("click", () => {
    textInput.value = "";
    updateSingleWordCount();
    resultSingle.classList.add("hidden");
    clearInspector();
  });

  // Batch input events
  batchInput.addEventListener("input", updateBatchCounter);
  batchInput.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.key === "Enter") {
      e.preventDefault();
      analyzeBatch();
    }
  });

  btnAnalyzeBatch.addEventListener("click", analyzeBatch);
  btnClearBatch.addEventListener("click", () => {
    batchInput.value = "";
    updateBatchCounter();
    resultBatch.classList.add("hidden");
    currentBatchData = null;
  });

  // Batch presets
  document.getElementById("btn-batch-preset-conv").addEventListener("click", () => {
    batchInput.value = BATCH_PRESETS.conv.join("\n");
    updateBatchCounter();
    analyzeBatch();
  });

  document.getElementById("btn-batch-preset-tech").addEventListener("click", () => {
    batchInput.value = BATCH_PRESETS.tech.join("\n");
    updateBatchCounter();
    analyzeBatch();
  });

  document.getElementById("btn-batch-preset-contrast").addEventListener("click", () => {
    batchInput.value = BATCH_PRESETS.contrast.join("\n");
    updateBatchCounter();
    analyzeBatch();
  });

  // File upload for batch
  const fileInput = document.getElementById("batch-file-input");
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target.result;
      const lines = content.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0);
      batchInput.value = lines.slice(0, 50).join("\n");
      updateBatchCounter();
      showToast(`Loaded ${Math.min(lines.length, 50)} sentences from ${file.name}`);
    };
    reader.readAsText(file);
    fileInput.value = "";
  });

  // Batch Export CSV
  btnExportCsv.addEventListener("click", exportBatchCsv);

  // Batch Search & Filters
  batchSearchInput.addEventListener("input", () => {
    batchSearchQuery = batchSearchInput.value.trim().toLowerCase();
    searchClearBtn.classList.toggle("hidden", batchSearchQuery.length === 0);
    renderBatchList();
  });

  searchClearBtn.addEventListener("click", () => {
    batchSearchInput.value = "";
    batchSearchQuery = "";
    searchClearBtn.classList.add("hidden");
    renderBatchList();
  });

  document.querySelectorAll(".batch-filters .filter-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".batch-filters .filter-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      activeBatchFilter = pill.dataset.filter;
      renderBatchList();
    });
  });
}

function renderSinglePresets() {
  presetsList.innerHTML = "";
  SINGLE_PRESETS.forEach(text => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "preset-chip";
    chip.textContent = text;
    chip.addEventListener("click", () => {
      textInput.value = text;
      updateSingleWordCount();
      analyzeSingleText();
    });
    presetsList.appendChild(chip);
  });
}

function updateSingleWordCount() {
  const words = textInput.value.trim() ? textInput.value.trim().split(/\s+/).length : 0;
  charCount.textContent = `${words} word${words === 1 ? '' : 's'}`;
}

function updateBatchCounter() {
  const lines = batchInput.value.trim() ? batchInput.value.trim().split(/\n+/).filter(l => l.trim().length > 0) : [];
  batchCounter.textContent = `${lines.length} sentence${lines.length === 1 ? '' : 's'}`;
}

// -----------------------------------------------------------------------------
// Health Check
// -----------------------------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      statusLabel.textContent = "Ready";
      systemStatus.querySelector(".status-dot").style.background = "#4ade80";
    }
  } catch (err) {
    statusLabel.textContent = "Offline";
    systemStatus.querySelector(".status-dot").style.background = "#ef4444";
  }
}

// -----------------------------------------------------------------------------
// 1. Single Sentence Analysis
// -----------------------------------------------------------------------------
async function analyzeSingleText() {
  const text = textInput.value.trim();
  if (!text) {
    showToast("Please enter a sentence to analyze.");
    textInput.focus();
    return;
  }

  loadingSingle.classList.remove("hidden");
  resultSingle.classList.add("hidden");
  const startTime = performance.now();

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text: text,
        model: modelSelector.value
      })
    });

    const elapsed = Math.round(performance.now() - startTime);

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Analysis failed");
    }

    const data = await res.json();
    renderSingleResults(data, elapsed);
    resultSingle.classList.remove("hidden");
  } catch (err) {
    showToast(`Error: ${err.message}`);
  } finally {
    loadingSingle.classList.add("hidden");
  }
}

function renderSingleResults(data, elapsed) {
  const { tokens, tags, confidences, cmi, summary } = data;

  singleLatency.textContent = `${data.execution_ms || elapsed}ms`;
  singleCmiVal.textContent = `${cmi}%`;

  // Clean mixing badge label
  if (cmi === 0) {
    singleCmiBadge.textContent = "Single language";
  } else if (cmi < 20) {
    singleCmiBadge.textContent = "Light mixing";
  } else if (cmi < 35) {
    singleCmiBadge.textContent = "Moderate mixing";
  } else {
    singleCmiBadge.textContent = "Heavy mixing";
  }

  // Breakdown counts
  cntHi.textContent = `Hindi: ${summary["HI"] || 0}`;
  cntEn.textContent = `English: ${summary["EN"] || 0}`;
  cntNe.textContent = `Names: ${summary["NE"] || 0}`;
  cntOther.textContent = `Symbols: ${summary["OTHER"] || 0}`;

  // Render Word Pills
  tokensDisplay.innerHTML = "";
  clearInspector();

  tokens.forEach((tok, idx) => {
    const tag = tags[idx];
    const conf = confidences[idx];
    const info = TAG_INFO[tag] || TAG_INFO["OTHER"];

    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `token-chip ${info.class}`;
    chip.innerHTML = `
      <span class="token-chip-word">${escapeHtml(tok)}</span>
      <span class="token-chip-tag">${tag}</span>
    `;

    chip.addEventListener("click", () => showTokenInspector(tok, tag, conf, info, chip));
    chip.addEventListener("mouseenter", () => showTokenInspector(tok, tag, conf, info, chip));

    tokensDisplay.appendChild(chip);
  });

  // Default inspector to first word
  if (tokens.length > 0) {
    const firstChip = tokensDisplay.querySelector(".token-chip");
    showTokenInspector(tokens[0], tags[0], confidences[0], TAG_INFO[tags[0]] || TAG_INFO["OTHER"], firstChip);
  }
}

function showTokenInspector(token, tag, conf, info, chipElement) {
  document.querySelectorAll(".token-chip").forEach(c => c.classList.remove("active"));
  if (chipElement) chipElement.classList.add("active");

  inspHint.classList.add("hidden");
  inspContent.classList.remove("hidden");

  inspWord.textContent = `"${token}"`;
  inspTag.textContent = `${info.label} (${tag})`;
  inspTag.style.background = `var(--color-${tag.toLowerCase()}-accent)`;
  inspConf.textContent = `Confidence: ${(conf * 100).toFixed(1)}%`;
  inspRole.textContent = `• ${info.desc}`;
}

function clearInspector() {
  inspHint.classList.remove("hidden");
  inspContent.classList.add("hidden");
}

// -----------------------------------------------------------------------------
// 2. Batch Analysis
// -----------------------------------------------------------------------------
async function analyzeBatch() {
  const raw = batchInput.value.trim();
  if (!raw) {
    showToast("Please enter sentences to analyze.");
    batchInput.focus();
    return;
  }

  const lines = raw.split("\n").map(l => l.trim()).filter(l => l.length > 0);
  if (lines.length === 0) return;

  loadingBatch.classList.remove("hidden");
  resultBatch.classList.add("hidden");

  try {
    const res = await fetch("/api/predict/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        texts: lines,
        model: modelSelector.value
      })
    });

    if (!res.ok) throw new Error("Batch processing failed");

    const data = await res.json();
    currentBatchData = data;
    renderBatchDashboard(data);
    resultBatch.classList.remove("hidden");
  } catch (err) {
    showToast(`Error: ${err.message}`);
  } finally {
    loadingBatch.classList.add("hidden");
  }
}

function renderBatchDashboard(data) {
  const results = data.results || [];
  const totalSent = data.total_sentences || results.length;
  const totalTokens = data.total_tokens || 0;
  const avgCmi = data.average_cmi || 0.0;
  const aggSummary = data.aggregate_summary || { HI: 0, EN: 0 };

  // KPI cards
  kpiSentences.textContent = totalSent;
  kpiTokens.textContent = `${totalTokens} total words`;
  kpiAvgCmi.textContent = `${avgCmi}%`;

  if (avgCmi === 0) {
    kpiCmiLevel.textContent = "Single language";
  } else if (avgCmi < 20) {
    kpiCmiLevel.textContent = "Light mixing";
  } else if (avgCmi < 35) {
    kpiCmiLevel.textContent = "Moderate mixing";
  } else {
    kpiCmiLevel.textContent = "Heavy mixing";
  }

  kpiDominantLang.textContent = data.batch_dominant_language || "Hindi";
  kpiLangRatio.textContent = `${aggSummary["HI"] || 0} Hindi · ${aggSummary["EN"] || 0} English`;

  // Filter counts
  filterCntAll.textContent = totalSent;
  filterCntMixed.textContent = results.filter(r => r.cmi > 0).length;
  filterCntMono.textContent = results.filter(r => r.cmi === 0).length;
  filterCntHi.textContent = results.filter(r => r.dominant_language === "Hindi").length;
  filterCntEn.textContent = results.filter(r => r.dominant_language === "English").length;

  renderBatchList();
}

function renderBatchList() {
  if (!currentBatchData || !currentBatchData.results) return;
  batchListContainer.innerHTML = "";

  let items = currentBatchData.results.map((item, originalIndex) => ({ ...item, originalIndex }));

  // Filter
  if (activeBatchFilter === "mixed") {
    items = items.filter(r => r.cmi > 0);
  } else if (activeBatchFilter === "mono") {
    items = items.filter(r => r.cmi === 0);
  } else if (activeBatchFilter === "hi") {
    items = items.filter(r => r.dominant_language === "Hindi");
  } else if (activeBatchFilter === "en") {
    items = items.filter(r => r.dominant_language === "English");
  }

  // Search
  if (batchSearchQuery) {
    items = items.filter(r => {
      const matchText = r.text.toLowerCase().includes(batchSearchQuery);
      const matchToken = (r.tokens || []).some(t => t.toLowerCase().includes(batchSearchQuery));
      return matchText || matchToken;
    });
  }

  if (items.length === 0) {
    batchListContainer.innerHTML = `<div style="text-align: center; color: var(--text-muted); padding: 1.5rem;">No sentences match your search or filter.</div>`;
    return;
  }

  items.forEach(item => {
    const card = document.createElement("div");
    card.className = "batch-item-card";

    // Word pills
    const wordsHtml = (item.tokens || []).map((tok, i) => {
      const tag = item.tags[i] || "OTHER";
      const info = TAG_INFO[tag] || TAG_INFO["OTHER"];
      return `
        <span class="token-chip ${info.class}">
          <span class="token-chip-word">${escapeHtml(tok)}</span>
          <span class="token-chip-tag">${tag}</span>
        </span>
      `;
    }).join("");

    const mainLangClass = item.dominant_language === "English" ? "lang-badge-en" : "lang-badge-hi";
    const cmiPillClass = item.cmi === 0 ? "cmi-pill-mono" : "";

    card.innerHTML = `
      <div class="batch-item-header">
        <div class="batch-item-title">
          <span class="batch-idx">#${item.originalIndex + 1}</span>
          <span class="batch-text">${escapeHtml(item.text)}</span>
        </div>
        <div class="batch-item-meta">
          <span class="lang-badge ${mainLangClass}">${item.dominant_language || 'Hindi'}</span>
          <span class="cmi-pill-badge ${cmiPillClass}">${item.cmi}% CMI</span>
        </div>
      </div>
      <div class="batch-words-row">
        ${wordsHtml}
      </div>
    `;

    batchListContainer.appendChild(card);
  });
}

// -----------------------------------------------------------------------------
// CSV Export
// -----------------------------------------------------------------------------
function exportBatchCsv() {
  if (!currentBatchData || !currentBatchData.results || currentBatchData.results.length === 0) {
    showToast("No batch data to export.");
    return;
  }

  const results = currentBatchData.results;
  const headers = [
    "Index",
    "Sentence",
    "Word_Count",
    "Mixing_Score_CMI",
    "Main_Language",
    "Hindi_Words",
    "English_Words",
    "Tagged_Sequence"
  ];

  const rows = results.map((item, idx) => {
    const summary = item.summary || {};
    const sequence = (item.tokens || []).map((t, i) => `${t}/${item.tags[i]}`).join(" ");

    return [
      idx + 1,
      `"${item.text.replace(/"/g, '""')}"`,
      item.tokens.length,
      item.cmi,
      `"${item.dominant_language || 'Hindi'}"`,
      summary["HI"] || 0,
      summary["EN"] || 0,
      `"${sequence.replace(/"/g, '""')}"`
    ].join(",");
  });

  const csvContent = "\uFEFF" + [headers.join(","), ...rows].join("\r\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `hinglish_batch_results_${new Date().toISOString().slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  showToast(`Exported ${results.length} sentences to CSV`);
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------
function showToast(msg) {
  toastMsg.textContent = msg;
  toastMsg.classList.add("show");
  setTimeout(() => toastMsg.classList.remove("show"), 2500);
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/[&<>"']/g, m => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  }[m]));
}
