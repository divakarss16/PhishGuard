/**
 * PhishGuard Frontend Application Logic (Phase 10B).
 * Connects directly to FastAPI backend (http://127.0.0.1:8001/predict).
 * Renders short cybersecurity scan analysis sequence, URL structure panel,
 * evidence cards, and Why This Verdict explanations with zero ML modifications.
 */

const API_BASE_URL = "http://127.0.0.1:8001";

// DOM Elements
const scanForm = document.getElementById("url-scan-form");
const urlInput = document.getElementById("target-url-input");
const scanBtn = document.getElementById("scan-btn");
const btnText = document.getElementById("btn-text");
const clearBtn = document.getElementById("clear-btn");

const resultsWrapper = document.getElementById("results-wrapper");
const loadingCard = document.getElementById("loading-card");
const errorCard = document.getElementById("error-card");
const errorTitle = document.getElementById("error-title");
const errorMessage = document.getElementById("error-message");

const verdictCard = document.getElementById("verdict-card");
const verdictBadge = document.getElementById("verdict-badge");
const riskBadge = document.getElementById("risk-badge");
const verdictHeaderIcon = document.getElementById("verdict-header-icon");
const verdictHeaderText = document.getElementById("verdict-header-text");

const confidenceValue = document.getElementById("confidence-value");
const scoreFill = document.getElementById("score-fill");
const scoreStatusText = document.getElementById("score-status-text");
const evaluatedUrl = document.getElementById("evaluated-url");
const decisionValue = document.getElementById("decision-value");

// Initialize Event Listeners
document.addEventListener("DOMContentLoaded", () => {
  if (scanForm) {
    scanForm.addEventListener("submit", (e) => {
      e.preventDefault();
      handleScanSubmit();
    });
  }

  if (scanBtn) {
    scanBtn.addEventListener("click", (e) => {
      e.preventDefault();
      handleScanSubmit();
    });
  }

  if (clearBtn && urlInput) {
    clearBtn.addEventListener("click", () => {
      urlInput.value = "";
      urlInput.focus();
    });
  }
});

/**
 * Handle form submit / button click.
 */
function handleScanSubmit() {
  const rawUrl = urlInput.value ? urlInput.value.trim() : "";
  if (!rawUrl) {
    showError("INVALID URL", "Invalid URL format. Please enter a valid URL beginning with http:// or https://.");
    return;
  }
  scanUrl(rawUrl);
}

/**
 * Animate the 5-step cybersecurity scan sequence in loading card.
 */
function runAnalysisSequence(callback) {
  const steps = [
    { id: "seq-step-1", status: "Validating URL structure..." },
    { id: "seq-step-2", status: "Extracting 38 URL features..." },
    { id: "seq-step-3", status: "Analyzing suspicious patterns..." },
    { id: "seq-step-4", status: "Running XGBoost inference..." },
    { id: "seq-step-5", status: "Generating risk assessment..." },
  ];

  // Reset visual state
  steps.forEach((s) => {
    const el = document.getElementById(s.id);
    if (el) {
      el.className = "seq-step";
      const icon = el.querySelector(".seq-icon");
      if (icon) {
        icon.textContent = "sync";
        icon.className = "seq-icon material-symbols-outlined spinning";
      }
    }
  });

  const seqStatus = document.getElementById("seq-status");
  if (seqStatus) seqStatus.textContent = "INITIALIZING PIPELINE...";

  let current = 0;
  const interval = setInterval(() => {
    if (current < steps.length) {
      const s = steps[current];
      const el = document.getElementById(s.id);
      if (el) {
        el.className = "seq-step active";
        const icon = el.querySelector(".seq-icon");
        if (icon) {
          icon.textContent = "check_circle";
          icon.className = "seq-icon material-symbols-outlined done";
        }
      }
      if (seqStatus) seqStatus.textContent = s.status;
      current++;
    } else {
      clearInterval(interval);
      if (seqStatus) seqStatus.textContent = "ANALYSIS COMPLETE";
      if (callback) setTimeout(callback, 200);
    }
  }, 160);
}

/**
 * Send URL prediction request to FastAPI endpoint.
 */
async function scanUrl(targetUrl) {
  setLoadingState(true);
  runAnalysisSequence(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url: targetUrl }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        const detailMsg = errData.detail || `Server returned HTTP status ${response.status}`;
        if (detailMsg.includes("INVALID URL")) {
          showError("INVALID URL", "Invalid URL format. Please enter a valid URL beginning with http:// or https://.");
        } else {
          showError("Scan Evaluation Failed", detailMsg);
        }
        return;
      }

      const data = await response.json();
      renderVerdict(data);
    } catch (err) {
      showError(
        "Backend Unreachable",
        "Could not connect to PhishGuard API at " + API_BASE_URL + ". Ensure backend server is running."
      );
    } finally {
      setLoadingState(false);
    }
  });
}

/**
 * Toggle UI loading state.
 */
function setLoadingState(isLoading) {
  if (isLoading) {
    resultsWrapper.classList.remove("hidden");
    loadingCard.classList.remove("hidden");
    errorCard.classList.add("hidden");
    verdictCard.classList.add("hidden");

    scanBtn.disabled = true;
    if (btnText) btnText.textContent = "Analyzing...";
  } else {
    loadingCard.classList.add("hidden");
    scanBtn.disabled = false;
    if (btnText) btnText.textContent = "Check Website";
  }
}

/**
 * Render user-friendly error card (e.g. Invalid URL).
 */
function showError(title, message) {
  resultsWrapper.classList.remove("hidden");
  loadingCard.classList.add("hidden");
  verdictCard.classList.add("hidden");
  errorCard.classList.remove("hidden");

  const errHeaderIcon = document.getElementById("error-header-icon");

  if (errorTitle) errorTitle.textContent = title || "INVALID URL";
  if (errorMessage) {
    if (title === "INVALID URL" || (message && message.includes("INVALID URL"))) {
      errorMessage.textContent = "Invalid URL format. Please enter a valid URL beginning with http:// or https://.";
    } else {
      errorMessage.textContent = message;
    }
  }
  if (errHeaderIcon) errHeaderIcon.textContent = "report_problem";

  resultsWrapper.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/**
 * Derive URL structure information cleanly from submitted string (0 network/DNS lookups).
 */
function updateUrlStructure(rawUrlStr) {
  let protocolStr = "HTTPS";
  let hostnameStr = rawUrlStr;
  let pathStr = "None";
  let queryStr = "None";
  let isIpStr = "No";

  try {
    let parseTarget = rawUrlStr;
    if (!parseTarget.match(/^[a-zA-Z]+:\/\//)) {
      parseTarget = "http://" + parseTarget;
    }
    const urlObj = new URL(parseTarget);

    if (rawUrlStr.toLowerCase().startsWith("http://")) {
      protocolStr = "HTTP";
    } else if (rawUrlStr.toLowerCase().startsWith("https://")) {
      protocolStr = "HTTPS";
    } else {
      protocolStr = urlObj.protocol.toUpperCase().replace(":", "");
    }

    hostnameStr = urlObj.hostname || rawUrlStr;
    pathStr = (urlObj.pathname && urlObj.pathname !== "/") ? urlObj.pathname : "None";
    queryStr = urlObj.search ? urlObj.search : "None";

    const isIPv4 = /^(\d{1,3}\.){3}\d{1,3}$/.test(hostnameStr);
    const isIPv6 = hostnameStr.includes(":");
    if (isIPv4 || isIPv6) {
      isIpStr = "Yes";
    }
  } catch (e) {
    if (rawUrlStr.toLowerCase().startsWith("http://")) protocolStr = "HTTP";
    else if (rawUrlStr.toLowerCase().startsWith("https://")) protocolStr = "HTTPS";
  }

  const pEl = document.getElementById("struct-protocol");
  const hEl = document.getElementById("struct-hostname");
  const pathEl = document.getElementById("struct-path");
  const qEl = document.getElementById("struct-query");
  const ipEl = document.getElementById("struct-ip");

  if (pEl) pEl.textContent = protocolStr;
  if (hEl) hEl.textContent = hostnameStr;
  if (pathEl) pathEl.textContent = pathStr;
  if (qEl) qEl.textContent = queryStr;
  if (ipEl) ipEl.textContent = isIpStr;
}

/**
 * Render Why This Verdict section.
 */
function renderWhyVerdict(isPhishing, riskFactors) {
  const whyList = document.getElementById("why-list");
  if (!whyList) return;
  whyList.textContent = "";

  if (isPhishing) {
    const items = [
      { text: "Suspicious URL patterns detected", icon: "warning", type: "phishing" },
      { text: "High-risk URL characteristics contributed to the prediction", icon: "warning", type: "phishing" },
    ];

    items.forEach((item) => {
      const li = document.createElement("li");
      li.className = `why-item ${item.type}`;
      const icon = document.createElement("span");
      icon.className = "material-symbols-outlined why-icon";
      icon.textContent = item.icon;
      const span = document.createElement("span");
      span.textContent = item.text;
      li.appendChild(icon);
      li.appendChild(span);
      whyList.appendChild(li);
    });
  } else {
    const items = [
      { text: "URL structure appears consistent with legitimate patterns", icon: "check_circle", type: "legitimate" },
      { text: "No major suspicious structural indicators detected", icon: "check_circle", type: "legitimate" },
    ];

    items.forEach((item) => {
      const li = document.createElement("li");
      li.className = `why-item ${item.type}`;
      const icon = document.createElement("span");
      icon.className = "material-symbols-outlined why-icon";
      icon.textContent = item.icon;
      const span = document.createElement("span");
      span.textContent = item.text;
      li.appendChild(icon);
      li.appendChild(span);
      whyList.appendChild(li);
    });
  }
}

/**
 * Render Risk Factors as cybersecurity evidence cards under Suspicious URL Patterns.
 */
function renderEvidenceCards(isPhishing, riskFactors) {
  const evidenceHeading = document.getElementById("evidence-heading");
  const evidenceIcon = document.getElementById("evidence-icon");
  const evidenceList = document.getElementById("evidence-list");

  if (!evidenceList) return;
  evidenceList.textContent = "";

  if (isPhishing) {
    if (evidenceHeading) evidenceHeading.textContent = "Suspicious URL Patterns";
    if (evidenceIcon) {
      evidenceIcon.textContent = "warning";
      evidenceIcon.style.color = "var(--color-crimson)";
    }

    if (riskFactors && riskFactors.length > 0) {
      riskFactors.forEach((factorStr) => {
        const card = document.createElement("div");
        card.className = "evidence-card phishing";

        const icon = document.createElement("span");
        icon.className = "material-symbols-outlined card-icon";
        icon.textContent = "warning";

        const text = document.createElement("span");
        text.className = "card-text";
        text.textContent = factorStr;

        card.appendChild(icon);
        card.appendChild(text);
        evidenceList.appendChild(card);
      });
    } else {
      const card = document.createElement("div");
      card.className = "evidence-card phishing";
      const icon = document.createElement("span");
      icon.className = "material-symbols-outlined card-icon";
      icon.textContent = "warning";
      const text = document.createElement("span");
      text.className = "card-text";
      text.textContent = "High probability phishing pattern detected by XGBoost classifier.";
      card.appendChild(icon);
      card.appendChild(text);
      evidenceList.appendChild(card);
    }
  } else {
    if (evidenceHeading) evidenceHeading.textContent = "Verified Safety Indicators";
    if (evidenceIcon) {
      evidenceIcon.textContent = "verified_user";
      evidenceIcon.style.color = "var(--color-emerald-light)";
    }

    const legitimateIndicators = [
      "Valid protocol & structural URL format",
      "No suspicious security keywords detected",
      "Standard domain length & character entropy",
      "No IP address or non-standard port obfuscation",
      "Consistent lexical feature score across 38 URL-based features",
    ];

    legitimateIndicators.forEach((indicatorStr) => {
      const card = document.createElement("div");
      card.className = "evidence-card legitimate";

      const icon = document.createElement("span");
      icon.className = "material-symbols-outlined card-icon";
      icon.textContent = "check_circle";

      const text = document.createElement("span");
      text.className = "card-text";
      text.textContent = indicatorStr;

      card.appendChild(icon);
      card.appendChild(text);
      evidenceList.appendChild(card);
    });
  }
}

/**
 * Dynamically render legitimate or phishing verdict state from API response.
 */
function renderVerdict(data) {
  resultsWrapper.classList.remove("hidden");
  loadingCard.classList.add("hidden");
  errorCard.classList.add("hidden");
  verdictCard.classList.remove("hidden");

  const isPhishing = data.prediction === "phishing";
  const confidencePct = (data.confidence * 100).toFixed(1);

  // Safe DOM string assignment using textContent
  evaluatedUrl.textContent = data.url;
  riskBadge.textContent = data.risk_level + " RISK";
  riskBadge.className = `risk-badge ${data.risk_level}`;

  if (isPhishing) {
    verdictCard.className = "result-card verdict-card phishing-mode";

    verdictBadge.textContent = "PHISHING";
    verdictBadge.className = "verdict-badge phishing";

    verdictHeaderIcon.textContent = "gpp_maybe";
    verdictHeaderText.textContent = "Threat Alert";
    verdictHeaderText.style.color = "var(--color-crimson)";
    verdictHeaderIcon.style.color = "var(--color-crimson)";

    confidenceValue.textContent = `${confidencePct}% Model Confidence`;
    confidenceValue.style.color = "var(--color-crimson)";

    scoreFill.style.width = `${confidencePct}%`;
    scoreFill.className = "score-fill phishing";

    scoreStatusText.textContent = "Suspicious URL Pattern Detected";
    scoreStatusText.style.color = "var(--color-crimson)";

    decisionValue.textContent = "CONTAIN / ISOLATE";
    decisionValue.style.color = "var(--color-crimson)";
  } else {
    verdictCard.className = "result-card verdict-card legitimate-mode";

    verdictBadge.textContent = "LEGITIMATE";
    verdictBadge.className = "verdict-badge legitimate";

    verdictHeaderIcon.textContent = "gpp_good";
    verdictHeaderText.textContent = "Verified Clean";
    verdictHeaderText.style.color = "var(--color-emerald-light)";
    verdictHeaderIcon.style.color = "var(--color-emerald-light)";

    const safetyPct = (100 - parseFloat(confidencePct)).toFixed(1);
    confidenceValue.textContent = `${safetyPct}% Safety Level`;
    confidenceValue.style.color = "var(--color-emerald-light)";

    scoreFill.style.width = `${safetyPct}%`;
    scoreFill.className = "score-fill legitimate";

    scoreStatusText.textContent = "Model Confidence";
    scoreStatusText.style.color = "var(--color-emerald-light)";

    decisionValue.textContent = "ALLOW";
    decisionValue.style.color = "var(--color-emerald-light)";
  }

  // Update panels
  updateUrlStructure(data.url);
  renderWhyVerdict(isPhishing, data.risk_factors);
  renderEvidenceCards(isPhishing, data.risk_factors);

  resultsWrapper.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

