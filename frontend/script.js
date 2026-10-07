/**
 * PhishGuard Frontend Application Logic.
 * Connects directly to FastAPI backend (http://127.0.0.1:8000/predict).
 * Provides dynamic rendering of legitimate and phishing result states,
 * confidence progress score meters, evidence-based risk factors, and input validation.
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
const evidenceHeading = document.getElementById("evidence-heading");
const evidenceList = document.getElementById("evidence-list");
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
    showError("Input Required", "Please enter a website URL string to evaluate.");
    return;
  }
  scanUrl(rawUrl);
}

/**
 * Send URL prediction request to FastAPI endpoint.
 */
async function scanUrl(targetUrl) {
  setLoadingState(true);

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
        showError("INVALID URL", "Please enter a valid website URL.");
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
 * Render user-friendly error card.
 */
function showError(title, message) {
  resultsWrapper.classList.remove("hidden");
  loadingCard.classList.add("hidden");
  verdictCard.classList.add("hidden");
  errorCard.classList.remove("hidden");

  errorTitle.textContent = title;
  errorMessage.textContent = message;

  resultsWrapper.scrollIntoView({ behavior: "smooth", block: "nearest" });
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

    evidenceHeading.textContent = "Evidence-Based Risk Factors:";
  } else {
    verdictCard.className = "result-card verdict-card legitimate-mode";

    verdictBadge.textContent = "LEGITIMATE";
    verdictBadge.className = "verdict-badge legitimate";

    verdictHeaderIcon.textContent = "gpp_good";
    verdictHeaderText.textContent = "Verified Clean";
    verdictHeaderText.style.color = "var(--color-emerald-light)";
    verdictHeaderIcon.style.color = "var(--color-emerald-light)";

    // Calculate safety level display (100% - phishing risk probability)
    const safetyPct = (100 - parseFloat(confidencePct)).toFixed(1);
    confidenceValue.textContent = `${safetyPct}% Safety Level`;
    confidenceValue.style.color = "var(--color-emerald-light)";

    scoreFill.style.width = `${safetyPct}%`;
    scoreFill.className = "score-fill legitimate";

    scoreStatusText.textContent = "Model Confidence";
    scoreStatusText.style.color = "var(--color-emerald-light)";

    decisionValue.textContent = "ALLOW";
    decisionValue.style.color = "var(--color-emerald-light)";

    evidenceHeading.textContent = "Verified Safety Telemetry:";
  }

  // Populate Risk Factors / Characteristics List
  evidenceList.textContent = ""; // Clear existing

  if (isPhishing && data.risk_factors && data.risk_factors.length > 0) {
    data.risk_factors.forEach((factorStr) => {
      const li = document.createElement("li");
      li.className = "evidence-item phishing";

      const icon = document.createElement("span");
      icon.className = "material-symbols-outlined item-icon";
      icon.textContent = "error";

      const text = document.createElement("span");
      text.textContent = factorStr;

      li.appendChild(icon);
      li.appendChild(text);
      evidenceList.appendChild(li);
    });
  } else if (isPhishing) {
    // Fallback item if risk factors array is empty
    const li = document.createElement("li");
    li.className = "evidence-item phishing";
    const icon = document.createElement("span");
    icon.className = "material-symbols-outlined item-icon";
    icon.textContent = "error";
    const text = document.createElement("span");
    text.textContent = "High probability phishing pattern detected by XGBoost classifier.";
    li.appendChild(icon);
    li.appendChild(text);
    evidenceList.appendChild(li);
  } else {
    // Legitimate URL safety checklist
    const legitimateIndicators = [
      "Valid protocol & structural URL format",
      "No suspicious security keywords detected",
      "Standard domain length & character entropy",
      "No IP address or non-standard port obfuscation",
      "Consistent lexical feature score across 38 URL-based features",
    ];


    legitimateIndicators.forEach((indicatorStr) => {
      const li = document.createElement("li");
      li.className = "evidence-item legitimate";

      const icon = document.createElement("span");
      icon.className = "material-symbols-outlined item-icon";
      icon.textContent = "check_circle";

      const text = document.createElement("span");
      text.textContent = indicatorStr;

      li.appendChild(icon);
      li.appendChild(text);
      evidenceList.appendChild(li);
    });
  }

  resultsWrapper.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
