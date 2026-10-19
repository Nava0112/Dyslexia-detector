const $ = (selector) => document.querySelector(selector);
const fileInput = $("#file-input");
const dropzone = $("#dropzone");
const browseButton = $("#browse-button");
const analyzeButton = $("#analyze-button");
const formError = $("#form-error");
const preview = $("#file-preview");
const serviceStatus = $("#service-status");
let selectedFile = null;

function showError(message) {
  formError.textContent = message;
  formError.classList.remove("hidden");
}

function clearError() {
  formError.textContent = "";
  formError.classList.add("hidden");
}

function formatBytes(bytes) {
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function setFile(file) {
  clearError();
  if (!file) return;
  const allowed = ["image/jpeg", "image/png", "image/webp"];
  if (!allowed.includes(file.type)) {
    showError("Please choose a JPG, PNG, or WebP image.");
    return;
  }
  if (file.size > 10 * 1024 * 1024) {
    showError("This image is larger than the 10 MB limit.");
    return;
  }
  selectedFile = file;
  $("#file-name").textContent = file.name;
  $("#file-size").textContent = formatBytes(file.size);
  $("#preview-image").src = URL.createObjectURL(file);
  preview.classList.remove("hidden");
  dropzone.classList.add("hidden");
  analyzeButton.disabled = false;
}

function removeFile() {
  selectedFile = null;
  fileInput.value = "";
  $("#preview-image").removeAttribute("src");
  preview.classList.add("hidden");
  dropzone.classList.remove("hidden");
  analyzeButton.disabled = true;
  clearError();
}

function setService(state, text) {
  serviceStatus.classList.remove("ready", "error");
  if (state) serviceStatus.classList.add(state);
  serviceStatus.querySelector("span:last-child").textContent = text;
}

async function checkService() {
  try {
    const response = await fetch("/ready");
    if (response.ok) setService("ready", "Model ready");
    else setService("error", "Model checkpoint unavailable");
  } catch {
    setService("error", "API unavailable");
  }
}

function setLoading(loading) {
  analyzeButton.disabled = loading || !selectedFile;
  analyzeButton.querySelector("span:first-child").textContent = loading ? "Analyzing sample..." : "Analyze handwriting";
  analyzeButton.querySelector(".button-arrow").textContent = loading ? "…" : "→";
}

function renderResults(data) {
  $("#results").classList.remove("hidden");
  $("#job-id").textContent = `ID ${data.job_id.slice(0, 8)}`;
  $("#crop-count").textContent = `${data.crop_count} crops`;
  $("#line-count").textContent = data.line_count;
  $("#processing-time").textContent = `${Math.round(data.processing_time_ms)} ms`;
  $("#result-disclaimer").textContent = data.disclaimer;
  const counts = data.class_counts || {};
  const total = Math.max(data.crop_count, 1);
  ["corrected", "normal", "reversal"].forEach((name) => {
    const count = counts[name[0].toUpperCase() + name.slice(1)] || 0;
    $(`#${name}-count`).textContent = count;
    $(`#${name}-bar`).style.width = `${(count / total) * 100}%`;
  });
  const score = data.screening_score;
  const scorePercent = score === null ? 0 : Math.round(score * 100);
  $("#score-value").textContent = score === null ? "—" : `${scorePercent}%`;
  $("#score-ring").style.setProperty("--progress", `${scorePercent}%`);
  $("#threshold-fill").style.width = `${scorePercent}%`;
  $("#threshold-marker").style.left = `${data.decision_threshold * 100}%`;
  const elevated = data.dyslexic === true;
  $("#result-pill").textContent = data.status === "insufficient_data" ? "Review needed" : (elevated ? "Elevated risk" : "Lower risk");
  $("#result-pill").classList.toggle("elevated", elevated);
  $("#result-title").textContent = data.status === "insufficient_data" ? "Not enough signal yet" : (elevated ? "Patterns warrant review" : "No elevated pattern signal");
  $("#result-copy").textContent = data.status === "insufficient_data" ? `At least 5 usable regions are needed. We found ${data.crop_count}.` : "This score is a screening indicator, not a diagnosis.";
  const imageUrl = data.artifacts && data.artifacts.annotated_image;
  if (imageUrl) {
    const absoluteUrl = `${imageUrl}?t=${Date.now()}`;
    $("#annotated-image").src = absoluteUrl;
    $("#annotated-image").classList.remove("hidden");
    $("#image-placeholder").classList.add("hidden");
    $("#download-image").href = imageUrl;
    $("#download-image").classList.remove("disabled");
  }
  $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function analyze() {
  if (!selectedFile) return;
  clearError();
  setLoading(true);
  const body = new FormData();
  body.append("file", selectedFile);
  try {
    const response = await fetch("/analyze", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The analysis could not be completed.");
    renderResults(data);
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false);
  }
}

browseButton.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("click", (event) => {
  if (event.target !== browseButton) fileInput.click();
});
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") fileInput.click();
});
fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
$("#remove-file").addEventListener("click", removeFile);
analyzeButton.addEventListener("click", analyze);
$("#new-analysis").addEventListener("click", () => {
  removeFile();
  $("#results").classList.add("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
});
["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
}));
["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
}));
dropzone.addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));
checkService();
