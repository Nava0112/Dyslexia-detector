import { LLM_CONFIG } from "./llm-config.js";
import { useDyslexiaGuidance } from "./useDyslexiaGuidance.js";

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  }[character]));
}

function formatGuidance(markdown) {
  if (typeof window.marked === "undefined" || typeof window.DOMPurify === "undefined") {
    return `<p>${escapeHtml(markdown).replace(/\n\n/g, "</p><p>").replace(/\n/g, "<br>")}</p>`;
  }
  const rendered = window.marked.parse(markdown, {
    headerIds: false,
    mangle: false,
    breaks: true,
  });
  return window.DOMPurify.sanitize(rendered, {
    ALLOWED_TAGS: ["h2", "h3", "h4", "p", "ol", "ul", "li", "strong", "em", "br"],
    ALLOWED_ATTR: [],
  });
}

export class LlmGuidancePanel {
  constructor(element) {
    this.element = element;
    this.guidance = null;
  }

  connect(analysisResponse) {
    this.guidance = useDyslexiaGuidance(analysisResponse);
    this.unsubscribe = this.guidance.subscribe((state) => this.render(state));
    this.guidance.generate();
  }

  render(state) {
    if (state.status === "loading") {
      this.element.innerHTML = '<div class="guidance-state"><span class="guidance-spinner"></span><p>Preparing your personalized guidance...</p></div>';
      return;
    }
    if (state.status === "error") {
      this.element.innerHTML = `<div class="guidance-state guidance-error"><p>${escapeHtml(state.error)}</p><button class="secondary-button guidance-retry" type="button">Try again</button><small>Your screening results remain available above.</small></div>`;
      this.element.querySelector(".guidance-retry").addEventListener("click", () => this.guidance.generate());
      return;
    }
    if (state.status !== "success") return;
    const preferences = this.guidance.getPreferences();
    this.element.innerHTML = `
      <div class="guidance-toolbar">
        <label>Age group <select class="guidance-age">${LLM_CONFIG.ageGroups.map((item) => `<option>${item}</option>`).join("")}</select></label>
        <label>Difficulty <select class="guidance-difficulty">${LLM_CONFIG.difficulties.map((item) => `<option>${item}</option>`).join("")}</select></label>
        <button class="secondary-button guidance-regenerate" type="button">Regenerate exercises</button>
      </div>
      <div class="guidance-content">${formatGuidance(state.content)}</div>
      <p class="guidance-privacy">Only numerical screening metrics are used for guidance. Your handwriting image is never sent to the language model.</p>`;
    this.element.querySelector(".guidance-age").value = preferences.ageGroup;
    this.element.querySelector(".guidance-difficulty").value = preferences.difficulty;
    this.element.querySelector(".guidance-age").addEventListener("change", (event) => this.guidance.setAgeGroup(event.target.value));
    this.element.querySelector(".guidance-difficulty").addEventListener("change", (event) => this.guidance.setDifficulty(event.target.value));
    this.element.querySelector(".guidance-regenerate").addEventListener("click", () => this.guidance.generate());
  }
}
