export function useDyslexiaGuidance(analysisResponse) {
  let state = { status: "idle", content: "", error: "" };
  let listeners = [];
  let ageGroup = "Child";
  let difficulty = "Standard";

  const notify = () => listeners.forEach((listener) => listener({ ...state }));
  const subscribe = (listener) => {
    listeners.push(listener);
    listener({ ...state });
    return () => {
      listeners = listeners.filter((item) => item !== listener);
    };
  };

  const generate = async () => {
    state = { status: "loading", content: "", error: "" };
    notify();
    const counts = analysisResponse.class_counts || {};
    const summary = {
      status: analysisResponse.status,
      screening_score: analysisResponse.screening_score,
      activeThreshold: analysisResponse.decision_threshold,
      crop_count: analysisResponse.crop_count,
      line_count: analysisResponse.line_count,
      class_counts: {
        Normal: Number(counts.Normal || 0),
        Corrected: Number(counts.Corrected || 0),
        Reversal: Number(counts.Reversal || 0),
      },
      age_group: ageGroup,
      difficulty,
    };
    try {
      const response = await fetch("/guidance", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(summary),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Guidance could not be generated.");
      state = { status: "success", content: result.content, error: "" };
    } catch (error) {
      state = {
        status: "error",
        content: "",
        error: error.message || "Guidance could not be generated.",
      };
    }
    notify();
  };

  return {
    subscribe,
    generate,
    setAgeGroup(value) {
      ageGroup = value;
      generate();
    },
    setDifficulty(value) {
      difficulty = value;
      generate();
    },
    getPreferences: () => ({ ageGroup, difficulty }),
  };
}
