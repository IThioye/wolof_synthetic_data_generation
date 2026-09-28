const state = { record: null };

const els = {
  form: document.querySelector("#annotation-form"),
  loading: document.querySelector("#loading"),
  empty: document.querySelector("#empty-state"),
  saved: document.querySelector("#saved-count"),
  remaining: document.querySelector("#remaining-count"),
  total: document.querySelector("#total-count"),
  progress: document.querySelector("#progress-bar"),
  covered: document.querySelector("#covered-count"),
  missing: document.querySelector("#missing-count"),
  unique: document.querySelector("#unique-count"),
  lookupProgress: document.querySelector("#lookup-progress-bar"),
  position: document.querySelector("#position-input"),
  recordNumber: document.querySelector("#record-number"),
  lookupBadge: document.querySelector("#lookup-badge"),
  lookupEvidence: document.querySelector("#lookup-evidence"),
  lookupEvidenceStatus: document.querySelector("#lookup-evidence-status"),
  lookupPosCandidates: document.querySelector("#lookup-pos-candidates"),
  lookupPosLabels: document.querySelector("#lookup-pos-labels"),
  lookupLemmas: document.querySelector("#lookup-lemmas"),
  lookupTranslations: document.querySelector("#lookup-translations"),
  lexicalHelp: document.querySelector("#lexical-help"),
  existingBadge: document.querySelector("#existing-badge"),
  videoLink: document.querySelector("#video-link"),
  comment: document.querySelector("#comment-text"),
  formalSentence: document.querySelector("#formal-sentence"),
  informalToken: document.querySelector("#informal-token"),
  formalToken: document.querySelector("#formal-token"),
  automaticCategory: document.querySelector("#automatic-category"),
  lemma: document.querySelector("#lemma"),
  pos: document.querySelector("#pos"),
  sourceLanguage: document.querySelector("#source-language"),
  targetLanguage: document.querySelector("#target-language"),
  entityType: document.querySelector("#entity-type"),
  protected: document.querySelector("#protected"),
  errorTypes: document.querySelector("#error-types"),
  transformationSteps: document.querySelector("#transformation-steps"),
  reviewStatus: document.querySelector("#review-status"),
  notes: document.querySelector("#notes"),
  reusable: document.querySelector("#lexical-reusable"),
  saveButton: document.querySelector("#save-button"),
  saveMessage: document.querySelector("#save-message"),
};

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error || `Request failed (${response.status})`);
  return body;
}

function updateProgress(progress) {
  if (!progress) return;
  els.saved.textContent = progress.saved.toLocaleString();
  els.remaining.textContent = progress.remaining.toLocaleString();
  els.total.textContent = `${progress.total.toLocaleString()} gold-train token occurrences`;
  els.progress.style.width = `${progress.total ? 100 * progress.saved / progress.total : 0}%`;
  els.covered.textContent = progress.lookup_covered.toLocaleString();
  els.missing.textContent = progress.lookup_missing.toLocaleString();
  els.unique.textContent = `${progress.unique_formal_tokens.toLocaleString()} unique formal units`;
  els.lookupProgress.style.width = `${progress.unique_formal_tokens ? 100 * progress.lookup_covered / progress.unique_formal_tokens : 0}%`;
  els.position.max = Math.max(progress.total - 1, 0);
}

function populateSelect(select, options, selected) {
  select.replaceChildren(...options.map((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    option.selected = value === selected;
    return option;
  }));
}

function applyPersonEntityDefaults() {
  if (els.entityType.value !== "PERSON") return;
  const properNounOption = [...els.pos.options].find((option) => option.value === "PROPN");
  if (properNounOption) els.pos.value = "PROPN";
  const protectedOption = [...els.protected.options].find((option) => option.value === "yes");
  if (protectedOption) els.protected.value = "yes";
}

function renderErrorTypes(options, selectedValues, definitions) {
  const selected = new Set(selectedValues || []);
  els.errorTypes.replaceChildren(...options.map((value) => {
    const label = document.createElement("label");
    const input = document.createElement("input");
    const text = document.createElement("span");
    input.type = "checkbox";
    input.value = value;
    input.checked = selected.has(value);
    input.addEventListener("change", () => {
      const all = [...els.errorTypes.querySelectorAll('input[type="checkbox"]')];
      if (value === "identity" && input.checked) {
        all.filter((item) => item !== input).forEach((item) => { item.checked = false; });
      } else if (value !== "identity" && input.checked) {
        const identity = all.find((item) => item.value === "identity");
        if (identity) identity.checked = false;
      }
    });
    text.textContent = value.replaceAll("_", " ");
    const definition = definitions?.[value] || "";
    label.title = definition;
    text.setAttribute("aria-label", definition ? `${value}: ${definition}` : value);
    label.append(input, text);
    return label;
  }));
}

function renderTransformations(steps) {
  if (!steps.length) {
    const item = document.createElement("li");
    item.textContent = "No character change (identity).";
    els.transformationSteps.replaceChildren(item);
    return;
  }
  const visibleText = (value) => {
    if (!value) return "empty";
    return value.replaceAll(" ", "[space]");
  };
  els.transformationSteps.replaceChildren(...steps.map((step) => {
    const item = document.createElement("li");
    const formal = visibleText(step.formal_text);
    const informal = visibleText(step.informal_text);
    if (step.operation === "replace") {
      item.textContent = `Replace "${formal}" with "${informal}"`;
    } else if (step.operation === "delete") {
      item.textContent = `Delete "${formal}"`;
    } else {
      item.textContent = `Insert "${informal}"`;
    }
    const position = document.createElement("small");
    position.textContent = `formal index ${step.formal_start}`;
    item.append(position);
    return item;
  }));
}

function renderLookupEvidence(evidence) {
  const hasEvidence = evidence && Object.keys(evidence).length > 0;
  els.lookupEvidence.classList.toggle("hidden", !hasEvidence);
  if (!hasEvidence) return;
  const entries = evidence.entry_count || "1";
  const ambiguity = [];
  if (evidence.language_ambiguous) ambiguity.push("language");
  if (evidence.ambiguous_pos) ambiguity.push("POS");
  els.lookupEvidenceStatus.textContent = ambiguity.length
    ? `${entries} dictionary entries - ambiguous ${ambiguity.join(" and ")}; manual selection required`
    : `${entries} dictionary entries - all senses share the displayed POS`;
  els.lookupPosCandidates.textContent = (evidence.pos_candidates || []).join(", ") || "No reliable automatic mapping";
  els.lookupPosLabels.textContent = (evidence.dictionary_pos || []).join("; ") || "Not provided";
  els.lookupLemmas.textContent = (evidence.lemmas || []).slice(0, 8).join("; ") || "Not provided";
  els.lookupTranslations.textContent = (evidence.translations || []).slice(0, 8).join("; ") || "Not provided";
}

function applyLanguageSpecificPosOptions() {
  if (!state.record) return;
  const evidence = state.record.prefill.lookup_evidence || {};
  if (!evidence.language_ambiguous) return;
  const language = els.targetLanguage.value;
  const candidates = (evidence.pos_candidates_by_language || {})[language] || [];
  if (candidates.length) {
    const selected = els.pos.value;
    populateSelect(els.pos, ["UNKNOWN", ...candidates], candidates.includes(selected) ? selected : "UNKNOWN");
    els.pos.disabled = false;
  }
  if (state.record.identity_lookup && ["wo", "fr"].includes(language)) {
    els.sourceLanguage.value = language;
  }
  if (language === "fr") els.protected.value = "yes";
  if (language === "wo" && els.protected.value === "uncertain") els.protected.value = "no";
}

function renderRecord(record) {
  state.record = record;
  els.loading.classList.add("hidden");
  els.empty.classList.add("hidden");
  els.form.classList.remove("hidden");
  els.saveMessage.textContent = "";
  els.saveMessage.classList.remove("error");
  els.position.value = record.position;
  els.recordNumber.textContent = `Source ${record.source_index} - token ${record.token_position} - position ${record.position}`;
  els.comment.textContent = record.comment;
  els.formalSentence.textContent = record.formal_sentence;
  els.informalToken.textContent = record.informal_token;
  els.formalToken.textContent = record.formal_token;
  els.automaticCategory.textContent = record.automatic_category || "none";
  els.videoLink.href = record.video_url || "#";
  els.videoLink.classList.toggle("hidden", !record.video_url);
  els.existingBadge.classList.toggle("hidden", !record.already_annotated);

  const prefill = record.prefill;
  const lookupLabel = prefill.lookup_source === "none"
    ? "No lookup entry - manual classification"
    : `Prefilled from ${prefill.lookup_source.replaceAll("_", " ")}`;
  els.lookupBadge.textContent = lookupLabel;
  renderLookupEvidence(prefill.lookup_evidence);
  els.lemma.value = prefill.lemma || "";
  populateSelect(els.pos, record.options.pos, prefill.pos);
  populateSelect(els.sourceLanguage, record.options.language, prefill.source_language);
  populateSelect(els.targetLanguage, record.options.language, prefill.target_language);
  populateSelect(els.entityType, record.options.entity, prefill.entity_type);
  populateSelect(els.protected, record.options.protected, prefill.protected);
  renderErrorTypes(record.options.error, prefill.error_types, record.options.error_definitions);
  renderTransformations(record.transformation_steps);
  populateSelect(els.reviewStatus, record.options.review, prefill.review_status);
  els.notes.value = prefill.notes || "";
  els.lemma.disabled = !record.lemma_editable;
  els.targetLanguage.disabled = !record.target_language_editable;
  els.entityType.disabled = !record.entity_editable;
  els.protected.disabled = !record.protected_editable;
  els.pos.disabled = !record.pos_editable;
  els.lexicalHelp.textContent = record.target_language_editable && record.lexical_fields_locked
    ? "This form occurs in both Wolof and French resources. Select its contextual language and POS; the choice will not become a global token rule."
    : record.lexical_fields_locked
    ? (record.pos_editable
      ? "Dictionary lexical fields are prefilled and locked. Select POS because this form is missing a unique dictionary POS."
      : "Dictionary lexical fields are prefilled and locked; review only the occurrence-specific properties below.")
    : "No lookup entry was found. Classify the lexical fields, then review the occurrence-specific properties.";
  const lookupNeedsClassification = !record.lexical_fields_locked
    || record.pos_editable
    || record.target_language_editable;
  els.reusable.checked = lookupNeedsClassification && Boolean(prefill.lexical_reusable);
  els.reusable.disabled = !lookupNeedsClassification || !record.reusable_allowed;
  els.sourceLanguage.disabled = record.source_language_locked;
  els.reviewStatus.disabled = record.identity_lookup && !record.target_language_editable && !record.pos_editable;
  [...els.errorTypes.querySelectorAll('input[type="checkbox"]')]
    .forEach((input) => { input.disabled = record.identity_lookup; });
  applyLanguageSpecificPosOptions();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderEmpty() {
  state.record = null;
  els.loading.classList.add("hidden");
  els.form.classList.add("hidden");
  els.empty.classList.remove("hidden");
}

async function loadNext(after = -1) {
  els.loading.classList.remove("hidden");
  els.form.classList.add("hidden");
  try {
    const result = await api(`/api/next?after=${after}`);
    updateProgress(result.progress);
    result.record ? renderRecord(result.record) : renderEmpty();
  } catch (error) {
    showFatal(error);
  }
}

async function loadPosition(position) {
  els.loading.classList.remove("hidden");
  els.form.classList.add("hidden");
  try {
    renderRecord(await api(`/api/records/${position}`));
  } catch (error) {
    showFatal(error);
  }
}

async function saveCurrent(event) {
  event.preventDefault();
  if (!state.record || els.saveButton.disabled) return;
  els.saveButton.disabled = true;
  els.saveMessage.textContent = "Saving...";
  els.saveMessage.classList.remove("error");

  const payload = {
    position: state.record.position,
    occurrence_id: state.record.occurrence_id,
    informal_token: state.record.informal_token,
    formal_token: state.record.formal_token,
    lemma: els.lemma.value.trim(),
    pos: els.pos.value,
    source_language: els.sourceLanguage.value,
    target_language: els.targetLanguage.value,
    entity_type: els.entityType.value,
    protected: els.protected.value,
    error_types: [...els.errorTypes.querySelectorAll('input[type="checkbox"]:checked')]
      .map((input) => input.value),
    review_status: els.reviewStatus.value,
    notes: els.notes.value.trim(),
    lexical_reusable: els.reusable.checked,
    lookup_source: state.record.prefill.lookup_source,
  };

  try {
    const result = await api("/api/annotations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    updateProgress(result.progress);
    result.next_position === null ? renderEmpty() : await loadPosition(result.next_position);
  } catch (error) {
    els.saveMessage.textContent = error.message;
    els.saveMessage.classList.add("error");
  } finally {
    els.saveButton.disabled = false;
  }
}

function showFatal(error) {
  els.loading.classList.remove("hidden");
  els.loading.textContent = error.message;
  els.loading.style.color = "#a43832";
}

els.form.addEventListener("submit", saveCurrent);
els.entityType.addEventListener("change", applyPersonEntityDefaults);
els.targetLanguage.addEventListener("change", applyLanguageSpecificPosOptions);
document.querySelector("#resume-button").addEventListener("click", () => loadNext(state.record?.position ?? -1));
document.querySelector("#skip-button").addEventListener("click", () => loadNext(state.record?.position ?? -1));
document.querySelector("#go-position").addEventListener("click", () => loadPosition(Number(els.position.value)));
els.position.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    loadPosition(Number(els.position.value));
  }
});
document.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") {
    event.preventDefault();
    els.form.requestSubmit();
  }
});

loadNext();
