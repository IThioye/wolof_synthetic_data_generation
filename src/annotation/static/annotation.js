const state = {
  record: null,
  prefetched: null,
  sentenceDirty: false,
  startedAt: 0,
};

const els = {
  form: document.querySelector("#annotation-form"),
  loading: document.querySelector("#loading"),
  empty: document.querySelector("#empty-state"),
  recordNumber: document.querySelector("#record-number"),
  existingBadge: document.querySelector("#existing-badge"),
  videoLink: document.querySelector("#video-link"),
  comment: document.querySelector("#comment-text"),
  tokenRows: document.querySelector("#token-rows"),
  tokenDetails: document.querySelector("#token-details"),
  manualSentence: document.querySelector("#manual-sentence"),
  autoSentence: document.querySelector("#auto-sentence"),
  changeSummary: document.querySelector("#change-summary"),
  saveButton: document.querySelector("#save-button"),
  saveMessage: document.querySelector("#save-message"),
  position: document.querySelector("#position-input"),
  saved: document.querySelector("#saved-count"),
  remaining: document.querySelector("#remaining-count"),
  total: document.querySelector("#total-count"),
  progressBar: document.querySelector("#progress-bar"),
  kept: document.querySelector("#kept-count"),
  goldRemaining: document.querySelector("#gold-remaining-count"),
  goldTarget: document.querySelector("#gold-target-count"),
  goldProgressBar: document.querySelector("#gold-progress-bar"),
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
  els.total.textContent = `${progress.total.toLocaleString()} candidate comments`;
  const percentage = progress.total ? (100 * progress.saved / progress.total) : 0;
  els.progressBar.style.width = `${percentage}%`;
  els.kept.textContent = progress.kept.toLocaleString();
  els.goldRemaining.textContent = progress.annotation_remaining.toLocaleString();
  els.goldTarget.textContent = `${progress.kept_videos.toLocaleString()} videos · target ${progress.annotation_target.toLocaleString()} pairs`;
  const goldPercentage = progress.annotation_target
    ? Math.min(100, 100 * progress.kept / progress.annotation_target)
    : 100;
  els.goldProgressBar.style.width = `${goldPercentage}%`;
  els.position.max = Math.max(progress.total - 1, 0);
}

function selectedTokenRows() {
  return [...els.tokenRows.querySelectorAll(".token-row")].map((element) => {
    const selected = element.querySelector("select").value;
    const custom = element.querySelector("input").value.trim();
    return {
      token: element.dataset.token,
      category: element.dataset.category,
      selected,
      manual: custom || selected,
    };
  });
}

function replaceSurfaceWords(text, replacements) {
  let index = 0;
  return text.replace(/[\p{L}\p{N}_]+(?:[-'’][\p{L}\p{N}_]+)*/gu, () => replacements[index++] ?? "");
}

function sentenceFromTokens() {
  const replacements = selectedTokenRows().map((row) => row.manual);
  return replaceSurfaceWords(state.record?.comment || "", replacements);
}

function refreshSentenceFromTokens() {
  els.manualSentence.value = sentenceFromTokens();
  state.sentenceDirty = true;
  els.manualSentence.focus();
}

function createTokenRow(row) {
  const container = document.createElement("div");
  container.className = `token-row${row.learned ? " learned" : ""}${row.changed ? " proposed-change" : ""}${row.review_required ? " needs-review" : ""}`;
  container.dataset.token = row.token;
  container.dataset.category = row.category;

  const token = document.createElement("span");
  token.className = "token-value";
  if (row.learned) {
    const dot = document.createElement("i");
    dot.className = "learned-dot";
    token.append(dot);
  }
  token.append(document.createTextNode(row.token));

  const category = document.createElement("span");
  category.className = "category";
  const categoryName = document.createElement("strong");
  categoryName.textContent = row.category.replaceAll("_", " ");
  const reason = document.createElement("small");
  reason.textContent = row.reason || "";
  category.append(categoryName, reason);

  const select = document.createElement("select");
  select.setAttribute("aria-label", `Candidate for ${row.token}`);
  row.suggestions.forEach((suggestion) => {
    const option = document.createElement("option");
    option.value = suggestion;
    const detail = row.candidate_details?.[suggestion];
    option.textContent = detail ? `${suggestion} · d=${Number(detail.score).toFixed(2)}` : suggestion;
    if (detail) {
      option.title = [detail.pos, detail.definition].filter(Boolean).join(" — ");
    }
    option.selected = suggestion === row.auto;
    select.append(option);
  });

  const custom = document.createElement("input");
  custom.type = "text";
  custom.placeholder = "Optional override";
  custom.setAttribute("aria-label", `Override for ${row.token}`);

  container.append(token, category, select, custom);
  return container;
}

function useSource(saveImmediately = false) {
  if (!state.record) return;
  els.manualSentence.value = state.record.comment;
  document.querySelector('input[name="status"][value="keep"]').checked = true;
  state.sentenceDirty = true;
  if (saveImmediately) els.form.requestSubmit();
  else els.manualSentence.focus();
}

function useSuggestion() {
  if (!state.record) return;
  els.manualSentence.value = state.record.auto_sentence;
  document.querySelector('input[name="status"][value="keep"]').checked = true;
  state.sentenceDirty = true;
  els.manualSentence.focus();
}

function renderRecord(record) {
  state.record = record;
  state.sentenceDirty = false;
  state.startedAt = performance.now();
  els.loading.classList.add("hidden");
  els.empty.classList.add("hidden");
  els.form.classList.remove("hidden");
  els.saveMessage.textContent = "";
  els.saveMessage.classList.remove("error");
  els.position.value = record.position;
  els.recordNumber.textContent = `Comment #${record.source_index} · position ${record.position}`;
  els.comment.textContent = record.comment;
  els.autoSentence.textContent = record.auto_sentence;
  // Start from the observed text to avoid silently anchoring the annotator to
  // a noisy fuzzy-match suggestion. Good suggestions remain one click away.
  els.manualSentence.value = record.comment;
  const changeText = record.suggested_change_count === 1
    ? "1 lookup change"
    : `${record.suggested_change_count} lookup changes`;
  const reviewText = record.review_token_count === 1
    ? "1 token to review"
    : `${record.review_token_count} tokens to review`;
  els.changeSummary.textContent = `${changeText} · ${reviewText}`;
  document.querySelector('input[name="status"][value="keep"]').checked = true;
  els.tokenDetails.open = false;

  els.existingBadge.classList.toggle("hidden", !record.already_annotated);
  els.existingBadge.textContent = record.already_annotated
    ? `Previously saved as ${record.latest_status}; saving appends history`
    : "";
  els.videoLink.classList.toggle("hidden", !record.video_url);
  els.videoLink.href = record.video_url || "#";

  els.tokenRows.replaceChildren(...record.token_rows.map(createTokenRow));
  prefetchNext(record.position);
  window.scrollTo({ top: 0, behavior: "smooth" });
  requestAnimationFrame(() => els.manualSentence.focus());
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

async function prefetchNext(after) {
  state.prefetched = null;
  try {
    const result = await api(`/api/next?after=${after}`);
    state.prefetched = result.record;
    updateProgress(result.progress);
  } catch (_) {
    // Foreground navigation retries if this optional optimization fails.
  }
}

function applyLearnedUpdates(record, updates) {
  if (!record) return record;
  const guardedCategories = new Set([
    "protected", "french_exact", "ambiguous_exact", "possible_french", "ambiguous_near",
  ]);
  record.token_rows.forEach((row) => {
    const correction = updates[row.token.toLocaleLowerCase()];
    if (!correction || guardedCategories.has(row.category)) return;
    row.suggestions = [correction, ...row.suggestions.filter((item) => item !== correction)].slice(0, 8);
    row.auto = correction;
    row.learned = true;
    row.changed = correction !== row.token;
    row.accepted = true;
    row.review_required = false;
    row.category = "learned_correction";
    row.reason = "One previously confirmed human correction";
  });
  record.auto_sentence = replaceSurfaceWords(record.comment, record.token_rows.map((row) => row.auto));
  record.suggested_change_count = record.token_rows.filter((row) => row.changed).length;
  record.review_token_count = record.token_rows.filter((row) => row.review_required).length;
  return record;
}

async function saveCurrent(event) {
  event.preventDefault();
  if (!state.record || els.saveButton.disabled) return;
  els.saveButton.disabled = true;
  els.saveMessage.textContent = "Saving…";
  els.saveMessage.classList.remove("error");

  const payload = {
    position: state.record.position,
    source_index: state.record.source_index,
    status: document.querySelector('input[name="status"]:checked').value,
    auto_sentence: state.record.auto_sentence,
    manual_sentence: els.manualSentence.value.trim(),
    token_rows: selectedTokenRows(),
    elapsed_ms: Math.round(performance.now() - state.startedAt),
  };

  try {
    const result = await api("/api/annotations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    updateProgress(result.progress);
    const prefetched = applyLearnedUpdates(state.prefetched, result.learned_updates);
    if (prefetched && prefetched.position === result.next_position) {
      renderRecord(prefetched);
    } else if (result.next_position === null) {
      renderEmpty();
    } else {
      await loadPosition(result.next_position);
    }
  } catch (error) {
    els.saveMessage.textContent = error.message;
    els.saveMessage.classList.add("error");
  } finally {
    els.saveButton.disabled = false;
  }
}

function selectStatus(value) {
  const input = document.querySelector(`input[name="status"][value="${value}"]`);
  if (input) input.checked = true;
}

function showFatal(error) {
  els.loading.classList.remove("hidden");
  els.loading.textContent = error.message;
  els.loading.style.color = "#a43832";
}

els.form.addEventListener("submit", saveCurrent);
els.manualSentence.addEventListener("input", () => { state.sentenceDirty = true; });
document.querySelector("#use-source-button").addEventListener("click", () => useSource(false));
document.querySelector("#use-auto-button").addEventListener("click", useSuggestion);
document.querySelector("#keep-unchanged-button").addEventListener("click", () => useSource(true));
document.querySelector("#apply-tokens-button").addEventListener("click", refreshSentenceFromTokens);
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
    return;
  }
  if (!event.altKey) return;
  const key = event.key.toLowerCase();
  const shortcuts = {
    "1": () => selectStatus("keep"),
    "2": () => selectStatus("discard_false_positive"),
    "3": () => selectStatus("discard_uninteresting"),
    "4": () => selectStatus("skip_uncertain"),
    "o": () => useSource(true),
    "a": useSuggestion,
    "t": () => { els.tokenDetails.open = !els.tokenDetails.open; },
    "s": () => loadNext(state.record?.position ?? -1),
  };
  if (shortcuts[key]) {
    event.preventDefault();
    shortcuts[key]();
  }
});

loadNext();
