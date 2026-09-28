const state = {
  record: null,
  prefetched: null,
  sentenceDirty: false,
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
  manualSentence: document.querySelector("#manual-sentence"),
  autoSentence: document.querySelector("#auto-sentence"),
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
  els.goldRemaining.textContent = progress.gold_remaining.toLocaleString();
  els.goldTarget.textContent = `${progress.kept_videos.toLocaleString()} / ${progress.minimum_videos.toLocaleString()} videos · target ${progress.minimum_kept.toLocaleString()} pairs`;
  const goldPercentage = progress.minimum_kept
    ? Math.min(100, 100 * progress.kept / progress.minimum_kept)
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

function sentenceFromTokens() {
  return selectedTokenRows().map((row) => row.manual).join(" ");
}

function refreshSentence() {
  if (!state.sentenceDirty) els.manualSentence.value = sentenceFromTokens();
}

function createTokenRow(row) {
  const container = document.createElement("div");
  container.className = `token-row${row.learned ? " learned" : ""}`;
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
  category.textContent = row.category;

  const select = document.createElement("select");
  select.setAttribute("aria-label", `Candidate for ${row.token}`);
  row.suggestions.forEach((suggestion) => {
    const option = document.createElement("option");
    option.value = suggestion;
    option.textContent = suggestion;
    option.selected = suggestion === row.auto;
    select.append(option);
  });
  select.addEventListener("change", refreshSentence);

  const custom = document.createElement("input");
  custom.type = "text";
  custom.placeholder = "Type an override";
  custom.setAttribute("aria-label", `Override for ${row.token}`);
  custom.addEventListener("input", refreshSentence);

  container.append(token, category, select, custom);
  return container;
}

function renderRecord(record) {
  state.record = record;
  state.sentenceDirty = false;
  els.loading.classList.add("hidden");
  els.empty.classList.add("hidden");
  els.form.classList.remove("hidden");
  els.saveMessage.textContent = "";
  els.saveMessage.classList.remove("error");
  els.position.value = record.position;
  els.recordNumber.textContent = `Comment #${record.source_index} · position ${record.position}`;
  els.comment.textContent = record.comment;
  els.autoSentence.textContent = record.auto_sentence;
  els.manualSentence.value = record.auto_sentence;
  document.querySelector('input[name="status"][value="keep"]').checked = true;

  els.existingBadge.classList.toggle("hidden", !record.already_annotated);
  els.existingBadge.textContent = record.already_annotated
    ? `Previously saved as ${record.latest_status}; saving appends history`
    : "";
  els.videoLink.classList.toggle("hidden", !record.video_url);
  els.videoLink.href = record.video_url || "#";

  els.tokenRows.replaceChildren(...record.token_rows.map(createTokenRow));
  prefetchNext(record.position);
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

async function prefetchNext(after) {
  state.prefetched = null;
  try {
    const result = await api(`/api/next?after=${after}`);
    state.prefetched = result.record;
    updateProgress(result.progress);
  } catch (_) {
    // Prefetch is an optional speed optimization; foreground navigation retries.
  }
}

function applyLearnedUpdates(record, updates) {
  if (!record) return record;
  record.token_rows.forEach((row) => {
    const correction = updates[row.token.toLocaleLowerCase()];
    if (!correction) return;
    row.suggestions = [correction, ...row.suggestions.filter((item) => item !== correction)].slice(0, 8);
    row.auto = correction;
    row.learned = true;
  });
  record.auto_sentence = record.token_rows.map((row) => row.auto).join(" ");
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

function showFatal(error) {
  els.loading.classList.remove("hidden");
  els.loading.textContent = error.message;
  els.loading.style.color = "#a43832";
}

els.form.addEventListener("submit", saveCurrent);
els.manualSentence.addEventListener("input", () => { state.sentenceDirty = true; });
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
