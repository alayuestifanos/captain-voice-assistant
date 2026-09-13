const queryEl = document.getElementById("query");
const langEl = document.getElementById("lang");
const askBtn = document.getElementById("askBtn");
const resultEl = document.getElementById("result");
const statusEl = document.getElementById("statusLine");

const warningsEl = document.getElementById("warnings");
const answerEnEl = document.getElementById("answerEn");
const citationsEl = document.getElementById("citations");
const translatedHeading = document.getElementById("translatedHeading");
const answerTranslatedEl = document.getElementById("answerTranslated");
const audioBlock = document.getElementById("audioBlock");
const audioPlayer = document.getElementById("audioPlayer");
const ttsMeta = document.getElementById("ttsMeta");
const chunksEl = document.getElementById("chunks");
const traceIdEl = document.getElementById("traceId");

async function loadLanguages() {
  try {
    const res = await fetch("/api/languages");
    const data = await res.json();
    langEl.innerHTML = "";
    for (const [code, name] of Object.entries(data.options)) {
      const opt = document.createElement("option");
      opt.value = code;
      opt.textContent = `${name} (${code})`;
      if (code === data.default) opt.selected = true;
      langEl.appendChild(opt);
    }
  } catch (e) {
    langEl.innerHTML = '<option value="am">Amharic (am)</option>';
  }
}

function setBusy(busy) {
  askBtn.disabled = busy;
  askBtn.textContent = busy ? "Thinking…" : "Ask";
  statusEl.textContent = busy ? "Retrieving context, generating, translating, and synthesizing audio…" : "";
}

async function ask() {
  const text = queryEl.value.trim();
  if (!text) return;
  setBusy(true);
  resultEl.classList.add("hidden");

  try {
    const res = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, target_lang: langEl.value }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Request failed");
    }
    render(data);
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
    return;
  } finally {
    setBusy(false);
  }
}

function render(data) {
  resultEl.classList.remove("hidden");

  if (data.warnings && data.warnings.length) {
    warningsEl.classList.remove("hidden");
    warningsEl.innerHTML = data.warnings.map((w) => `&#9888; ${escapeHtml(w)}`).join("<br/>");
  } else {
    warningsEl.classList.add("hidden");
  }

  answerEnEl.textContent = data.answer_en;
  citationsEl.textContent = data.citations && data.citations.length
    ? `Sources: ${data.citations.join("; ")}`
    : (data.grounded ? "" : "No matching content found in the knowledge base.");

  translatedHeading.textContent = `Translated (${data.target_lang_name})`;
  answerTranslatedEl.textContent = data.translation_skipped
    ? "(target language is English — translation skipped)"
    : data.translated_text;

  if (data.audio_url) {
    audioBlock.classList.remove("hidden");
    audioPlayer.src = data.audio_url;
    ttsMeta.textContent = `Engine: ${data.tts_engine} · Voice: ${data.voice_label}`;
  } else {
    audioBlock.classList.add("hidden");
  }

  chunksEl.innerHTML = "";
  for (const c of data.retrieved_chunks) {
    const div = document.createElement("div");
    div.className = "chunk";
    div.innerHTML = `<span class="chunk-score">score ${c.score}</span>
      <div class="chunk-title">${escapeHtml(c.title)}</div>
      <div>${escapeHtml(c.text)}</div>
      <div class="meta">${escapeHtml(c.source)}</div>`;
    chunksEl.appendChild(div);
  }
  if (!data.retrieved_chunks.length) {
    chunksEl.innerHTML = '<p class="meta">No relevant chunks retrieved.</p>';
  }

  traceIdEl.textContent = `${data.trace_id}  (see logs/traces/${data.trace_id}.json)`;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

askBtn.addEventListener("click", ask);
queryEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) ask();
});

loadLanguages();
