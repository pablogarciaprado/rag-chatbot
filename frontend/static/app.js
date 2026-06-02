const $ = (id) => document.getElementById(id);

// Conversation history: array of { role: "user" | "assistant", content: string }
// Does NOT include the in-flight message — it is appended only after success.
let conversationHistory = [];
let isThinking = false;
let isIndexed = false;
let isIndexing = false;
let uploadedFileCount = 0;
let uploadedFileNames = [];

// ── Chat rendering ────────────────────────────────────────────────────────────

function appendMessage(role, content, sources = []) {
  const chatWindow = $("chatWindow");

  // Remove empty-state placeholder on first real message.
  const emptyEl = $("chatEmpty");
  if (emptyEl) emptyEl.remove();

  const msg = document.createElement("div");
  msg.className = `msg ${role}`;

  const label = document.createElement("div");
  label.className = "msg-label";
  label.textContent = role === "user" ? "You" : "Assistant";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";

  // If the message is from the assistant, add the citation sources to the bubble.
  if (role === "assistant") {
    const text = document.createElement("div");
    text.className = "msg-bubble-text";
    text.textContent = content;
    bubble.appendChild(text);

    // If there are sources, add the citation sources to the bubble.
    if (sources && sources.length > 0) {
      const footer = document.createElement("div");
      footer.className = "msg-sources";

      const footerLabel = document.createElement("span");
      footerLabel.className = "msg-sources-label";
      footerLabel.textContent = "Sources";
      footer.appendChild(footerLabel);

      sources.forEach((src) => {
        const pill = document.createElement("span");
        pill.className = "msg-source-pill";
        pill.title = src.path || src.file || "";

        const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        icon.setAttribute("width", "10");
        icon.setAttribute("height", "10");
        icon.setAttribute("viewBox", "0 0 24 24");
        icon.setAttribute("fill", "none");
        icon.setAttribute("stroke", "currentColor");
        icon.setAttribute("stroke-width", "2");
        icon.setAttribute("stroke-linecap", "round");
        icon.setAttribute("stroke-linejoin", "round");
        icon.innerHTML = `<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>`;
        pill.appendChild(icon);

        const pillText = document.createElement("span");
        let label = src.file || src.path || "Unknown";
        if (src.page != null) label += ` · p. ${src.page}`;
        if (src.confidence_pct != null) label += ` · ${src.confidence_pct}%`;
        pillText.textContent = label;
        pill.appendChild(pillText);

        footer.appendChild(pill);
      });

      bubble.appendChild(footer);
    }
  } else {
    bubble.textContent = content;
  }

  msg.appendChild(label);
  msg.appendChild(bubble);
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function showThinkingBubble() {
  const chatWindow = $("chatWindow");

  const emptyEl = $("chatEmpty");
  if (emptyEl) emptyEl.remove();

  const msg = document.createElement("div");
  msg.id = "thinkingMsg";
  msg.className = "msg assistant";

  const label = document.createElement("div");
  label.className = "msg-label";
  label.textContent = "Assistant";

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble thinking";
  bubble.innerHTML = `<span class="spinner"></span> Thinking\u2026`;

  msg.appendChild(label);
  msg.appendChild(bubble);
  chatWindow.appendChild(msg);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function removeThinkingBubble() {
  const el = $("thinkingMsg");
  if (el) el.remove();
}

function restoreEmptyState() {
  const chatWindow = $("chatWindow");
  if (chatWindow.children.length === 0) {
    const empty = document.createElement("div");
    empty.className = "chat-empty";
    empty.id = "chatEmpty";
    empty.textContent = "No messages yet \u2014 start the conversation below.";
    chatWindow.appendChild(empty);
  }
}

// ── Send message ──────────────────────────────────────────────────────────────

async function sendMessage() {
  if (isThinking) return;

  const textarea  = $("question");
  const statusEl  = $("statusText");
  const askBtn    = $("askBtn");
  const question  = textarea.value.trim();

  if (!question) {
    setStatus(statusEl, "Please type a message first.", "muted");
    return;
  }

  if (!isIndexed) {
    setStatus(statusEl, "Index your documents before asking questions.", "error");
    return;
  }

  // Optimistically render the user bubble and clear the input.
  appendMessage("user", question);
  textarea.value = "";
  autoResize(textarea);
  setStatus(statusEl, "", "");

  // Lock UI and show thinking indicator.
  isThinking = true;
  askBtn.disabled = true;
  showThinkingBubble();

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        history: conversationHistory,  // all prior turns, not including current question
      }),
    });

    removeThinkingBubble();

    if (!res.ok) {
      let detail = `Error ${res.status}`;
      try {
        const body = await res.json();
        detail = body.detail ?? detail;
      } catch {
        detail = (await res.text()) || detail;
      }
      setStatus(statusEl, detail, "error");
      return;
    }

    const data = await res.json();
    const answer = data.answer ?? "";
    const sources = data.sources ?? [];

    // Append the assistant's message (and its citation sources) to the chat window.
    appendMessage("assistant", answer, sources);

    // Commit both turns to history after a successful round-trip.
    conversationHistory.push({ role: "user", content: question });
    conversationHistory.push({ role: "assistant", content: answer });

  } catch {
    removeThinkingBubble();
    setStatus(statusEl, "Request failed. Please try again.", "error");
  } finally {
    isThinking = false;
    askBtn.disabled = false;
    textarea.focus();
  }
}

// ── Clear / new chat ──────────────────────────────────────────────────────────

function clearChat() {
  $("chatWindow").innerHTML = "";
  conversationHistory = [];
  setStatus($("statusText"), "", "");
  restoreEmptyState();
}

// ── Auto-resize textarea ──────────────────────────────────────────────────────

function autoResize(textarea) {
  textarea.style.height = "auto";
  textarea.style.height = Math.min(textarea.scrollHeight, 140) + "px";
}

// ── Upload files ──────────────────────────────────────────────────────────────

async function uploadFiles(files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file, file.name);
  }

  const res = await fetch("/upload", { method: "POST", body: formData });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed (${res.status}): ${text}`);
  }

  return res.json();
}

async function handleFiles(fileList) {
  const files    = Array.from(fileList);
  const statusEl = $("uploadStatus");

  if (files.length === 0) return;

  $("uploadZone").classList.remove("dragging");
  setUploadStatus(statusEl, `Uploading ${files.length} file(s)\u2026`, "");

  try {
    const data    = await uploadFiles(files);
    const saved   = data.saved ?? [];
    const skipped = data.skipped ?? [];
    const msg = `${saved.length} file${saved.length !== 1 ? "s" : ""} uploaded. Click Index documents to make them searchable.${
      skipped.length ? ` ${skipped.length} unsupported file${skipped.length !== 1 ? "s" : ""} skipped.` : ""
    }`;
    setUploadStatus(statusEl, msg, "success");
    isIndexed = false;
    if (saved.length > 0) {
      addUploadedFiles(saved);
      updateIndexButton();
    }
    await refreshIndexStatus();
  } catch (e) {
    setUploadStatus(statusEl, e?.message ?? "Upload error.", "error");
  } finally {
    $("fileInput").value = "";
  }
}

// ── Index documents ───────────────────────────────────────────────────────────

async function indexDocuments() {
  if (isIndexing) return;

  const statusEl = $("uploadStatus");
  const indexBtn = $("indexBtn");

  isIndexing = true;
  indexBtn.disabled = true;
  setUploadStatus(statusEl, "Indexing documents\u2026", "");

  try {
    const res = await fetch("/index", { method: "POST" });

    if (!res.ok) {
      let detail = `Error ${res.status}`;
      try {
        const body = await res.json();
        detail = body.detail ?? detail;
      } catch {
        detail = (await res.text()) || detail;
      }
      setUploadStatus(statusEl, detail, "error");
      return;
    }

    const data = await res.json();
    const docs = data.documents ?? 0;
    const chunks = data.chunks ?? 0;
    isIndexed = true;
    setUploadStatus(
      statusEl,
      `Indexed ${docs} parsed unit${docs !== 1 ? "s" : ""} (${chunks} chunk${chunks !== 1 ? "s" : ""}). You can ask questions now.`,
      "success",
    );
  } catch (e) {
    setUploadStatus(statusEl, e?.message ?? "Indexing failed.", "error");
  } finally {
    isIndexing = false;
    updateIndexButton();
  }
}

async function refreshIndexStatus() {
  try {
    const res = await fetch("/index/status");
    if (!res.ok) return;

    const data = await res.json();
    isIndexed = Boolean(data.indexed);
    if (Array.isArray(data.files)) {
      uploadedFileNames = data.files;
      uploadedFileCount = uploadedFileNames.length;
      renderUploadedFileList();
    } else {
      const count = Number(data.file_count) || 0;
      if (count > 0) {
        uploadedFileCount = count;
      }
    }
    updateIndexButton();
  } catch {
    // Keep current uploadedFileCount if status check fails.
  }
}

function updateIndexButton() {
  const indexBtn = $("indexBtn");
  if (!indexBtn) return;

  const canIndex = !isIndexing && uploadedFileCount > 0;
  indexBtn.disabled = !canIndex;
  indexBtn.textContent = isIndexed ? "Re-index documents" : "Index documents";
}

// ── Wire-up ───────────────────────────────────────────────────────────────────

function wireUpload() {
  const zone      = $("uploadZone");
  const fileInput = $("fileInput");

  $("indexBtn").addEventListener("click", indexDocuments);

  fileInput.addEventListener("change", () => {
    if (fileInput.files?.length) handleFiles(fileInput.files);
  });

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("dragging");
  });

  zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));

  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    if (e.dataTransfer?.files?.length) handleFiles(e.dataTransfer.files);
  });
}

function wireChat() {
  const textarea = $("question");

  $("askBtn").addEventListener("click", sendMessage);
  $("clearBtn").addEventListener("click", clearChat);

  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  textarea.addEventListener("input", () => autoResize(textarea));
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function setStatus(el, message, type) {
  if (type === "loading") {
    el.innerHTML = `<span class="spinner"></span> Working\u2026`;
  } else {
    el.textContent = message;
    el.style.color = type === "error" ? "var(--danger)" : "var(--text-muted)";
  }
}

function setUploadStatus(el, message, type) {
  el.textContent = message;
  el.className = type; // "success" | "error" | ""
}

function addUploadedFiles(names) {
  for (const name of names) {
    if (!uploadedFileNames.includes(name)) {
      uploadedFileNames.push(name);
    }
  }
  uploadedFileNames.sort((a, b) => a.localeCompare(b));
  uploadedFileCount = uploadedFileNames.length;
  renderUploadedFileList();
}

function renderUploadedFileList() {
  const listEl = $("uploadedFileList");
  if (!listEl) return;

  listEl.innerHTML = "";

  if (uploadedFileNames.length === 0) {
    listEl.hidden = true;
    return;
  }

  listEl.hidden = false;

  for (const name of uploadedFileNames) {
    const item = document.createElement("li");
    item.className = "uploaded-file-item";

    const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    icon.setAttribute("width", "14");
    icon.setAttribute("height", "14");
    icon.setAttribute("viewBox", "0 0 24 24");
    icon.setAttribute("fill", "none");
    icon.setAttribute("stroke", "currentColor");
    icon.setAttribute("stroke-width", "2");
    icon.setAttribute("stroke-linecap", "round");
    icon.setAttribute("stroke-linejoin", "round");
    icon.innerHTML = `<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>`;

    const label = document.createElement("span");
    label.textContent = name;

    item.appendChild(icon);
    item.appendChild(label);
    listEl.appendChild(item);
  }
}

wireUpload();
wireChat();
refreshIndexStatus();
