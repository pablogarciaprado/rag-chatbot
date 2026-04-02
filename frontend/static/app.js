const $ = (id) => document.getElementById(id);

// Conversation history: array of { role: "user" | "assistant", content: string }
// Does NOT include the in-flight message — it is appended only after success.
let conversationHistory = [];
let isThinking = false;

// ── Chat rendering ────────────────────────────────────────────────────────────

function appendMessage(role, content) {
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
  bubble.textContent = content;

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

    appendMessage("assistant", answer);

    // Commit both turns to history after a successful round-trip.
    conversationHistory.push({ role: "user",      content: question });
    conversationHistory.push({ role: "assistant", content: answer  });

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
    const saved   = data.saved?.length   ?? 0;
    const skipped = data.skipped?.length ?? 0;
    const msg = `${saved} file${saved !== 1 ? "s" : ""} uploaded successfully.${
      skipped ? ` ${skipped} unsupported file${skipped !== 1 ? "s" : ""} skipped.` : ""
    }`;
    setUploadStatus(statusEl, msg, "success");
  } catch (e) {
    setUploadStatus(statusEl, e?.message ?? "Upload error.", "error");
  } finally {
    $("fileInput").value = "";
  }
}

// ── Wire-up ───────────────────────────────────────────────────────────────────

function wireUpload() {
  const zone      = $("uploadZone");
  const fileInput = $("fileInput");

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

wireUpload();
wireChat();
