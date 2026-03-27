const $ = (id) => document.getElementById(id);

// ── Ask question ──────────────────────────────────────────────────────────────

async function askQuestion() {
  const question = $("question").value.trim();
  const answerEl = $("answer");
  const statusEl = $("statusText");
  const askBtn   = $("askBtn");

  if (!question) {
    setStatus(statusEl, "Please enter a question.", "muted");
    return;
  }

  askBtn.disabled = true;
  answerEl.textContent = "";
  answerEl.classList.add("empty");
  setStatus(statusEl, "thinking", "loading");

  try {
    const res = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const text = await res.text();
      setStatus(statusEl, `Error ${res.status}: ${text}`, "error");
      return;
    }

    const data = await res.json();
    answerEl.textContent = data.answer ?? "";
    answerEl.classList.remove("empty");
    setStatus(statusEl, "", "");
  } catch (e) {
    setStatus(statusEl, "Request failed. Please try again.", "error");
  } finally {
    askBtn.disabled = false;
  }
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
  const files = Array.from(fileList);
  const statusEl = $("uploadStatus");

  if (files.length === 0) return;

  $("uploadZone").classList.remove("dragging");
  setUploadStatus(statusEl, `Uploading ${files.length} file(s)…`, "");

  try {
    const data = await uploadFiles(files);
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

  // Click-to-open is handled natively by the <label for="fileInput"> in HTML.
  // JS only needs to handle drag-and-drop and the change event.

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

function wireAskButton() {
  $("askBtn").addEventListener("click", askQuestion);
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) askQuestion();
  });
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function setStatus(el, message, type) {
  if (type === "loading") {
    el.innerHTML = `<span class="spinner"></span> Working…`;
  } else {
    el.textContent = message;
    el.style.color = type === "error"
      ? "var(--danger)"
      : "var(--text-muted)";
  }
}

function setUploadStatus(el, message, type) {
  el.textContent = message;
  el.className = type; // "success" | "error" | ""
}

wireUpload();
wireAskButton();
