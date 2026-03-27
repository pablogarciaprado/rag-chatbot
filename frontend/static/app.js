// Utility function (convenience wrapper) to get an element by its ID
// It creates a tiny helper so later code can do:
// $("question") instead of document.getElementById("question")
const $ = (id) => document.getElementById(id);

async function askQuestion() {
  const question = $("question").value.trim();
  const answerEl = $("answer");
  const statusEl = $("status");

  if (!question) {
    statusEl.textContent = "Please enter a question.";
    return;
  }

  statusEl.textContent = "Working...";
  answerEl.textContent = "";

  const res = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }), // same as { "question": question }
  });

  if (!res.ok) {
    const text = await res.text();
    statusEl.textContent = `Error (${res.status}): ${text}`;
    return;
  }

  const data = await res.json();
  statusEl.textContent = "";
  answerEl.textContent = data.answer ?? "";
}

async function uploadFiles(files) {
  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file, file.name);
  }

  const res = await fetch("/upload", {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Upload failed (${res.status}): ${text}`);
  }

  return res.json();
}

function wireUpload() {
  const uploadBtn = $("uploadBtn");
  const fileInput = $("fileInput");
  const uploadHint = $("uploadHint");

  uploadBtn.addEventListener("click", () => {
    uploadHint.textContent = "Select files to upload...";
    fileInput.click();
  });

  fileInput.addEventListener("change", async () => {
    const filesList = fileInput.files ? Array.from(fileInput.files) : [];
    const filesCount = filesList.length;

    if (filesCount === 0) {
      uploadHint.textContent = "";
      return;
    }

    uploadBtn.disabled = true;
    uploadHint.textContent = `Uploading ${filesCount} file(s)...`;

    try {
      const data = await uploadFiles(filesList);
      const savedCount = data.saved?.length ?? 0;
      const skippedCount = data.skipped?.length ?? 0;

      uploadHint.textContent = `Uploaded ${savedCount} file(s). Next question will use them. ${
        skippedCount ? `(Skipped ${skippedCount} unsupported file(s).)` : ""
      }`;
    } catch (e) {
      uploadHint.textContent = e?.message ? `Upload error: ${e.message}` : "Upload error.";
    } finally {
      uploadBtn.disabled = false;
      fileInput.value = "";
    }
  });
}

function wireAskButton() {
  $("askBtn").addEventListener("click", askQuestion);
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      askQuestion();
    }
  });
}

wireUpload();
wireAskButton();

