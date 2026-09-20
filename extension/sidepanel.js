/**
 * Side panel logic — handles UI state, message routing, and user interactions.
 */

// ============================================================================
// DOM References
// ============================================================================

const connectionIndicator = document.getElementById("connectionIndicator");
const connectionStatus = document.getElementById("connectionStatus");
const commandInput = document.getElementById("commandInput");
const submitBtn = document.getElementById("submitBtn");
const cancelBtn = document.getElementById("cancelBtn");

const statusSection = document.getElementById("statusSection");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");
const statusBadge = document.getElementById("statusBadge");
const currentStep = document.getElementById("currentStep");
const statusDetails = document.getElementById("statusDetails");
const timeline = document.getElementById("timeline");

const confirmationSection = document.getElementById("confirmationSection");
const confirmationDesc = document.getElementById("confirmationDesc");
const confirmationDetails = document.getElementById("confirmationDetails");
const confirmBtn = document.getElementById("confirmBtn");
const rejectBtn = document.getElementById("rejectBtn");

const resultsSection = document.getElementById("resultsSection");
const resultCard = document.getElementById("resultCard");
const resultIcon = document.getElementById("resultIcon");
const resultTitle = document.getElementById("resultTitle");
const resultRetried = document.getElementById("resultRetried");
const resultSummary = document.getElementById("resultSummary");
const resultDuration = document.getElementById("resultDuration");
const resultExplainWrap = document.getElementById("resultExplainWrap");
const resultExplain = document.getElementById("resultExplain");
const feedbackRow = document.getElementById("feedbackRow");
const feedbackUp = document.getElementById("feedbackUp");
const feedbackDown = document.getElementById("feedbackDown");
const feedbackStatus = document.getElementById("feedbackStatus");
const feedbackAsk = document.getElementById("feedbackAsk");
const feedbackComment = document.getElementById("feedbackComment");
const feedbackSend = document.getElementById("feedbackSend");
const feedbackSkip = document.getElementById("feedbackSkip");

const historyList = document.getElementById("historyList");

// ============================================================================
// State
// ============================================================================

let currentTaskId = null;
let isConnected = false;
let taskHistory = [];
let lastInstruction = "";
let lastCompletedTaskId = null;
let lastTimelineText = "";

// ============================================================================
// Connection Management
// ============================================================================

function updateConnectionUI(connected) {
  isConnected = connected;
  connectionIndicator.classList.toggle("connected", connected);
  connectionStatus.textContent = connected ? "Connected" : "Disconnected";
  connectionStatus.classList.toggle("connected", connected);
  submitBtn.disabled = !connected || !commandInput.value.trim();

  if (connected && statusDetails && statusDetails.textContent.includes("Not connected to backend")) {
    statusDetails.textContent = "";
    if (!currentTaskId) {
      statusSection.style.display = "none";
    }
  }
}

// Check initial connection status
chrome.runtime.sendMessage({ type: "get_ws_status" }, (response) => {
  if (response) {
    updateConnectionUI(response.connected);
  }
});

// ============================================================================
// Message Handler (from background.js)
// ============================================================================

chrome.runtime.onMessage.addListener((message) => {
  console.log("[SP] Received:", message.type, message);

  switch (message.type) {
    case "ws_status":
      updateConnectionUI(message.connected);
      break;

    case "connected":
      updateConnectionUI(true);
      break;

    case "status_update":
      handleStatusUpdate(message);
      break;

    case "confirmation_request":
      handleConfirmationRequest(message);
      break;

    case "task_complete":
      handleTaskComplete(message);
      break;

    case "feedback_recorded":
      handleFeedbackRecorded(message);
      break;

    case "error":
      handleError(message);
      break;

    case "voice_transcript":
      // The backend has already started a task from this transcript; show it
      // as the running task. Submitting it again here ran every voice task
      // from the extension twice (e.g. a message sent twice).
      setVoiceState("idle");
      setVoiceStatus("");
      showTaskStarted(message.text || "");
      addTimelineItem("🎙️ You said: " + (message.text || ""), "success");
      break;
  }
});

// ============================================================================
// Event Handlers
// ============================================================================

// Enable/disable submit button based on input
commandInput.addEventListener("input", () => {
  submitBtn.disabled = !isConnected || !commandInput.value.trim();
});

// Submit on Ctrl+Enter
commandInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    if (!submitBtn.disabled) {
      submitTask();
    }
  }
});

submitBtn.addEventListener("click", submitTask);

// ============================================================================
// Voice input: record -> backend Whisper -> transcript
// ============================================================================
const micBtn = document.getElementById("micBtn");
const voiceStatus = document.getElementById("voiceStatus");

const MAX_RECORDING_MS = 60_000; // hard safety cap so a stuck recording can't run forever
const MIN_BLOB_BYTES = 1200; // below this it's silence/click noise, not speech

// "idle" -> "recording" -> "transcribing" -> "idle"
let voiceState = "idle";
let recorder = null;
let chunks = [];
let maxDurationTimer = null;

function setVoiceStatus(text, isError = false) {
  voiceStatus.textContent = text || "";
  voiceStatus.classList.toggle("err", !!isError);
}

function setVoiceState(next) {
  voiceState = next;
  micBtn.classList.toggle("rec", next === "recording");
  micBtn.classList.toggle("busy", next === "transcribing");
  micBtn.disabled = next === "transcribing";
}

/** Convert a Blob to base64 without spreading bytes as call args (which
 * throws "Maximum call stack size exceeded" on Chrome for anything much
 * over ~1-2s of audio — the previous implementation silently dropped every
 * voice task longer than a couple of seconds). FileReader handles any size. */
function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const dataUrl = reader.result || "";
      const commaIndex = dataUrl.indexOf(",");
      resolve(commaIndex >= 0 ? dataUrl.slice(commaIndex + 1) : "");
    };
    reader.onerror = () => reject(reader.error || new Error("FileReader failed"));
    reader.readAsDataURL(blob);
  });
}

function stopRecording() {
  if (maxDurationTimer) {
    clearTimeout(maxDurationTimer);
    maxDurationTimer = null;
  }
  if (recorder && recorder.state !== "inactive") {
    recorder.stop();
  }
}

micBtn.addEventListener("click", async () => {
  if (!isConnected) {
    setVoiceStatus("Not connected to backend.", true);
    return;
  }

  if (voiceState === "transcribing") {
    return; // ignore clicks while a transcription request is in flight
  }

  if (voiceState === "recording") {
    stopRecording();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recorder = new MediaRecorder(stream);
    chunks = [];

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };

    recorder.onerror = (e) => {
      stream.getTracks().forEach((t) => t.stop());
      setVoiceState("idle");
      setVoiceStatus("Recording error: " + (e.error?.message || "unknown"), true);
    };

    recorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      if (maxDurationTimer) {
        clearTimeout(maxDurationTimer);
        maxDurationTimer = null;
      }

      const blob = new Blob(chunks, { type: recorder.mimeType || "audio/webm" });
      chunks = [];

      if (blob.size < MIN_BLOB_BYTES) {
        setVoiceState("idle");
        setVoiceStatus("Recording too short — hold the mic a little longer.", true);
        return;
      }

      setVoiceState("transcribing");
      setVoiceStatus("Transcribing voice…");
      appendTimeline("Transcribing voice…");

      try {
        const b64 = await blobToBase64(blob);
        chrome.runtime.sendMessage({
          type: "voice_task",
          audio_base64: b64,
          mime: blob.type,
          scope: "browser",
        });
      } catch (err) {
        setVoiceState("idle");
        setVoiceStatus("Could not process the recording: " + err.message, true);
      }
    };

    recorder.start();
    setVoiceState("recording");
    setVoiceStatus("Listening… tap 🎙️ again to send.");
    appendTimeline("Listening… tap 🎙️ again to send.");

    // Safety cap: never let a forgotten recording run forever.
    maxDurationTimer = setTimeout(() => {
      if (voiceState === "recording") {
        setVoiceStatus("Max recording length reached — sending now.");
        stopRecording();
      }
    }, MAX_RECORDING_MS);
  } catch (e) {
    setVoiceState("idle");
    const msg = e.name === "NotAllowedError"
      ? "Microphone permission denied. Allow mic access for this extension and try again."
      : e.name === "NotFoundError"
      ? "No microphone found."
      : "Microphone unavailable: " + e.message;
    setVoiceStatus(msg, true);
    appendTimeline(msg);
  }
});


cancelBtn.addEventListener("click", () => {
  if (currentTaskId) {
    chrome.runtime.sendMessage({
      type: "task_cancel",
      task_id: currentTaskId,
    });
  }
  statusSection.style.display = "none";
  currentTaskId = null;
});

confirmBtn.addEventListener("click", () => {
  if (currentTaskId) {
    chrome.runtime.sendMessage({
      type: "confirmation_response",
      task_id: currentTaskId,
      confirmed: true,
    });
    confirmationSection.style.display = "none";
    statusSection.style.display = "block";
    statusBadge.textContent = "Executing";
    statusBadge.className = "status-badge acting";
  }
});

rejectBtn.addEventListener("click", () => {
  if (currentTaskId) {
    chrome.runtime.sendMessage({
      type: "confirmation_response",
      task_id: currentTaskId,
      confirmed: false,
    });
    confirmationSection.style.display = "none";
    statusSection.style.display = "block";
  }
});

// ============================================================================
// Feedback (👍 / 👎) — weights what the agent reuses on similar tasks
// ============================================================================

function sendFeedback(rating, comment = "") {
  if (!lastCompletedTaskId) return;
  chrome.runtime.sendMessage({
    type: "task_feedback",
    task_id: lastCompletedTaskId,
    rating: rating,
    comment: comment,
  });
  feedbackUp.disabled = true;
  feedbackDown.disabled = true;
  (rating > 0 ? feedbackUp : feedbackDown).classList.add("chosen");
  feedbackAsk.style.display = "none";
  feedbackStatus.textContent = "Saving your feedback…";
}

feedbackUp.addEventListener("click", () => sendFeedback(1));
// 👎 asks what went wrong: the note is stored with the task and shown to the
// agent next time it plans something similar.
feedbackDown.addEventListener("click", () => {
  feedbackAsk.style.display = "flex";
  feedbackComment.focus();
});
feedbackSend.addEventListener("click", () => sendFeedback(-1, feedbackComment.value.trim()));
feedbackSkip.addEventListener("click", () => sendFeedback(-1));
feedbackComment.addEventListener("keydown", (e) => {
  if (e.key === "Enter") feedbackSend.click();
});

function handleFeedbackRecorded(data) {
  if (data.task_id !== lastCompletedTaskId) return;
  feedbackStatus.textContent = data.message || (data.applied ? "Feedback saved." : "Feedback could not be saved.");
  // The rating may have been spoken ("Emma, good job, done") rather than clicked.
  feedbackUp.disabled = true;
  feedbackDown.disabled = true;
  (data.rating > 0 ? feedbackUp : feedbackDown).classList.add("chosen");
  feedbackAsk.style.display = "none";
}

function resetFeedback(taskId) {
  lastCompletedTaskId = taskId || null;
  feedbackUp.disabled = false;
  feedbackDown.disabled = false;
  feedbackUp.classList.remove("chosen");
  feedbackDown.classList.remove("chosen");
  feedbackAsk.style.display = "none";
  feedbackComment.value = "";
  feedbackStatus.textContent = "";
  feedbackRow.style.display = taskId ? "flex" : "none";
}

// ============================================================================
// Task Submission
// ============================================================================

function appendTimeline(text) {
  const timelineEl = document.getElementById("timeline");
  if (!timelineEl) return;
  const item = document.createElement("li");
  item.textContent = text;
  timelineEl.appendChild(item);
}

function submitTask() {
  const instruction = commandInput.value.trim();
  if (!instruction || !isConnected) return;

  // Send to background → backend
  chrome.runtime.sendMessage({
    type: "task_submit",
    instruction: instruction,
  });
  showTaskStarted(instruction);
}

/** Reset the panel for a task that has just started (typed or voice). */
function showTaskStarted(instruction) {
  lastInstruction = instruction;

  // Reset UI
  commandInput.value = "";
  submitBtn.disabled = true;

  // Show status section
  statusSection.style.display = "block";
  resultsSection.style.display = "none";
  confirmationSection.style.display = "none";

  // Reset progress
  progressFill.style.width = "0%";
  progressText.textContent = "0%";
  statusBadge.textContent = "Planning";
  statusBadge.className = "status-badge";
  currentStep.textContent = "Analyzing your instruction...";
  statusDetails.textContent = "";
  timeline.innerHTML = "";
  lastTimelineText = "";
}

// ============================================================================
// Status Updates
// ============================================================================

function handleStatusUpdate(data) {
  currentTaskId = data.task_id;
  statusSection.style.display = "block";

  // Update progress
  const progress = Math.round((data.progress || 0) * 100);
  progressFill.style.width = `${progress}%`;
  progressText.textContent = `${progress}%`;

  // Update status badge
  const statusMap = {
    thinking: "Recalling",
    planning: "Planning",
    perceiving: "Observing",
    acting: "Executing",
    verifying: "Verifying",
    replanning: "Retrying",
    learning: "Learning",
    waiting_confirmation: "Awaiting Confirmation",
    completed: "Completed",
    failed: "Failed",
  };
  statusBadge.textContent = statusMap[data.status] || data.status;
  statusBadge.className = `status-badge ${data.status}`;

  // Update current step
  if (data.current_step) {
    currentStep.textContent = data.current_step;
  }

  // Update step counter
  if (data.total_steps > 0) {
    statusDetails.textContent = `Step ${data.current_step_index + 1} of ${data.total_steps}`;
  }

  // Live action stream: every tool call the agent makes shows up here.
  const streamed = ["acting", "replanning", "thinking", "waiting_confirmation"];
  if (data.current_step && streamed.includes(data.status) && data.current_step !== lastTimelineText) {
    const failed = data.current_step.startsWith("⚠️");
    addTimelineItem(data.current_step, failed ? "failed" : "pending");
    lastTimelineText = data.current_step;
  }

  // Show details if provided
  if (data.details) {
    statusDetails.textContent = data.details;
  }
}

function addTimelineItem(text, status = "pending") {
  const icons = { success: "✓", pending: "●", failed: "✗" };
  const item = document.createElement("div");
  item.className = "timeline-item";
  item.innerHTML = `
    <span class="timeline-icon ${status}">${icons[status]}</span>
    <span class="timeline-text">${escapeHTML(text)}</span>
  `;
  timeline.appendChild(item);

  // Mark previous pending items as success
  const items = timeline.querySelectorAll(".timeline-item");
  for (let i = 0; i < items.length - 1; i++) {
    const icon = items[i].querySelector(".timeline-icon");
    if (icon.classList.contains("pending")) {
      icon.classList.remove("pending");
      icon.classList.add("success");
      icon.textContent = icons.success;
    }
  }

  // Scroll to bottom
  timeline.scrollTop = timeline.scrollHeight;
}

// ============================================================================
// Confirmation Request
// ============================================================================

function handleConfirmationRequest(data) {
  currentTaskId = data.task_id;
  statusSection.style.display = "none";
  confirmationSection.style.display = "block";

  confirmationDesc.textContent = data.action_description || "This action requires your confirmation.";

  let detailsHTML = "";
  if (data.product_name) detailsHTML += `<div><strong>Product:</strong> ${escapeHTML(data.product_name)}</div>`;
  if (data.price) detailsHTML += `<div><strong>Price:</strong> ${escapeHTML(data.price)}</div>`;
  if (data.quantity && (data.product_name || data.price)) detailsHTML += `<div><strong>Quantity:</strong> ${data.quantity}</div>`;
  if (data.details) {
    for (const [key, value] of Object.entries(data.details)) {
      detailsHTML += `<div><strong>${escapeHTML(key)}:</strong> ${escapeHTML(String(value))}</div>`;
    }
  }
  confirmationDetails.innerHTML = detailsHTML || "<div>No additional details</div>";
}

// ============================================================================
// Task Complete
// ============================================================================

function handleTaskComplete(data) {
  statusSection.style.display = "none";
  confirmationSection.style.display = "none";
  resultsSection.style.display = "block";

  const success = data.success;

  resultCard.className = `result-card ${success ? "success" : "failure"}`;
  resultIcon.textContent = success ? "✓" : "✗";
  resultTitle.textContent = success ? "Task Completed" : "Task Failed";
  resultRetried.style.display = data.retried ? "block" : "none";
  resultSummary.textContent = data.summary || (success ? "All steps completed successfully." : data.error || "An error occurred.");

  if (data.duration_seconds) {
    resultDuration.textContent = `Completed in ${data.duration_seconds.toFixed(1)}s`;
  } else {
    resultDuration.textContent = "";
  }

  if (data.explanation) {
    resultExplain.textContent = data.explanation;
    resultExplainWrap.style.display = "block";
  } else {
    resultExplain.textContent = "";
    resultExplainWrap.style.display = "none";
  }

  // Cancelled tasks have nothing meaningful to rate.
  const cancelled = (data.summary || "").toLowerCase().includes("cancelled by user");
  resetFeedback(cancelled ? null : data.task_id);

  // Mark all timeline items as complete/failed
  const timelineItems = timeline.querySelectorAll(".timeline-icon.pending");
  timelineItems.forEach((icon) => {
    icon.classList.remove("pending");
    icon.classList.add(success ? "success" : "failed");
    icon.textContent = success ? "✓" : "✗";
  });

  // Add to history
  addHistoryItem({
    instruction: lastInstruction || data.summary || "Task",
    success: success,
    time: new Date().toLocaleTimeString(),
  });

  currentTaskId = null;
}

// ============================================================================
// Error Handling
// ============================================================================

function handleError(data) {
  // If a voice transcription was in flight, this error almost certainly
  // belongs to it (bad audio, no speech detected, transcription failure) —
  // surface it next to the mic instead of a hidden status panel the user
  // has no reason to be looking at yet.
  if (voiceState === "transcribing") {
    setVoiceState("idle");
    setVoiceStatus(data.message || "Voice transcription failed.", true);
  }
  if (data.message) {
    statusSection.style.display = "block";
    statusDetails.textContent = `Error: ${data.message}`;
  }
}

// ============================================================================
// History
// ============================================================================

function addHistoryItem(item) {
  taskHistory.unshift(item);

  // Remove empty message
  const emptyMsg = historyList.querySelector(".history-empty");
  if (emptyMsg) emptyMsg.remove();

  const el = document.createElement("div");
  el.className = "history-item";
  el.innerHTML = `
    <span class="history-dot ${item.success ? "success" : "failed"}"></span>
    <span class="history-text">${escapeHTML(item.instruction)}</span>
    <span class="history-time">${item.time}</span>
  `;

  historyList.insertBefore(el, historyList.firstChild);

  // Keep only last 10
  while (historyList.children.length > 10) {
    historyList.removeChild(historyList.lastChild);
  }
}

// ============================================================================
// Utilities
// ============================================================================

function escapeHTML(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
