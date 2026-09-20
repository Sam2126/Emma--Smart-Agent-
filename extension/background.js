/**
 * Background service worker — manages WebSocket connection to the Python backend.
 *
 * Responsibilities:
 * - Maintain persistent WebSocket connection to ws://localhost:8765
 * - Relay messages between the side panel UI and the backend
 * - Handle reconnection when the MV3 service worker wakes up
 */

const WS_URL = "ws://localhost:8765";
const RECONNECT_INTERVAL_MS = 3000;
const MAX_RECONNECT_ATTEMPTS = 20;

let ws = null;
let reconnectAttempts = 0;
let reconnectTimer = null;

// ============================================================================
// WebSocket Connection Management
// ============================================================================

function connectWebSocket() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }

  console.log("[BG] Connecting to backend...", WS_URL);

  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    console.log("[BG] Connected to backend");
    reconnectAttempts = 0;
    // Notify side panel
    broadcastToSidePanel({ type: "ws_status", connected: true });
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      console.log("[BG] Received:", data.type, data);
      // Forward all backend messages to the side panel
      broadcastToSidePanel(data);
    } catch (e) {
      console.error("[BG] Failed to parse message:", e);
    }
  };

  ws.onerror = (error) => {
    console.error("[BG] WebSocket error:", error);
  };

  ws.onclose = (event) => {
    console.log("[BG] Disconnected from backend", event.code, event.reason);
    ws = null;
    broadcastToSidePanel({ type: "ws_status", connected: false });
    scheduleReconnect();
  };
}

function scheduleReconnect() {
  if (reconnectTimer) {
    clearTimeout(reconnectTimer);
  }

  if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
    console.log("[BG] Max reconnect attempts reached");
    broadcastToSidePanel({
      type: "error",
      message: "Cannot connect to backend. Is the server running?",
    });
    return;
  }

  reconnectAttempts++;
  const delay = Math.min(RECONNECT_INTERVAL_MS * reconnectAttempts, 15000);
  console.log(`[BG] Reconnecting in ${delay}ms (attempt ${reconnectAttempts})`);

  reconnectTimer = setTimeout(() => {
    connectWebSocket();
  }, delay);
}

function sendToBackend(message) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(message));
    console.log("[BG] Sent:", message.type);
    return true;
  } else {
    console.warn("[BG] Cannot send — not connected");
    broadcastToSidePanel({
      type: "error",
      message: "Not connected to backend. Please start the server.",
    });
    return false;
  }
}

// ============================================================================
// Communication with Side Panel
// ============================================================================

function broadcastToSidePanel(message) {
  // Use chrome.runtime messaging to communicate with the side panel
  chrome.runtime.sendMessage(message).catch(() => {
    // Side panel may not be open — that's fine
  });
}

// Listen for messages from the side panel
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log("[BG] Message from side panel:", message.type);

  switch (message.type) {
    case "task_submit":
      sendToBackend(message);
      sendResponse({ ok: true });
      break;

    case "voice_task":
      sendToBackend(message);
      sendResponse({ ok: true });
      break;

    case "confirmation_response":
      sendToBackend(message);
      sendResponse({ ok: true });
      break;

    case "task_cancel":
      sendToBackend(message);
      sendResponse({ ok: true });
      break;

    case "task_feedback":
      sendToBackend(message);
      sendResponse({ ok: true });
      break;

    // content.js reports every page it loads; nothing to do with it here.
    // It used to fall through to "Unknown message type" on every page.
    case "page_info":
      sendResponse({ ok: true });
      break;

    case "get_ws_status":
      sendResponse({
        connected: ws && ws.readyState === WebSocket.OPEN,
      });
      break;

    case "reconnect":
      reconnectAttempts = 0;
      connectWebSocket();
      sendResponse({ ok: true });
      break;

    default:
      console.warn("[BG] Unknown message type:", message.type);
      sendResponse({ ok: false, error: "Unknown message type" });
  }

  return true; // Keep the message channel open for async response
});

// ============================================================================
// Extension Icon Click → Open Side Panel
// ============================================================================

chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ tabId: tab.id });
});

// Enable side panel for all tabs
chrome.sidePanel.setOptions({
  enabled: true,
});

// ============================================================================
// Service Worker Lifecycle
// ============================================================================

// Connect on install/startup
chrome.runtime.onInstalled.addListener(() => {
  console.log("[BG] Extension installed");
  connectWebSocket();
});

chrome.runtime.onStartup.addListener(() => {
  console.log("[BG] Extension started");
  connectWebSocket();
});

// Also try to connect immediately (for service worker wake-ups)
connectWebSocket();
