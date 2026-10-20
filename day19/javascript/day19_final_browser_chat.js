"use strict";

const connectionStatus = document.querySelector("#connection-status");
const requestStatus = document.querySelector("#request-status");
const messages = document.querySelector("#messages");
const form = document.querySelector("#question-form");
const input = document.querySelector("#question-input");
const sendButton = document.querySelector("#send-button");
const cancelButton = document.querySelector("#cancel-button");
const statusButton = document.querySelector("#status-button");

const CLIENT_ID_KEY = "day19-client-id";
const PROTOCOL_VERSION = "1.0";
const clientId = getOrCreateClientId();

let websocket = null;
let activeRequestId = null;
let activeAnswerElement = null;
let reconnectAttempt = 0;
let reconnectTimer = null;
let heartbeatTimer = null;

function getOrCreateClientId() {
  const saved = localStorage.getItem(CLIENT_ID_KEY);
  if (saved) return saved;

  const created = `browser-${crypto.randomUUID()}`;
  localStorage.setItem(CLIENT_ID_KEY, created);
  return created;
}

function addMessage(role, text, className) {
  const box = document.createElement("article");
  box.className = `message ${className}`;
  const label = document.createElement("span");
  label.className = "label";
  label.textContent = role;
  const content = document.createElement("span");
  content.textContent = text;
  box.append(label, content);
  messages.appendChild(box);
  messages.scrollTop = messages.scrollHeight;
  return content;
}

function renderHistory(history) {
  messages.replaceChildren();
  if (history.length === 0) {
    addMessage("系统", "当前会话还没有历史消息。", "system");
    return;
  }
  for (const item of history) {
    const isUser = item.role === "user";
    addMessage(isUser ? "你" : "助手", item.content, isUser ? "user" : "assistant");
  }
}

function socketIsOpen() {
  return websocket !== null && websocket.readyState === WebSocket.OPEN;
}

function updateControls() {
  const online = socketIsOpen();
  const busy = activeRequestId !== null;
  input.disabled = !online || busy;
  sendButton.disabled = !online || busy;
  cancelButton.disabled = !online || !busy;
  statusButton.disabled = !online;
  if (online && !busy) input.focus();
}

function command(type, requestId, payload = {}) {
  return {
    version: PROTOCOL_VERSION,
    type: type,
    request_id: requestId,
    payload: payload,
  };
}

function sendCommand(type, requestId, payload = {}) {
  if (!socketIsOpen()) return;
  websocket.send(JSON.stringify(command(type, requestId, payload)));
}

function finishRequest(requestId) {
  if (requestId === activeRequestId) activeRequestId = null;
  activeAnswerElement = null;
  updateControls();
}

function websocketUrl() {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}/ws?client_id=${encodeURIComponent(clientId)}`;
}

function startHeartbeat() {
  clearInterval(heartbeatTimer);
  heartbeatTimer = setInterval(() => {
    sendCommand("ping", `ping-${crypto.randomUUID()}`);
  }, 10000);
}

function scheduleReconnect() {
  if (reconnectTimer !== null) return;
  const seconds = Math.min(2 ** reconnectAttempt, 8);
  reconnectAttempt += 1;
  connectionStatus.className = "status waiting";
  connectionStatus.textContent = `连接中断，${seconds} 秒后重连……`;

  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, seconds * 1000);
}

function handleServerMessage(message) {
  const requestId = message.request_id;
  const payload = message.payload ?? {};

  if (message.type === "connected") {
    renderHistory(payload.history ?? []);
    addMessage(
      "系统",
      `会话 ${payload.client_id} 已连接；连接编号 ${message.connection_id}`,
      "system",
    );
  } else if (message.type === "queued") {
    requestStatus.textContent = "请求已经进入模型队列。";
  } else if (message.type === "start") {
    requestStatus.textContent = `模型正在回答：${requestId}`;
    activeAnswerElement = addMessage("助手", "", "assistant");
  } else if (message.type === "token") {
    if (activeAnswerElement) activeAnswerElement.textContent += payload.text;
    messages.scrollTop = messages.scrollHeight;
  } else if (message.type === "done") {
    requestStatus.textContent =
      `回答完成：${payload.chunk_count} 块；历史 ${payload.history_messages} 条。`;
    finishRequest(requestId);
  } else if (message.type === "cancel_requested") {
    requestStatus.textContent = "服务端已收到取消请求……";
  } else if (message.type === "cancelled") {
    requestStatus.textContent = "回答已取消，部分回答没有写入会话历史。";
    finishRequest(requestId);
  } else if (message.type === "status_result") {
    addMessage("服务状态", JSON.stringify(payload, null, 2), "system");
  } else if (message.type === "busy") {
    addMessage("提示", `当前运行中的请求：${payload.active_request_id}`, "warning");
  } else if (message.type === "error" || message.type === "cancel_not_found") {
    const description = payload.message ?? JSON.stringify(payload.details ?? payload);
    addMessage("错误", description, "error");
    if (requestId === activeRequestId) finishRequest(requestId);
  }
}

function connect() {
  if (websocket &&
      (websocket.readyState === WebSocket.OPEN ||
       websocket.readyState === WebSocket.CONNECTING)) return;

  connectionStatus.className = "status waiting";
  connectionStatus.textContent = "正在建立连接……";
  websocket = new WebSocket(websocketUrl());

  websocket.addEventListener("open", () => {
    reconnectAttempt = 0;
    connectionStatus.className = "status online";
    connectionStatus.textContent = "连接状态：已连接";
    startHeartbeat();
    updateControls();
  });

  websocket.addEventListener("message", (event) => {
    try {
      handleServerMessage(JSON.parse(event.data));
    } catch (error) {
      addMessage("前端错误", String(error), "error");
    }
  });

  websocket.addEventListener("close", () => {
    clearInterval(heartbeatTimer);
    if (activeRequestId !== null) {
      addMessage("系统", "连接中断，当前回答未完成且不会写入历史。", "warning");
      activeRequestId = null;
      activeAnswerElement = null;
    }
    updateControls();
    scheduleReconnect();
  });

  websocket.addEventListener("error", () => {
    connectionStatus.className = "status offline";
    connectionStatus.textContent = "连接发生错误。";
  });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question || activeRequestId !== null || !socketIsOpen()) return;

  activeRequestId = `req-${crypto.randomUUID()}`;
  addMessage("你", question, "user");
  requestStatus.textContent = "问题已发送……";
  updateControls();
  sendCommand("ask", activeRequestId, {question: question});
  input.value = "";
});

input.addEventListener("keydown", (event) => {
  if (event.ctrlKey && event.key === "Enter") form.requestSubmit();
});

cancelButton.addEventListener("click", () => {
  if (activeRequestId !== null) {
    sendCommand("cancel", activeRequestId);
    cancelButton.disabled = true;
  }
});

statusButton.addEventListener("click", () => {
  sendCommand("status", `status-${crypto.randomUUID()}`);
});

connect();
