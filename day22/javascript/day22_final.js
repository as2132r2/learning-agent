"use strict";

const $ = (selector) => document.querySelector(selector);
const connection = $("#connection");
const messages = $("#messages");
const question = $("#question");
const askForm = $("#ask-form");
const askButton = $("#ask");
const cancelButton = $("#cancel");
const requestStatus = $("#request-status");
const rrfSummary = $("#rrf-summary");
const rerankSummary = $("#rerank-summary");
const rrfResults = $("#rrf-results");
const rerankResults = $("#rerank-results");
const uploadForm = $("#upload-form");
const fileInput = $("#file");
const uploadButton = $("#upload");
const uploadStatus = $("#upload-status");

let socket = null;
let activeRequestId = null;
let answerElement = null;
let reconnectAttempt = 0;

function appendMessage(role, text, kind) {
  const element = document.createElement("div");
  element.className = `message ${kind}`;
  element.textContent = `${role}：${text}`;
  messages.append(element);
  messages.scrollTop = messages.scrollHeight;
  return element;
}

function updateControls() {
  const online = socket && socket.readyState === WebSocket.OPEN;
  question.disabled = !online || Boolean(activeRequestId);
  askButton.disabled = !online || Boolean(activeRequestId);
  cancelButton.disabled = !online || !activeRequestId;
}

function send(type, requestId, payload = {}) {
  socket.send(JSON.stringify({
    version: "1.0",
    type,
    request_id: requestId,
    payload,
  }));
}

function finishRequest() {
  activeRequestId = null;
  answerElement = null;
  updateControls();
  question.focus();
}

function resultCard(item, mode) {
  const article = document.createElement("article");
  article.className = "result-card";

  const title = document.createElement("strong");
  title.textContent = `${item.rank}. ${item.location}`;

  const metrics = document.createElement("p");
  metrics.className = "metrics";
  if (mode === "rrf") {
    metrics.textContent =
      `向量排名 ${item.vector_rank ?? "—"} · ` +
      `关键词排名 ${item.keyword_rank ?? "—"} · ` +
      `RRF ${item.rrf_score}`;
  } else {
    metrics.textContent =
      `原RRF排名 ${item.rrf_rank} · 相关性 ${item.relevance_score}分`;
  }

  article.append(title, metrics);

  if (mode === "rerank") {
    const reason = document.createElement("p");
    reason.className = "reason";
    reason.textContent = item.reason;
    article.append(reason);
  }

  const content = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "查看正文";
  const body = document.createElement("p");
  body.className = "content";
  body.textContent = item.content;
  content.append(summary, body);
  article.append(content);
  return article;
}

function renderResults(container, items, mode) {
  container.replaceChildren();
  for (const item of items) {
    container.append(resultCard(item, mode));
  }
}

function handleMessage(message) {
  const payload = message.payload || {};

  if (message.type === "connected") {
    connection.textContent = `已连接 · 阈值 ${payload.threshold}`;
  } else if (message.type === "retrieving") {
    requestStatus.textContent = "正在并行执行向量检索和关键词检索……";
  } else if (message.type === "rrf_results") {
    renderResults(rrfResults, payload.results || [], "rrf");
    rrfSummary.textContent =
      `向量 ${payload.vector_count} 条，关键词 ${payload.keyword_count} 条；` +
      `关键词：${(payload.terms || []).join("、") || "无"}`;
    requestStatus.textContent = "RRF 初排完成，等待模型精排……";
  } else if (message.type === "rerank_queued") {
    requestStatus.textContent = "等待模型并发许可，准备执行 Reranker……";
  } else if (message.type === "reranked") {
    renderResults(rerankResults, payload.results || [], "rerank");
    rerankSummary.textContent =
      `最高 ${payload.best_score} 分，守门阈值 ${payload.threshold} 分`;
    requestStatus.textContent = "Reranker 完成，正在执行相关性守门……";
  } else if (message.type === "rejected") {
    appendMessage("系统", payload.message, "error");
    requestStatus.textContent =
      `已拒绝：最高 ${payload.best_score} 分 < 阈值 ${payload.threshold} 分。`;
    finishRequest();
  } else if (message.type === "start") {
    requestStatus.textContent =
      `守门通过，使用 ${payload.accepted_count} 条资料流式生成回答……`;
    answerElement = appendMessage("助手", "", "assistant");
  } else if (message.type === "token" && answerElement) {
    answerElement.textContent += payload.text;
    messages.scrollTop = messages.scrollHeight;
  } else if (message.type === "done") {
    requestStatus.textContent = "回答完成；请核对精排分数和引用来源。";
    finishRequest();
  } else if (message.type === "cancelled") {
    requestStatus.textContent = "当前任务已经取消。";
    finishRequest();
  } else if (["error", "busy", "cancel_not_found"].includes(message.type)) {
    appendMessage(
      "错误",
      payload.message || JSON.stringify(payload),
      "error",
    );
    if (message.request_id === activeRequestId) finishRequest();
  }
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws`);

  socket.onopen = () => {
    reconnectAttempt = 0;
    connection.textContent = "已连接";
    updateControls();
  };
  socket.onmessage = (event) => handleMessage(JSON.parse(event.data));
  socket.onclose = () => {
    connection.textContent = "已断开，自动重连中";
    activeRequestId = null;
    answerElement = null;
    updateControls();
    const delay = Math.min(2 ** reconnectAttempt++, 8) * 1000;
    setTimeout(connect, delay);
  };
  socket.onerror = () => {
    connection.textContent = "连接错误";
  };
}

askForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text || activeRequestId) return;

  activeRequestId = `req-${crypto.randomUUID()}`;
  appendMessage("你", text, "user");
  rrfResults.replaceChildren();
  rerankResults.replaceChildren();
  rrfSummary.textContent = "等待检索。";
  rerankSummary.textContent = "等待重排。";
  send("ask", activeRequestId, {question: text});
  question.value = "";
  updateControls();
});

cancelButton.addEventListener("click", () => {
  if (activeRequestId) send("cancel", activeRequestId);
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const selected = fileInput.files[0];
  if (!selected) return;

  const body = new FormData();
  body.append("document", selected);
  uploadButton.disabled = true;
  uploadStatus.textContent = "上传、校验并同步两套索引中……";

  try {
    const response = await fetch("/upload", {method: "POST", body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.message);
    uploadStatus.textContent =
      `${data.status}：${data.original_name}；` +
      `FTS 共 ${data.fts_chunk_count} 个文本块。`;
    uploadForm.reset();
  } catch (error) {
    uploadStatus.textContent = `上传失败：${error.message || error}`;
  } finally {
    uploadButton.disabled = false;
  }
});

connect();
