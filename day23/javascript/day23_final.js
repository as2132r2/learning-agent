"use strict";

const $ = (selector) => document.querySelector(selector);
const connection = $("#connection");
const messages = $("#messages");
const status = $("#request-status");
const question = $("#question");
const askForm = $("#ask-form");
const askButton = $("#ask");
const cancelButton = $("#cancel");
const rrfResults = $("#rrf-results");
const rerankResults = $("#rerank-results");
const rrfSummary = $("#rrf-summary");
const rerankSummary = $("#rerank-summary");
const versions = $("#versions");
const uploadForm = $("#upload-form");
const fileInput = $("#file");
const uploadButton = $("#upload");
const uploadStatus = $("#upload-status");

let socket = null;
let activeRequestId = null;
let reconnectAttempt = 0;

function addMessage(role, text, kind) {
  const box = document.createElement("div");
  box.className = `message ${kind}`;
  box.textContent = `${role}：${text}`;
  messages.append(box);
  messages.scrollTop = messages.scrollHeight;
}

function updateControls() {
  const online = socket && socket.readyState === WebSocket.OPEN;
  question.disabled = !online || Boolean(activeRequestId);
  askButton.disabled = !online || Boolean(activeRequestId);
  cancelButton.disabled = !online || !activeRequestId;
}

function finish() {
  activeRequestId = null;
  updateControls();
  question.focus();
}

function send(type, requestId, payload = {}) {
  socket.send(JSON.stringify({
    version: "1.0",
    type,
    request_id: requestId,
    payload,
  }));
}

function retrievalCard(item, reranked = false) {
  const card = document.createElement("div");
  card.className = "result-card";
  const title = document.createElement("strong");
  title.textContent = `${item.rank}. ${item.location}`;
  const metrics = document.createElement("p");
  metrics.className = "metrics";
  metrics.textContent = reranked
    ? `相关性 ${item.relevance_score} · 原RRF排名 ${item.rrf_rank}`
    : `向量 ${item.vector_rank ?? "—"} · 关键词 ${item.keyword_rank ?? "—"} · RRF ${item.rrf_score}`;
  card.append(title, metrics);
  if (reranked) {
    const reason = document.createElement("p");
    reason.textContent = item.reason;
    card.append(reason);
  }
  return card;
}

function renderRetrieval(container, items, reranked = false) {
  container.replaceChildren();
  for (const item of items) container.append(retrievalCard(item, reranked));
}

function metric(label, value, good) {
  const box = document.createElement("div");
  box.className = `metric ${good ? "good" : "bad"}`;
  box.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
  return box;
}

function renderVersion(data) {
  const article = document.createElement("article");
  article.className = `version ${data.passed ? "passed" : "failed"}`;
  const heading = document.createElement("h3");
  heading.textContent = `版本${data.version_number} · ${data.label} · ${data.passed ? "通过" : "未通过"}`;
  const answer = document.createElement("pre");
  answer.textContent = data.rendered_answer;
  const metrics = document.createElement("div");
  metrics.className = "metric-grid";
  metrics.append(
    metric("引用覆盖率", `${Math.round(data.citation.coverage_rate * 100)}%`, data.citation.passed),
    metric("证据支持率", data.groundedness ? `${Math.round(data.groundedness.rate * 100)}%` : "—", data.groundedness?.passed),
    metric("问题相关性", data.quality?.relevance_score ?? "—", (data.quality?.relevance_score ?? 0) >= 80),
    metric("回答完整性", data.quality?.completeness_score ?? "—", (data.quality?.completeness_score ?? 0) >= 80),
    metric("综合最短板", data.final?.overall_score ?? "—", data.final?.passed),
  );
  article.append(heading, answer, metrics);

  if (data.quality) {
    const detail = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "查看遗漏、偏题与评估说明";
    const text = document.createElement("pre");
    text.textContent =
      `遗漏：${JSON.stringify(data.quality.missing_points, null, 2)}\n` +
      `偏题：${JSON.stringify(data.quality.irrelevant_points, null, 2)}\n` +
      `说明：${data.quality.explanation}\n` +
      `错误：${JSON.stringify(data.final?.errors || [], null, 2)}`;
    detail.append(summary, text);
    article.append(detail);
  }
  versions.append(article);
}

function handle(message) {
  const payload = message.payload || {};
  if (message.type === "connected") {
    connection.textContent = `已连接 · 阈值${payload.rerank_threshold} · 最多修复${payload.max_repair_attempts}次`;
  } else if (message.type === "retrieving") {
    status.textContent = "正在执行向量检索、关键词检索和RRF融合……";
  } else if (message.type === "rrf_results") {
    renderRetrieval(rrfResults, payload.results || []);
    rrfSummary.textContent = `向量${payload.vector_count}条，关键词${payload.keyword_count}条`;
  } else if (message.type === "pipeline_queued") {
    status.textContent = "等待模型流水线并发许可……";
  } else if (message.type === "reranked") {
    renderRetrieval(rerankResults, payload.results || [], true);
    rerankSummary.textContent = `最高${payload.best_score}分，通过资料${payload.accepted_count}条`;
    status.textContent = "精排完成，准备生成结构化初稿……";
  } else if (message.type === "drafting") {
    status.textContent = "正在生成带结论—引用关系的初稿……";
  } else if (message.type === "auditing") {
    status.textContent = `正在审计版本${payload.version}的引用、忠实度和完整性……`;
  } else if (message.type === "version_assessed") {
    renderVersion(payload);
    status.textContent = payload.passed ? "该版本通过全部质量检查。" : "该版本未通过，准备受控修复。";
  } else if (message.type === "repairing") {
    status.textContent = `正在执行第${payload.attempt}/${payload.maximum}次修复……`;
  } else if (message.type === "done") {
    addMessage("助手", payload.answer, "assistant");
    status.textContent = `最终回答通过质量门控；实际修复${payload.repair_attempts}次。`;
    finish();
  } else if (message.type === "retrieval_rejected" || message.type === "quality_rejected") {
    addMessage("系统", payload.message, "error");
    status.textContent = "系统拒绝返回未经验证的回答。";
    finish();
  } else if (message.type === "cancelled") {
    status.textContent = "任务已取消。";
    finish();
  } else if (["error", "busy", "cancel_not_found"].includes(message.type)) {
    addMessage("错误", payload.message || JSON.stringify(payload), "error");
    if (message.request_id === activeRequestId) finish();
  }
}

function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  socket = new WebSocket(`${protocol}://${location.host}/ws`);
  socket.onopen = () => { reconnectAttempt = 0; connection.textContent = "已连接"; updateControls(); };
  socket.onmessage = (event) => handle(JSON.parse(event.data));
  socket.onclose = () => {
    connection.textContent = "已断开，自动重连";
    activeRequestId = null;
    updateControls();
    setTimeout(connect, Math.min(2 ** reconnectAttempt++, 8) * 1000);
  };
  socket.onerror = () => { connection.textContent = "连接错误"; };
}

askForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (!text || activeRequestId) return;
  activeRequestId = `req-${crypto.randomUUID()}`;
  addMessage("你", text, "user");
  rrfResults.replaceChildren();
  rerankResults.replaceChildren();
  versions.replaceChildren();
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
  uploadStatus.textContent = "上传并同步向量/关键词索引中……";
  try {
    const response = await fetch("/upload", {method: "POST", body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.message);
    uploadStatus.textContent = `${data.status}：${data.original_name}，FTS文本块${data.fts_chunk_count}个。`;
    uploadForm.reset();
  } catch (error) {
    uploadStatus.textContent = `上传失败：${error.message || error}`;
  } finally {
    uploadButton.disabled = false;
  }
});

connect();
