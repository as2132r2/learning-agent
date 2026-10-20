"use strict";

const $ = selector => document.querySelector(selector);
const connection = $("#connection"), messages = $("#messages"), statusBox = $("#status");
const question = $("#question"), askForm = $("#ask-form"), askBtn = $("#ask"), cancelBtn = $("#cancel");
const userId = $("#user-id"), sessionId = $("#session-id"), risk = $("#risk");
const maxCalls = $("#max-calls"), maxTokens = $("#max-tokens"), downgrade = $("#downgrade");
const plan = $("#plan"), metrics = $("#metrics"), rrf = $("#rrf"), reranked = $("#reranked"), versions = $("#versions");
const uploadForm = $("#upload-form"), file = $("#file"), uploadBtn = $("#upload"), uploadStatus = $("#upload-status");
const feedbackPanel = $("#feedback-panel"), feedbackReason = $("#feedback-reason"), feedbackStatus = $("#feedback-status");

let ws = null, active = null, retry = 0, feedbackRequestId = null, feedbackUserId = null;

function addMessage(role, text, kind) {
  const element = document.createElement("div");
  element.className = `message ${kind}`;
  element.textContent = `${role}：${text}`;
  messages.append(element);
  messages.scrollTop = messages.scrollHeight;
}

function updateControls() {
  const online = ws && ws.readyState === WebSocket.OPEN;
  question.disabled = !online || Boolean(active);
  askBtn.disabled = !online || Boolean(active);
  cancelBtn.disabled = !online || !active;
}

function finishRequest() {
  active = null;
  updateControls();
  question.focus();
}

function send(type, requestId, payload) {
  ws.send(JSON.stringify({version: "1.0", type, request_id: requestId, payload}));
}

function resultCard(item, mode) {
  const element = document.createElement("article");
  element.className = "result";
  const title = document.createElement("strong");
  title.textContent = `${item.rank}. ${item.location}`;
  const detail = document.createElement("p");
  detail.textContent = mode === "rrf"
    ? `向量 ${item.vector_rank ?? "—"} · 关键词 ${item.keyword_rank ?? "—"} · RRF ${item.rrf_score}`
    : `相关性 ${item.relevance_score} · ${item.reason}`;
  element.append(title, detail);
  return element;
}

function renderResults(container, items, mode) {
  container.replaceChildren();
  for (const item of items) container.append(resultCard(item, mode));
}

function metric(label, value) {
  const element = document.createElement("div");
  element.className = "metric";
  const name = document.createElement("span"), number = document.createElement("strong");
  name.textContent = label;
  number.textContent = value;
  element.append(name, number);
  return element;
}

function showMetrics(payload) {
  metrics.replaceChildren(
    metric("缓存命中", payload.cache_hits),
    metric("真实调用", payload.model_calls),
    metric("实际 Token", payload.total_tokens),
    metric("墙钟耗时", `${payload.wall_seconds}s`),
    metric("预算预留", `${payload.reserved_calls}次/${payload.reserved_tokens}`),
    metric("预算实耗", `${payload.actual_calls}次/${payload.actual_tokens}`)
  );
  const details = document.createElement("details"), summary = document.createElement("summary"), pre = document.createElement("pre");
  summary.textContent = "查看逐阶段指标";
  pre.textContent = JSON.stringify(payload.stages, null, 2);
  details.append(summary, pre);
  metrics.append(details);
}

function showVersion(payload) {
  const element = document.createElement("article");
  element.className = `version ${payload.passed ? "passed" : "failed"}`;
  const title = document.createElement("h3"), answer = document.createElement("pre"), scores = document.createElement("p");
  title.textContent = `版本 ${payload.version} · ${payload.label} · ${payload.passed ? "通过" : "失败"}`;
  answer.textContent = payload.answer;
  scores.textContent = `引用 ${Math.round(payload.citation_coverage * 100)}% · 忠实度 ${payload.groundedness === null ? "未执行" : Math.round(payload.groundedness * 100) + "%"} · 相关性 ${payload.relevance ?? "未执行"} · 完整性 ${payload.completeness ?? "未执行"}`;
  element.append(title, answer, scores);
  versions.append(element);
}

function prepareFeedback(requestId) {
  feedbackRequestId = requestId;
  feedbackUserId = userId.value.trim();
  feedbackReason.value = "";
  feedbackStatus.textContent = "请选择评价。";
  feedbackStatus.className = "";
  feedbackPanel.hidden = false;
  for (const button of feedbackPanel.querySelectorAll("[data-rating]")) button.classList.remove("selected");
}

async function submitFeedback(rating, button) {
  if (!feedbackRequestId || !feedbackUserId) return;
  feedbackStatus.textContent = "正在保存评价……";
  feedbackStatus.className = "";
  for (const item of feedbackPanel.querySelectorAll("[data-rating]")) item.disabled = true;
  try {
    const response = await fetch("/feedback", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        request_id: feedbackRequestId,
        user_id: feedbackUserId,
        rating,
        reason: feedbackReason.value.trim() || null
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "保存失败");
    for (const item of feedbackPanel.querySelectorAll("[data-rating]")) item.classList.remove("selected");
    button.classList.add("selected");
    feedbackStatus.textContent = `${data.operation === "created" ? "评价已保存" : "评价已更新"} · ${data.feedback_id}`;
    feedbackStatus.className = "success";
  } catch (error) {
    feedbackStatus.textContent = `保存失败：${error.message || error}`;
    feedbackStatus.className = "failure";
  } finally {
    for (const item of feedbackPanel.querySelectorAll("[data-rating]")) item.disabled = false;
  }
}

function handle(message) {
  const payload = message.payload || {};
  if (message.type === "connected") {
    connection.textContent = `已连接 · 缓存 ${payload.cache_entries} 条 · 反馈已启用`;
  } else if (message.type === "plan") {
    plan.textContent = JSON.stringify(payload, null, 2);
    statusBox.textContent = "预算预检通过，开始检索……";
  } else if (message.type === "retrieving") {
    statusBox.textContent = "正在执行混合检索……";
  } else if (message.type === "rrf_results") {
    renderResults(rrf, payload.results || [], "rrf");
  } else if (message.type === "reranked") {
    renderResults(reranked, payload.results || [], "rerank");
    statusBox.textContent = `精排完成，通过资料 ${payload.accepted_count} 条。`;
  } else if (message.type === "version_assessed") {
    showVersion(payload);
  } else if (message.type === "repairing") {
    statusBox.textContent = "初稿未通过，正在修复一次……";
  } else if (message.type === "done") {
    addMessage("助手", payload.answer, "assistant");
    showMetrics(payload);
    prepareFeedback(message.request_id);
    statusBox.textContent = "回答与追踪数据已保存，现在可以评价。";
    finishRequest();
  } else if (["budget_rejected", "budget_exceeded", "retrieval_rejected", "quality_rejected"].includes(message.type)) {
    addMessage("系统", payload.message || "请求被质量或预算策略拒绝。", "error");
    if (payload.stages) showMetrics(payload);
    finishRequest();
  } else if (message.type === "cancelled") {
    statusBox.textContent = "任务已取消。";
    finishRequest();
  } else if (["error", "busy", "cancel_not_found"].includes(message.type)) {
    addMessage("错误", payload.message || JSON.stringify(payload), "error");
    if (message.request_id === active) finishRequest();
  }
}

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onopen = () => { retry = 0; connection.textContent = "已连接"; updateControls(); };
  ws.onmessage = event => handle(JSON.parse(event.data));
  ws.onclose = () => {
    connection.textContent = "已断开，自动重连";
    active = null;
    updateControls();
    setTimeout(connect, Math.min(2 ** retry++, 8) * 1000);
  };
  ws.onerror = () => { connection.textContent = "连接错误"; };
}

askForm.onsubmit = event => {
  event.preventDefault();
  const text = question.value.trim(), uid = userId.value.trim(), sid = sessionId.value.trim();
  if (!text || !uid || !sid || active) return;
  active = `req-${crypto.randomUUID()}`;
  addMessage("你", text, "user");
  rrf.replaceChildren(); reranked.replaceChildren(); versions.replaceChildren(); metrics.replaceChildren();
  feedbackPanel.hidden = true;
  send("ask", active, {
    question: text,
    session_id: sid,
    user_id: uid,
    risk: risk.value,
    max_calls: Number(maxCalls.value),
    max_tokens: Number(maxTokens.value),
    allow_downgrade: downgrade.checked
  });
  question.value = "";
  updateControls();
};

cancelBtn.onclick = () => active && send("cancel", active, {
  question: null,
  session_id: sessionId.value.trim(),
  user_id: userId.value.trim()
});

for (const button of feedbackPanel.querySelectorAll("[data-rating]")) {
  button.onclick = () => submitFeedback(button.dataset.rating, button);
}

uploadForm.onsubmit = async event => {
  event.preventDefault();
  const selected = file.files[0];
  if (!selected) return;
  const body = new FormData();
  body.append("document", selected);
  uploadBtn.disabled = true;
  uploadStatus.textContent = "上传并同步双索引中……";
  try {
    const response = await fetch("/upload", {method: "POST", body});
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || data.error);
    uploadStatus.textContent = `${data.status}：${data.original_name}，FTS 块数 ${data.fts_chunk_count}`;
    uploadForm.reset();
  } catch (error) {
    uploadStatus.textContent = `上传失败：${error.message || error}`;
  } finally {
    uploadBtn.disabled = false;
  }
};

connect();
