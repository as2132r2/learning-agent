const state = { selectedTraceId: null };

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

const formatMs = (value) => `${Number(value || 0).toFixed(1)} ms`;

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || data.error_message || `HTTP ${response.status}`);
  return data;
}

function renderMetrics(data) {
  const slowest = data.slowest_stage;
  const cards = [
    ["Trace 总数", data.total_traces],
    ["成功率", `${data.success_rate}%`],
    ["失败请求", data.failed_traces],
    ["平均总耗时", formatMs(data.average_duration_ms)],
    ["最慢平均阶段", slowest ? `${slowest.name} · ${formatMs(slowest.average_ms)}` : "暂无"],
  ];
  document.querySelector("#metrics").innerHTML = cards.map(([label, value]) => `
    <article class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></article>
  `).join("");
}

function renderTraceList(traces) {
  const container = document.querySelector("#trace-list");
  if (!traces.length) {
    container.innerHTML = '<div class="empty">还没有 Trace，请先运行请求。</div>';
    return;
  }
  container.innerHTML = traces.map((trace) => `
    <button class="trace-item ${trace.trace_id === state.selectedTraceId ? "active" : ""}"
            data-trace-id="${escapeHtml(trace.trace_id)}">
      <div class="trace-top">
        <span class="trace-id">${escapeHtml(trace.trace_id)}</span>
        <span class="badge ${escapeHtml(trace.status)}">${escapeHtml(trace.status)}</span>
      </div>
      <div class="question">${escapeHtml(trace.question)}</div>
      <div class="meta">${formatMs(trace.duration_ms)} · ${escapeHtml(trace.started_at)}</div>
    </button>
  `).join("");

  container.querySelectorAll(".trace-item").forEach((button) => {
    button.addEventListener("click", () => selectTrace(button.dataset.traceId));
  });
}

function renderTraceDetail(data) {
  const { trace, spans, diagnostics } = data;
  const rootDuration = Math.max(Number(trace.duration_ms || 0), 1);
  const detail = document.querySelector("#trace-detail");
  const errorBlock = trace.error_type
    ? `<div class="diagnostic"><strong>${escapeHtml(trace.error_type)}</strong><br>${escapeHtml(trace.error_message)}</div>`
    : `<div class="diagnostic">最慢阶段：<strong>${escapeHtml(diagnostics.slowest_stage || "暂无")}</strong> · ${formatMs(diagnostics.slowest_duration_ms)}</div>`;

  detail.innerHTML = `
    <div class="trace-summary">
      <div class="summary-cell"><span>TRACE ID</span><strong>${escapeHtml(trace.trace_id)}</strong></div>
      <div class="summary-cell"><span>REQUEST ID</span><strong>${escapeHtml(trace.request_id)}</strong></div>
      <div class="summary-cell"><span>总耗时 / 状态</span><strong>${formatMs(trace.duration_ms)} · ${escapeHtml(trace.status)}</strong></div>
    </div>
    ${errorBlock}
    <div class="span-list">
      ${spans.map((span) => {
        const width = Math.min(100, Math.max(1, Number(span.duration_ms) / rootDuration * 100));
        return `
          <article class="span-card ${span.status === "error" ? "error" : ""}">
            <div class="span-head">
              <div><span class="span-kind">${escapeHtml(span.kind)}</span><div class="span-name">${escapeHtml(span.name)}</div></div>
              <strong>${formatMs(span.duration_ms)}</strong>
            </div>
            <div class="bar-track"><div class="bar" style="width:${width}%"></div></div>
            <div class="attributes">span=${escapeHtml(span.span_id)}<br>parent=${escapeHtml(span.parent_span_id || "ROOT")}<br>${escapeHtml(JSON.stringify(span.attributes))}${span.error_type ? `<br>error=${escapeHtml(span.error_type)}: ${escapeHtml(span.error_message)}` : ""}</div>
          </article>`;
      }).join("")}
    </div>`;
}

async function selectTrace(traceId) {
  state.selectedTraceId = traceId;
  const data = await fetchJson(`/api/traces/${encodeURIComponent(traceId)}`);
  renderTraceDetail(data);
  document.querySelectorAll(".trace-item").forEach((item) => {
    item.classList.toggle("active", item.dataset.traceId === traceId);
  });
}

async function refreshAll(selectNewest = false) {
  const [overview, traceData] = await Promise.all([
    fetchJson("/api/overview"),
    fetchJson("/api/traces"),
  ]);
  renderMetrics(overview);
  renderTraceList(traceData.traces);
  document.querySelector("#connection").textContent = "服务在线";
  document.querySelector("#connection").className = "status online";

  if (selectNewest && traceData.traces.length) {
    await selectTrace(traceData.traces[0].trace_id);
  } else if (state.selectedTraceId) {
    await selectTrace(state.selectedTraceId);
  }
}

document.querySelector("#ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = document.querySelector("#submit-button");
  const output = document.querySelector("#request-result");
  button.disabled = true;
  output.textContent = "正在运行 RAG 流水线……";
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: document.querySelector("#question").value,
        simulate_error: document.querySelector("#simulate-error").checked,
      }),
    });
    const data = await response.json();
    output.textContent = JSON.stringify(data, null, 2);
    if (data.trace_id) state.selectedTraceId = data.trace_id;
    await refreshAll(false);
  } catch (error) {
    output.textContent = `请求失败：${error.message}`;
  } finally {
    button.disabled = false;
  }
});

document.querySelector("#refresh-button").addEventListener("click", () => {
  refreshAll(false).catch(showConnectionError);
});

function showConnectionError(error) {
  const connection = document.querySelector("#connection");
  connection.textContent = `连接失败：${error.message}`;
  connection.className = "status offline";
}

refreshAll(true).catch(showConnectionError);
setInterval(() => refreshAll(false).catch(showConnectionError), 5000);
