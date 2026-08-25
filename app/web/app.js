"use strict";

const state = { mode: "answer", controller: null, lastAnswer: "" };
const byId = (id) => document.getElementById(id);
const form = byId("query-form");
const submitButton = byId("submit-button");
const submitLabel = byId("submit-label");
const answerCard = byId("answer-card");
const answerText = byId("answer-text");
const modelLabel = byId("model-label");
const emptyState = byId("empty-state");
const sourcesSection = byId("sources-section");
const sourcesList = byId("sources-list");
const sourceCount = byId("source-count");
const resultSummary = byId("result-summary");
const notice = byId("notice");
const copyButton = byId("copy-button");

document.querySelectorAll(".mode-tab").forEach((button) => {
  button.addEventListener("click", () => {
    state.mode = button.dataset.mode;
    document.querySelectorAll(".mode-tab").forEach((item) => {
      const active = item === button;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", String(active));
    });
    submitLabel.textContent = state.mode === "answer" ? "근거 기반 답변 생성" : "검색 근거 확인";
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const query = byId("query").value.trim();
  if (!query) {
    showError("질문을 입력해주세요.");
    byId("query").focus();
    return;
  }
  if (state.controller) state.controller.abort();
  state.controller = new AbortController();
  const requestMode = state.mode;
  setLoading(true);
  resetResult();

  const sourceType = byId("source-type").value;
  const payload = {
    source_type: sourceType,
    top_k: Number(byId("top-k").value),
    include_content: byId("include-content").checked,
    filters: buildFilters(sourceType),
  };
  if (requestMode === "answer") payload.question = query;
  else payload.query = query;

  try {
    const response = await fetch(requestMode === "answer" ? "/v1/answer" : "/v1/retrieve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: state.controller.signal,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `요청 실패 (${response.status})`);
    renderResult(data, requestMode);
  } catch (error) {
    if (error.name !== "AbortError") showError(error.message || "요청을 처리할 수 없습니다.");
  } finally {
    setLoading(false);
  }
});

copyButton.addEventListener("click", async () => {
  if (!state.lastAnswer) return;
  try {
    await navigator.clipboard.writeText(state.lastAnswer);
    copyButton.textContent = "복사 완료";
    window.setTimeout(() => { copyButton.textContent = "답변 복사"; }, 1400);
  } catch (_error) {
    showError("브라우저에서 Clipboard 접근을 허용하지 않았습니다.");
  }
});

async function checkHealth() {
  const dot = byId("health-dot");
  try {
    const response = await fetch("/health", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok) throw new Error("health check failed");
    dot.classList.add("ready");
    byId("health-label").textContent = "로컬 연결 정상";
    byId("equipment-label").textContent = data.equipment || "EquipmentRAG";
  } catch (_error) {
    dot.classList.add("error");
    byId("health-label").textContent = "연결 확인 필요";
    byId("equipment-label").textContent = "Local API unavailable";
  }
}

function buildFilters(sourceType) {
  const filters = {};
  if (sourceType === "code" || sourceType === "all") {
    const code = compact({
      class_name: byId("class-name").value,
      method_name: byId("method-name").value,
    });
    if (Object.keys(code).length) filters.code = code;
  }
  if (sourceType === "document" || sourceType === "all") {
    const documentFilter = compact({
      unit: byId("unit").value,
      document_type: byId("document-type").value,
      revision: byId("revision").value,
    });
    if (Object.keys(documentFilter).length) filters.document = documentFilter;
  }
  return filters;
}

function compact(values) {
  return Object.fromEntries(Object.entries(values).filter(([, value]) => value.trim() !== ""));
}

function renderResult(data, requestMode) {
  const sources = Array.isArray(data.sources) ? data.sources : [];
  emptyState.hidden = true;
  notice.hidden = true;
  resultSummary.textContent = `${sources.length}개의 근거를 찾았습니다.`;
  if (requestMode === "answer") {
    state.lastAnswer = data.answer || "";
    answerText.textContent = state.lastAnswer;
    modelLabel.textContent = data.model ? `MODEL · ${data.model}` : data.finish_reason || "";
    answerCard.hidden = false;
    copyButton.hidden = !state.lastAnswer;
  }
  renderSources(sources);
}

function renderSources(sources) {
  sourcesList.replaceChildren();
  sourceCount.textContent = String(sources.length);
  sourcesSection.hidden = sources.length === 0;
  sources.forEach((source) => {
    const card = element("article", "source-card");
    const topline = element("div", "source-topline");
    topline.append(
      textElement("span", "source-id", source.source_id || source.source_type || "SOURCE"),
      textElement("span", "source-score", `score ${formatScore(source.score)}`),
    );
    card.append(topline, textElement("h4", "", source.file_name || source.relative_path || "Unknown source"));
    const metadata = element("ul", "metadata");
    metadata.setAttribute("aria-label", "출처 메타데이터");
    sourceMetadata(source).forEach((value) => metadata.append(textElement("li", "", value)));
    card.append(metadata);
    const content = source.code || source.text;
    if (content) card.append(textElement("pre", "source-content", content));
    sourcesList.append(card);
  });
}

function sourceMetadata(source) {
  const values = [];
  if (source.source_type) values.push(source.source_type === "document" ? "문서" : "코드");
  if (source.class_name) values.push(`Class · ${source.class_name}`);
  if (source.method_name) values.push(`Method · ${source.method_name}`);
  if (source.start_line) values.push(`Line · ${source.start_line}-${source.end_line}`);
  if (source.document_type) values.push(source.document_type);
  if (source.revision) values.push(source.revision);
  if (source.section) values.push(`Section · ${source.section}`);
  if (source.page) values.push(`Page · ${source.page}`);
  if (source.slide) values.push(`Slide · ${source.slide}`);
  if (source.sheet) values.push(`Sheet · ${source.sheet}`);
  if (source.cell_range) values.push(`Cells · ${source.cell_range}`);
  if (source.relative_path) values.push(source.relative_path);
  return values;
}

function resetResult() {
  state.lastAnswer = "";
  answerCard.hidden = true;
  copyButton.hidden = true;
  sourcesSection.hidden = true;
  emptyState.hidden = true;
  notice.hidden = false;
  notice.classList.remove("error");
  notice.querySelector("strong").textContent = "검색 중";
  notice.querySelector("p").textContent = "로컬 Index에서 관련 근거를 확인하고 있습니다.";
}

function showError(message) {
  notice.hidden = false;
  notice.classList.add("error");
  notice.querySelector("strong").textContent = "요청을 완료하지 못했습니다";
  notice.querySelector("p").textContent = message;
  emptyState.hidden = true;
  resultSummary.textContent = "입력과 로컬 서비스 상태를 확인해주세요.";
}

function setLoading(loading) {
  submitButton.disabled = loading;
  submitLabel.textContent = loading
    ? "처리 중..."
    : state.mode === "answer"
      ? "근거 기반 답변 생성"
      : "검색 근거 확인";
}

function element(tag, className) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  return node;
}

function textElement(tag, className, value) {
  const node = element(tag, className);
  node.textContent = String(value ?? "");
  return node;
}

function formatScore(value) {
  return typeof value === "number" ? value.toFixed(4) : "-";
}

checkHealth();
