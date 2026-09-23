const state = { status: "verified", query: "", photos: [], selected: null, detail: null, summary: null, requestId: 0 };
const $ = (id) => document.getElementById(id);
const labels = { all: "Все", verified: "Подтверждены", unknown: "Нет карточки", ambiguous: "Неоднозначны" };
const statusText = { verified: "Подтверждён", unknown: "Нет точной карточки", ambiguous: "Неоднозначно" };

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined && text !== null) element.textContent = String(text);
  return element;
}
function fieldList(target, entries) {
  target.replaceChildren();
  for (const [key, value] of entries) {
    const dt = node("dt", "", key);
    const dd = node("dd", "", value === null || value === undefined || value === "" ? "—" : value);
    target.append(dt, dd);
  }
}
function shortHash(hash) { return hash ? `${hash.slice(0, 12)}…${hash.slice(-8)}` : "—"; }
function score(value) { return Number(value).toFixed(4); }
function updateUrl() {
  const url = new URL(window.location.href);
  url.searchParams.set("status", state.status);
  if (state.selected) url.searchParams.set("photo", state.selected);
  else url.searchParams.delete("photo");
  history.replaceState(null, "", url);
}
function renderTabs() {
  const holder = $("statusTabs");
  holder.replaceChildren();
  for (const status of ["verified", "unknown", "ambiguous", "all"]) {
    const tab = node("button", `status-tab ${state.status === status ? "active" : ""}`);
    tab.type = "button";
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", state.status === status ? "true" : "false");
    tab.append(node("span", "", labels[status]), node("strong", "", status === "all" ? state.summary.total : state.summary.counts[status] || 0));
    tab.addEventListener("click", () => { state.status = status; renderTabs(); loadPhotos().catch(showError); });
    holder.append(tab);
  }
}
function renderList() {
  $("resultCount").textContent = String(state.photos.length);
  const holder = $("photoList");
  holder.replaceChildren();
  if (!state.photos.length) {
    holder.append(node("p", "muted", "Ничего не найдено. Попробуйте другой запрос."));
    return;
  }
  for (const item of state.photos) {
    const button = node("button", `photo-item ${state.selected === item.name ? "active" : ""}`);
    button.type = "button";
    button.title = item.name;
    const thumb = node("img");
    thumb.src = item.thumb_url;
    thumb.alt = "";
    thumb.loading = "lazy";
    const copy = node("div", "photo-item-content");
    copy.append(node("div", "photo-item-title", item.name),
                node("div", "photo-item-desc", item.evidence),
                node("div", `photo-item-state ${item.status}`, statusText[item.status]));
    button.append(thumb, copy);
    button.addEventListener("click", () => selectPhoto(item.name));
    holder.append(button);
  }
}
async function loadPhotos(preferred) {
  const requestId = ++state.requestId;
  const params = new URLSearchParams({ status: state.status, q: state.query });
  const response = await fetch(`/api/photos?${params}`);
  if (!response.ok) throw new Error(`Не удалось загрузить список: HTTP ${response.status}`);
  const result = await response.json();
  if (requestId !== state.requestId) return;
  state.photos = result.photos;
  const names = new Set(state.photos.map((item) => item.name));
  state.selected = names.has(preferred) ? preferred : names.has(state.selected) ? state.selected : state.photos[0]?.name || null;
  renderList();
  if (state.selected) await selectPhoto(state.selected, false);
  else { $("detail").hidden = true; $("loading").hidden = false; $("loading").textContent = "По этому запросу снимков нет."; updateUrl(); }
}
async function selectPhoto(name, refreshList = true) {
  if (!state.photos.some((item) => item.name === name)) return;
  const requestId = ++state.requestId;
  state.selected = name;
  if (refreshList) renderList();
  $("loading").hidden = false;
  $("loading").textContent = "Открываю снимок…";
  const response = await fetch(`/api/photos/${encodeURIComponent(name)}`);
  if (!response.ok) throw new Error(`Не удалось открыть фото: HTTP ${response.status}`);
  const detail = await response.json();
  if (requestId !== state.requestId) return;
  state.detail = detail;
  renderDetail();
  updateUrl();
  $("loading").hidden = true;
  $("detail").hidden = false;
}
function renderReference(candidate, verifiedOnly = false) {
  const detail = state.detail;
  const card = verifiedOnly ? detail.verified_card : candidate.card;
  const url = verifiedOnly ? detail.verified_reference_url : candidate.reference_url;
  $("referenceImage").hidden = !url;
  $("noReference").hidden = !!url;
  $("referenceLink").hidden = !url;
  if (url) { $("referenceImage").src = url; $("referenceLink").href = url; }
  $("referenceHeading").textContent = verifiedOnly ? "ПОДТВЕРЖДЁННЫЙ ЭТАЛОН" : `ЭТАЛОН · МЕСТО ${candidate.rank} ИЗ 20`;
  $("referenceCaption").textContent = card ? `${card.winery} · ${card.name}` : "";
  $("referenceScore").textContent = verifiedOnly ? "вне Top-20" : score(candidate.score);
  $("candidateName").textContent = card?.name || "—";
  $("candidateWinery").textContent = card?.winery || "";
  fieldList($("candidateMeta"), [
    ["slug", verifiedOnly ? detail.slug : candidate.slug],
    ["Ранг", verifiedOnly ? "вне Top-20" : `${candidate.rank} / 20`],
    ["Сходство", verifiedOnly ? "—" : score(candidate.score)],
    ["Отставание от #1", verifiedOnly ? "—" : score(candidate.gap_to_first)],
    ["Категория", card?.category], ["Сорт", card?.grape], ["Регион", card?.region],
    ["Цвет", card?.color], ["Описание", card?.description],
    ["Эталон SHA", verifiedOnly ? "см. каталог" : shortHash(candidate.reference_sha256)],
  ]);
  document.querySelectorAll(".candidate").forEach((button) => {
    button.classList.toggle("selected", !verifiedOnly && Number(button.dataset.rank) === candidate.rank);
  });
}
function renderCandidates() {
  const detail = state.detail;
  const grid = $("candidateGrid");
  grid.replaceChildren();
  for (const candidate of detail.candidates) {
    const isMatch = detail.slug === candidate.slug;
    const button = node("button", `candidate ${isMatch ? "verified-match" : ""}`);
    button.type = "button";
    button.dataset.rank = candidate.rank;
    button.setAttribute("aria-label", `Место ${candidate.rank}: ${candidate.card.name}, ${candidate.card.winery}, сходство ${score(candidate.score)}`);
    const imageBox = node("div", "candidate-image");
    const image = node("img");
    image.src = candidate.reference_url;
    image.alt = "";
    image.loading = "lazy";
    imageBox.append(image, node("span", "candidate-rank", `#${candidate.rank}`));
    const copy = node("div", "candidate-copy");
    copy.append(node("div", "candidate-name", candidate.card.name),
                node("div", "candidate-winery", candidate.card.winery),
                node("div", "candidate-score mono", score(candidate.score)),
                node("div", "candidate-gap mono", `Δ ${score(candidate.gap_to_first)}`));
    if (isMatch) copy.append(node("div", "candidate-match", "✓ Подтверждённая карточка"));
    button.append(imageBox, copy);
    button.addEventListener("click", () => {
      renderReference(candidate);
      document.querySelector(".compare-grid").scrollIntoView({ behavior: "smooth", block: "start" });
    });
    grid.append(button);
  }
}
function renderDetail() {
  const detail = state.detail;
  const index = state.photos.findIndex((item) => item.name === detail.name);
  $("position").textContent = `${index + 1} / ${state.photos.length}`;
  $("photoTitle").textContent = detail.name;
  $("decisionStatus").className = `badge ${detail.status}`;
  $("decisionStatus").textContent = statusText[detail.status];
  $("decisionText").textContent = detail.status === "verified" ? detail.slug : detail.evidence;
  $("scoreTop1").textContent = score(detail.top1.score);
  $("scoreGap").textContent = score(detail.gap_first_second);
  $("scoreSpread").textContent = score(detail.spread_top20);
  $("photoImage").src = detail.photo_url;
  $("originalLink").href = detail.photo_url;
  $("photoHash").textContent = `SHA-256 ${shortHash(detail.sha256)}`;
  $("evidence").textContent = detail.evidence;
  fieldList($("decisionMeta"), [
    ["Статус", statusText[detail.status]], ["slug", detail.slug],
    ["Место в Top-20", detail.slug ? detail.verified_rank || "вне Top-20" : "—"],
    ["Top-1", detail.top1.slug], ["Балл Top-1", score(detail.top1.score)],
    ["Отрыв #1–#2", score(detail.gap_first_second)],
    ["Разброс Top-20", score(detail.spread_top20)],
    ["Фото SHA-256", detail.sha256],
  ]);
  const lines = $("ocrLines");
  lines.replaceChildren();
  $("ocrCount").textContent = `(${detail.ocr_lines.length})`;
  if (!detail.ocr_lines.length) lines.append(node("span", "muted", "Текст не распознан"));
  for (const line of detail.ocr_lines) {
    const token = node("span", "ocr-token", line.text);
    token.append(node("small", "mono", `${Math.round(line.confidence * 100)}%`));
    lines.append(token);
  }
  $("rankNote").textContent = detail.slug ?
    (detail.verified_rank ? `Подтверждённая карточка на месте ${detail.verified_rank}` : "Подтверждённой карточки нет в Top-20") :
    "Кандидаты — гипотезы, не метки";
  renderCandidates();
  const chosen = detail.candidates.find((item) => item.slug === detail.slug);
  if (detail.slug && !chosen && detail.verified_reference_url) renderReference(null, true);
  else renderReference(chosen || detail.candidates[0]);
  $("prevButton").disabled = index <= 0;
  $("nextButton").disabled = index >= state.photos.length - 1;
}
function move(direction) {
  const index = state.photos.findIndex((item) => item.name === state.selected);
  const next = state.photos[index + direction];
  if (next) selectPhoto(next.name);
}
async function init() {
  const params = new URLSearchParams(window.location.search);
  const requestedStatus = params.get("status");
  if (labels[requestedStatus]) state.status = requestedStatus;
  const requestedPhoto = params.get("photo");
  const response = await fetch("/api/summary");
  if (!response.ok) throw new Error(`Не удалось загрузить сводку: HTTP ${response.status}`);
  state.summary = await response.json();
  $("datasetHash").textContent = `labels ${state.summary.provenance.labels_sha256.slice(0, 12)}`;
  fieldList($("provenanceHashes"), Object.entries(state.summary.provenance).map(([key, value]) => [key, value]));
  renderTabs();
  $("search").addEventListener("input", (event) => { state.query = event.target.value; loadPhotos().catch(showError); });
  $("prevButton").addEventListener("click", () => move(-1));
  $("nextButton").addEventListener("click", () => move(1));
  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement) return;
    if (event.key === "ArrowLeft") move(-1);
    if (event.key === "ArrowRight") move(1);
  });
  await loadPhotos(requestedPhoto);
}
function showError(error) {
  $("detail").hidden = true;
  $("loading").hidden = false;
  $("loading").textContent = `Ошибка: ${error.message}`;
  console.error(error);
}
init().catch(showError);
