"use strict";
const $ = (id) => document.getElementById(id);
const state = { convId: null, streaming: null, messages: [], modes: [], status: null };

async function api(path, opts = {}) {
  if (opts.json !== undefined) {
    opts.body = JSON.stringify(opts.json);
    opts.headers = { "Content-Type": "application/json" };
  }
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch (_) { /* not JSON */ }
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.status === 204 ? null : res.json();
}

const el = (tag, props = {}, ...kids) => {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...kids);
  return node;
};

// ---------------------------------------------------------------- rendering
marked.setOptions({ gfm: true, breaks: false });

function renderMarkdown(text, sources) {
  const n = sources ? sources.length : 0;
  const html = DOMPurify.sanitize(marked.parse(text || ""), { FORBID_TAGS: ["style", "form"] });
  const box = el("div", { className: "body" });
  box.innerHTML = html;
  box.querySelectorAll("a").forEach((a) => { a.target = "_blank"; a.rel = "noopener noreferrer"; });
  box.querySelectorAll("pre code").forEach((code) => {
    try { hljs.highlightElement(code); } catch (_) { /* unknown language */ }
    const btn = el("button", { className: "copy-code", textContent: "Copy", type: "button" });
    btn.onclick = () => copy(code.innerText, btn);
    code.parentElement.append(btn);
  });
  if (sources) {  // turn [S1] markers into chips (text nodes outside code only)
    const walker = document.createTreeWalker(box, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) {
      const t = walker.currentNode;
      if (/\[S\d+\]/.test(t.nodeValue) && !t.parentElement.closest("pre, code")) nodes.push(t);
    }
    for (const t of nodes) {
      const frag = document.createDocumentFragment();
      t.nodeValue.split(/(\[S\d+\])/).forEach((part) => {
        const m = part.match(/^\[S(\d+)\]$/);
        if (!m) return frag.append(part);
        const k = Number(m[1]);
        const src = sources[k - 1];
        frag.append(el("span", {
          className: "cite" + (k < 1 || k > n ? " bad" : ""), textContent: "S" + k,
          title: src ? src.title + (src.heading ? " — " + src.heading : "") : "No such source",
        }));
      });
      t.replaceWith(frag);
    }
  }
  return box;
}

function copy(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    const old = btn.textContent; btn.textContent = "Copied";
    setTimeout(() => { btn.textContent = old; }, 1200);
  });
}

function sourcesBlock(sources) {
  const list = el("ol");
  sources.forEach((s) => {
    const name = s.title + (s.heading ? " — " + s.heading : "");
    const head = s.url ? el("a", { href: s.url, target: "_blank", rel: "noopener noreferrer", textContent: name })
                       : el("span", { textContent: name, title: s.path || "" });
    head.style.color = "var(--text)";
    list.append(el("li", {}, el("b", { textContent: `S${s.n} ` }), head,
      el("q", { textContent: s.preview + (s.preview.length >= 400 ? "…" : "") })));
  });
  return el("details", {}, el("summary", { textContent: `Sources (${sources.length})` }), list);
}

function verifyBadge(v) {
  if (v.abstained) return el("span", { className: "badge warn", textContent: "Not in sources: no answer given" });
  const problems = [];
  if (v.invalid && v.invalid.length) problems.push("cites missing source " + v.invalid.map((x) => "S" + x).join(", "));
  if (v.uncited) problems.push(`${v.uncited} of ${v.checked} paragraphs have no citation`);
  if (v.weak && v.weak.length) problems.push("weak match with " + v.weak.map((x) => "S" + x).join(", "));
  if (!problems.length) return el("span", { className: "badge ok", textContent: "Citations check out" });
  return el("span", { className: "badge " + (v.invalid && v.invalid.length ? "bad" : "warn"),
    textContent: "Check this answer: " + problems.join("; ") });
}

function statsLine(meta) {
  const s = meta.stats || {};
  const bits = [];
  if (meta.model) bits.push(meta.model);
  if (s.tokens_per_second) bits.push(s.tokens_per_second + " tok/s");
  if (s.first_token_seconds !== undefined) bits.push("first token " + s.first_token_seconds + "s");
  if (s.output_tokens) bits.push(s.output_tokens + " tokens");
  if (meta.stopped) bits.push("stopped");
  if (s.context_full) bits.push("⚠ context window full: older text was cut; raise num_ctx in Settings");
  return bits.join(" · ");
}

function messageNode(m, isLast) {
  const node = el("div", { className: "msg " + m.role });
  if (m.role === "user") {
    node.append(el("div", { className: "body", textContent: m.content }));
    return node;
  }
  const meta = m.meta || {};
  node.append(renderMarkdown(m.content, meta.sources && meta.sources.length ? meta.sources : null));
  if (m.streaming) node.querySelector(".body").classList.add("cursor");
  const extra = el("div", { className: "extra" });
  if (meta.thinking) extra.append(el("details", {}, el("summary", { textContent: "Reasoning" }),
    el("div", { className: "thinking", textContent: meta.thinking })));
  (meta.tools || []).forEach((t) => extra.append(el("details", {},
    el("summary", { textContent: `Tool: ${t.name} → ${String(t.result).slice(0, 80)}` }),
    el("code", { textContent: JSON.stringify(t.arguments) }))));
  if (meta.sources && meta.sources.length) extra.append(sourcesBlock(meta.sources));
  (meta.warnings || []).forEach((w) => extra.append(el("span", { className: "badge warn", textContent: w })));
  if (meta.error) extra.append(el("span", { className: "badge bad", textContent: meta.error }));
  if (!m.streaming) {
    const actions = el("div", { className: "actions" });
    if (meta.verify) actions.append(verifyBadge(meta.verify));
    const cp = el("button", { textContent: "Copy", type: "button" });
    cp.onclick = () => copy(m.content, cp);
    actions.append(cp);
    if (isLast && state.convId) {
      const rg = el("button", { textContent: "Regenerate", type: "button" });
      rg.onclick = () => send({ regenerate: true });
      actions.append(rg);
    }
    actions.append(el("span", { textContent: statsLine(meta) }));
    extra.append(actions);
  }
  node.append(extra);
  return node;
}

const STARTERS = [
  ["Review this code", "code-review", "Review this code:\n\n```java\n\n```"],
  ["Design a Spring Boot service", "spring", "Design a Spring Boot microservice that "],
  ["Interview drill", "interview", "Drill me on Java concurrency."],
  ["Ask my documents", "general", ""],
];

function renderMessages() {
  const box = $("messages");
  const pinned = box.scrollHeight - box.scrollTop - box.clientHeight < 120;
  box.replaceChildren();
  if (!state.messages.length) {
    const chips = el("div", { className: "chips" });
    STARTERS.forEach(([label, mode, text]) => {
      const b = el("button", { textContent: label, type: "button" });
      b.onclick = () => {
        $("mode").value = mode; $("input").value = text;
        if (label === "Ask my documents") $("use-docs").checked = true;
        $("input").focus();
      };
      chips.append(b);
    });
    box.append(el("div", { className: "empty" }, el("h2", { textContent: "adi-gpt" }),
      el("p", { textContent: "Private and local. Turn on Docs to get cited answers from your own files." }), chips));
    return;
  }
  state.messages.forEach((m, i) => box.append(messageNode(m, i === state.messages.length - 1)));
  if (pinned) box.scrollTop = box.scrollHeight;
}

let frame = null;
function scheduleRender() {
  if (frame) return;
  frame = requestAnimationFrame(() => { frame = null; renderMessages(); });
}

// ------------------------------------------------------------ conversations
async function loadConversations() {
  const list = await api("/api/conversations?q=" + encodeURIComponent($("search").value));
  const nav = $("conversations");
  nav.replaceChildren();
  list.forEach((c) => {
    const a = el("a", { className: c.id === state.convId ? "active" : "" },
      el("div", { textContent: c.title }));
    if (c.snippet) a.append(el("small", { textContent: c.snippet }));
    a.onclick = () => openConversation(c.id);
    nav.append(a);
  });
}

async function openConversation(id) {
  if (state.streaming) state.streaming.abort();
  const c = await api("/api/conversations/" + id);
  state.convId = id;
  state.messages = c.messages;
  $("title").textContent = c.title;
  $("mode").value = c.mode;
  $("incognito").checked = !!c.incognito;
  $("export").hidden = $("delete").hidden = false;
  document.body.classList.remove("menu-open");
  renderMessages();
  $("messages").scrollTop = $("messages").scrollHeight;
  loadConversations();
}

function newChat() {
  if (state.streaming) state.streaming.abort();
  state.convId = null; state.messages = [];
  $("title").textContent = "New chat";
  $("export").hidden = $("delete").hidden = true;
  $("incognito").checked = false;
  document.body.classList.remove("menu-open");
  renderMessages(); loadConversations(); $("input").focus();
}

// --------------------------------------------------------------------- chat
async function send({ regenerate = false } = {}) {
  if (state.streaming) { state.streaming.abort(); return; }
  const text = $("input").value.trim();
  if (!regenerate && !text) return;
  if (regenerate) {
    while (state.messages.length && state.messages[state.messages.length - 1].role !== "user") state.messages.pop();
  } else {
    state.messages.push({ role: "user", content: text });
    $("input").value = ""; autosize();
  }
  const reply = { role: "assistant", content: "", meta: {}, streaming: true };
  state.messages.push(reply);
  renderMessages();
  $("messages").scrollTop = $("messages").scrollHeight;

  const ctrl = new AbortController();
  state.streaming = ctrl;
  $("send").textContent = "Stop";
  const body = {
    message: text, conversation_id: state.convId, mode: $("mode").value, regenerate,
    use_docs: $("use-docs").checked, use_web: $("use-web").checked, strict: $("strict").checked,
    think: $("think").checked, incognito: $("incognito").checked,
  };
  try {
    const res = await fetch("/api/chat", { method: "POST", signal: ctrl.signal,
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error("Server error " + res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buffer.indexOf("\n\n")) >= 0) {
        const line = buffer.slice(0, cut); buffer = buffer.slice(cut + 2);
        if (line.startsWith("data: ")) handleEvent(JSON.parse(line.slice(6)), reply);
      }
    }
  } catch (err) {
    if (err.name === "AbortError") reply.meta.stopped = true;
    else reply.meta.error = err.message;
  }
  reply.streaming = false;
  state.streaming = null;
  $("send").textContent = "Send";
  renderMessages();
  loadConversations();
}

function handleEvent(ev, reply) {
  const meta = reply.meta;
  if (ev.type === "start") {
    state.convId = ev.conversation_id; $("title").textContent = ev.title;
    $("export").hidden = $("delete").hidden = false;
  } else if (ev.type === "token") reply.content += ev.text;
  else if (ev.type === "thinking") meta.thinking = (meta.thinking || "") + ev.text;
  else if (ev.type === "sources") meta.sources = ev.sources;
  else if (ev.type === "tool") (meta.tools = meta.tools || []).push(ev);
  else if (ev.type === "verify") meta.verify = ev;
  else if (ev.type === "warning") (meta.warnings = meta.warnings || []).push(ev.message);
  else if (ev.type === "error") meta.error = ev.message;
  else if (ev.type === "done") { meta.stats = ev.stats; meta.model = ev.model; }
  scheduleRender();
}

function autosize() {
  const t = $("input"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 220) + "px";
}

// ---------------------------------------------------------------- documents
async function loadDocs() {
  const docs = await api("/api/documents");
  const list = $("docs-list");
  list.replaceChildren();
  if (!docs.length) list.append(el("li", {}, el("span", { className: "hint", textContent: "Nothing indexed yet." })));
  docs.forEach((d) => {
    const del = el("button", { textContent: "Remove", type: "button" });
    del.onclick = async () => { await api("/api/documents/" + d.id, { method: "DELETE" }); loadDocs(); };
    list.append(el("li", {}, el("span", {}, d.title,
      el("small", { textContent: `${d.n_chunks} chunks · ${d.path}` })), del));
  });
}

async function withMessage(node, fn) {
  node.textContent = "Working…";
  try { node.textContent = await fn(); } catch (err) { node.textContent = "Failed: " + err.message; }
}

// ------------------------------------------------------------------- memory
async function loadMemory() {
  const items = await api("/api/memories");
  const list = $("memory-list");
  list.replaceChildren();
  if (!items.length) list.append(el("li", {}, el("span", { className: "hint", textContent: "No memories yet." })));
  items.forEach((m) => {
    const input = el("input", { value: m.text });
    input.style.flex = "1";
    input.onchange = async () => {
      try { await api("/api/memories/" + m.id, { method: "PATCH", json: { text: input.value } }); }
      catch (err) { alert(err.message); loadMemory(); }
    };
    const del = el("button", { textContent: "Delete", type: "button" });
    del.onclick = async () => { await api("/api/memories/" + m.id, { method: "DELETE" }); loadMemory(); };
    list.append(el("li", {}, input, el("small", { textContent: m.source }), del));
  });
}

// ----------------------------------------------------------------- settings
async function loadStatus() {
  const s = state.status = await api("/api/status");
  const dot = $("status-dot");
  const banner = $("banner");
  banner.hidden = true;
  if (!s.ok) {
    dot.className = "dot bad"; dot.title = s.error;
    banner.hidden = false;
    banner.replaceChildren("Ollama is not reachable at ", el("code", { textContent: s.ollama_host }),
      ". Start it with ", el("code", { textContent: "ollama serve" }), " and reload.");
  } else if (s.missing.length) {
    dot.className = "dot warn"; dot.title = "Model missing";
    banner.hidden = false;
    banner.replaceChildren("Model not installed. Run: ",
      ...s.missing.map((m) => el("code", { textContent: "ollama pull " + m })).flatMap((n) => [n, " "]));
  } else {
    dot.className = "dot ok"; dot.title = `Ollama ${s.ollama_version} · ${s.chat_model}`;
  }
  return s;
}

async function loadSettings() {
  const [s, cfg] = await Promise.all([loadStatus(), api("/api/settings")]);
  $("status-box").textContent = s.ok
    ? `Ollama ${s.ollama_version} at ${s.ollama_host} · ${s.documents} documents, ${s.chunks} chunks · model can: ${(s.capabilities || []).join(", ") || "unknown"}`
    : s.error;
  const sel = $("set-chat-model");
  sel.replaceChildren();
  const names = new Set([cfg.chat_model, ...(s.models || [])]);
  names.forEach((n) => sel.append(el("option", { value: n, textContent: n })));
  sel.value = cfg.chat_model;
  $("set-embed-model").value = cfg.embed_model;
  $("set-num-ctx").value = cfg.num_ctx;
  $("set-temperature").value = cfg.temperature === null ? "" : cfg.temperature;
  $("set-custom").value = cfg.custom_instructions;
  $("set-tools").checked = cfg.tools_enabled;
  $("set-auto-memory").checked = cfg.auto_memory;
}

async function saveSettings() {
  const temp = $("set-temperature").value;
  await api("/api/settings", { method: "PUT", json: {
    chat_model: $("set-chat-model").value, embed_model: $("set-embed-model").value.trim(),
    num_ctx: Number($("set-num-ctx").value), custom_instructions: $("set-custom").value,
    tools_enabled: $("set-tools").checked, auto_memory: $("set-auto-memory").checked,
    ...(temp === "" ? { reset_temperature: true } : { temperature: Number(temp) }),
  } });
  $("settings-dialog").close();
  loadStatus();
}

// ------------------------------------------------------------------- wiring
function wire() {
  $("composer").onsubmit = (e) => { e.preventDefault(); send(); };
  $("input").oninput = autosize;
  $("input").onkeydown = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); }
  };
  $("new-chat").onclick = newChat;
  $("menu").onclick = () => document.body.classList.toggle("menu-open");
  $("messages").onclick = () => document.body.classList.remove("menu-open");
  let timer;
  $("search").oninput = () => { clearTimeout(timer); timer = setTimeout(loadConversations, 200); };
  $("export").onclick = () => { window.location = `/api/conversations/${state.convId}/export`; };
  $("delete").onclick = async () => {
    if (!confirm("Delete this chat?")) return;
    await api("/api/conversations/" + state.convId, { method: "DELETE" });
    newChat();
  };
  const loaders = { "docs-dialog": loadDocs, "memory-dialog": loadMemory, "settings-dialog": loadSettings };
  document.querySelectorAll("[data-open]").forEach((b) => {
    b.onclick = () => { $(b.dataset.open).showModal(); loaders[b.dataset.open](); };
  });
  document.querySelectorAll("[data-close]").forEach((b) => { b.onclick = () => b.closest("dialog").close(); });

  $("index-folder").onclick = () => withMessage($("docs-msg"), async () => {
    const r = await api("/api/documents/index-folder", { method: "POST", json: { path: $("folder").value.trim() } });
    loadDocs();
    return `Indexed ${r.indexed}, unchanged ${r.unchanged}, skipped ${r.skipped}, removed ${r.removed}.`;
  });
  $("upload").onclick = () => withMessage($("docs-msg"), async () => {
    const form = new FormData();
    [...$("files").files].forEach((f) => form.append("files", f));
    if (![...$("files").files].length) return "Choose one or more files first.";
    const r = await api("/api/documents/upload", { method: "POST", body: form });
    loadDocs();
    return r.map((x) => `${x.file}: ${x.status}`).join(" · ");
  });
  $("probe-btn").onclick = async () => {
    const out = $("probe-out");
    out.textContent = "Searching…";
    try {
      const hits = await api("/api/documents/search", { method: "POST", json: { query: $("probe").value } });
      out.replaceChildren();
      if (!hits.length) out.textContent = "No matches.";
      hits.forEach((h) => out.append(el("div", {},
        el("b", { textContent: `${h.title}${h.heading ? " — " + h.heading : ""} ` }),
        el("span", { className: "badge " + (h.relevant ? "ok" : "warn"),
          textContent: `${h.relevant ? "used" : "below threshold"} · sim ${h.similarity ?? "–"} · keywords ${h.keyword_overlap}` }),
        el("div", { className: "hint", textContent: h.text.slice(0, 220) + "…" }))));
    } catch (err) { out.textContent = "Failed: " + err.message; }
  };
  $("memory-add").onclick = async () => {
    try { await api("/api/memories", { method: "POST", json: { text: $("memory-new").value } }); $("memory-new").value = ""; }
    catch (err) { alert(err.message); }
    loadMemory();
  };
  $("memory-clear").onclick = async () => {
    if (confirm("Delete every memory?")) { await api("/api/memories", { method: "DELETE" }); loadMemory(); }
  };
  $("settings-save").onclick = () => saveSettings().catch((err) => alert(err.message));
}

async function init() {
  wire();
  state.modes = await api("/api/modes");
  state.modes.forEach((m) => $("mode").append(el("option", { value: m.id, textContent: m.label })));
  renderMessages();
  loadConversations();
  loadStatus().catch(() => {});
}
init();
