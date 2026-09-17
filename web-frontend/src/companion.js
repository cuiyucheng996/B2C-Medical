let sessionId = "";
let resumeToken = "";
let closing = false;
const $ = (id) => document.getElementById(id);

function formatReportSummary(report) {
  if (!report || !report.filename) return "";
  const inds = report.indicators || [];
  const lines = [
    "【报告摘要】" + report.filename + " · " + (inds.length || 0) + " 项指标",
  ];
  inds.slice(0, 8).forEach((item) => {
    const name = item.name || "";
    const value = [item.value, item.unit, item.flag].filter(Boolean).join(" ");
    if (name) lines.push("- " + name + " " + value);
  });
  return lines.join("\n");
}

function pretty(value) {
  if (value == null || value === "") return "-";
  return JSON.stringify(value, null, 2);
}

function addMsg(role, text, actions) {
  const el = document.createElement("div");
  el.className = "msg " + role;
  const body = document.createElement("div");
  body.className = "msg-text";
  body.textContent = text;
  el.appendChild(body);
  $("messages").appendChild(el);
  attachActions(el, actions);
  $("messages").scrollTop = $("messages").scrollHeight;
  return el;
}

function attachActions(el, actions) {
  if (!el) return;
  const old = el.querySelector(".msg-actions");
  if (old) old.remove();
  if (!actions || !actions.length) return;
  const bar = document.createElement("div");
  bar.className = "msg-actions";
  for (const item of actions) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = item.label || item.id;
    btn.onclick = () => runAction(item.id);
    bar.appendChild(btn);
  }
  el.appendChild(bar);
}

function runAction(id) {
  if (id === "upload_report") {
    if (!sessionId) {
      alert("请先新建会话");
      return;
    }
    $("reportFile").click();
    return;
  }
  if (id === "book_expert") {
    send("帮我预约专家问诊");
  }
}

function showSessionMeta(data) {
  sessionId = data.session_id || sessionId;
  $("chipSession").textContent = sessionId || "未创建";
  $("chipUser").textContent = data.group_id || data.user_id || "-";
  $("dumpSession").textContent = pretty(data);
}

function thinkOn() {
  return Boolean($("chkThink") && $("chkThink").checked);
}

function persistThinkToggle() {
  try {
    localStorage.setItem("b2c-think-mode", thinkOn() ? "1" : "0");
  } catch {
    /* ignore */
  }
}

function persistUrl() {
  const id = userKey();
  if (!id) return "";
  return "/api/sessions/" + encodeURIComponent(id) + "/close";
}

async function persistSession({ silent } = { silent: false }) {
  const url = persistUrl();
  if (!url || closing) return null;
  closing = true;
  try {
    const res = await fetch(url, { method: "POST", keepalive: true });
    const data = await res.json().catch(() => ({}));
    if (data.ingest && data.ingest.skipped) {
      if (!silent) $("debugHint").textContent = "会话已不在服务端（可能刚重启）";
      return data;
    }
    if (!silent) {
      $("dumpIngest").textContent = pretty(data.ingest || data);
      $("debugHint").textContent = "会话结束入库";
      if (data.session) showSessionMeta(data.session);
      addMsg("sys", "本场已抽离入库。");
    }
    return data;
  } catch (err) {
    if (!silent) alert("入库失败");
    return null;
  } finally {
    closing = false;
  }
}

function persistOnLeave() {
  const url = persistUrl();
  if (!url) return;
  const blob = new Blob([], { type: "application/json" });
  if (navigator.sendBeacon) navigator.sendBeacon(url, blob);
  else fetch(url, { method: "POST", keepalive: true });
}

function showRound(data) {
  const round = data.round || {
    session_id: data.session && data.session.session_id,
    group_id: data.session && data.session.group_id,
    reply: data.reply,
    memory_facts: data.facts,
    ingest: data.ingest,
    session: data.session,
  };
  $("debugHint").textContent = "最近一次 /chat";
  $("dumpRound").textContent = pretty(round);
  $("dumpPrep").textContent = pretty(data.preprocess || round.preprocess);
  $("dumpFacts").textContent = pretty(data.facts);
  $("dumpIngest").textContent = pretty(data.ingest);
  if (data.session) showSessionMeta(data.session);
  const hitl = data.hitl || round.hitl;
  if (hitl && hitl.resume_token) resumeToken = hitl.resume_token;
  else if (!hitl || !hitl.pending) resumeToken = "";
}

$("btnEnd").onclick = async () => {
  if (!sessionId) {
    alert("没有进行中的会话");
    return;
  }
  await persistSession({ silent: false });
};

$("btnNew").onclick = async () => {
  const user_id = $("userId").value.trim();
  if (!user_id) {
    alert("请先填写 user_id（Neo4j UserInfo.id）。");
    $("userId").focus();
    return;
  }
  await persistSession({ silent: true });
  const res = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id }),
  });
  const data = await res.json();
  if (!res.ok) {
    alert(data.detail || "新建失败");
    return;
  }
  const canon = String(data.group_id || data.user_id || user_id || "").trim();
  if (canon) {
    $("userId").value = canon;
    try { localStorage.setItem("b2c-user-id", canon); } catch (e) {}
  }
  $("messages").innerHTML = "";
  addMsg("sys", "会话已建立。下面是进场根据长期记忆的推送。");
  const push = data.push || {};
  addMsg("ai", push.text || "（无推送）");
  showSessionMeta(data);
  $("dumpRound").textContent = pretty({
    event: "create_session",
    session_id: data.session_id,
    group_id: data.group_id,
    push: push,
  });
  $("dumpFacts").textContent = pretty(push.facts || []);
  $("dumpIngest").textContent = pretty({ deferred: true, reason: "persist_on_session_end" });
  $("debugHint").textContent = "进场推送（长期记忆）";
  resumeToken = "";
};

window.addEventListener("pagehide", persistOnLeave);

async function send(preset, opts) {
  const speak = Boolean(opts && opts.speak);
  const channel = (opts && opts.channel) || "text";
  const presetText = typeof preset === "string" ? preset : "";
  const text = String(presetText || $("text").value || "").trim();
  if (!sessionId) {
    alert("请先新建会话");
    return "";
  }
  if (!text) return "";
  if (!presetText) $("text").value = "";
  addMsg("user", text);
  $("btnSend").disabled = true;
  $("debugHint").textContent = "SSE 流式回复中";

  const res = await fetch("/api/sessions/" + encodeURIComponent(userKey()) + "/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({
      text,
      resume_token: resumeToken || "",
      input_channel: channel,
      think: thinkOn(),
    }),
  });
  if (!res.ok) {
    $("btnSend").disabled = false;
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    addMsg("sys", "错误: " + (err.detail || "请求失败"));
    return "";
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let assistantEl = null;
  let spoken = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() || "";
      for (const block of parts) {
        const line = block.trim();
        if (!line.startsWith("data: ")) continue;
        let data;
        try {
          data = JSON.parse(line.slice(6));
        } catch {
          continue;
        }
        if (data.event === "thinking") {
          $("debugHint").textContent = data.message || "正在回复…";
        } else if (data.event === "suspended") {
          resumeToken = data.resume_token || resumeToken;
          $("debugHint").textContent = "等待补充：" + (data.question || "");
          attachActions(assistantEl, data.actions);
        } else if (data.event === "token") {
          if (!assistantEl) {
            assistantEl = addMsg("ai", "");
          }
          const body = assistantEl.querySelector(".msg-text") || assistantEl;
          body.textContent += data.content || "";
          spoken = body.textContent;
          $("messages").scrollTop = $("messages").scrollHeight;
        } else if (data.event === "state") {
          showRound(data.data || {});
          attachActions(assistantEl, (data.data || {}).actions);
          $("debugHint").textContent = "SSE 结束";
        } else if (data.event === "error") {
          addMsg("sys", "错误: " + (data.message || "流式失败"));
        }
      }
    }
  } finally {
    $("btnSend").disabled = false;
  }
  if (speak && spoken && $("chkSpeak") && $("chkSpeak").checked) {
    await playTts(spoken);
  }
  return spoken;
}

$("btnSend").onclick = () => send();
$("text").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    send();
  }
});

let mediaRecorder = null;
let audioChunks = [];

async function playTts(text) {
  try {
    const res = await fetch("/api/voice/tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text.slice(0, 800) }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      addMsg("sys", "播报失败: " + (err.detail || res.statusText));
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => URL.revokeObjectURL(url);
    await audio.play();
  } catch {
    addMsg("sys", "播报失败");
  }
}

async function startTalk() {
  if (!sessionId) {
    alert("请先新建会话");
    return;
  }
  if (!navigator.mediaDevices || !window.MediaRecorder) {
    addMsg("sys", "当前浏览器不支持录音");
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioChunks = [];
  mediaRecorder = new MediaRecorder(stream);
  mediaRecorder.ondataavailable = (ev) => {
    if (ev.data && ev.data.size) audioChunks.push(ev.data);
  };
  mediaRecorder.start();
  $("btnMic").classList.add("mic-on");
  $("btnMic").textContent = "松开识别";
}

async function stopTalk() {
  const rec = mediaRecorder;
  mediaRecorder = null;
  $("btnMic").classList.remove("mic-on");
  $("btnMic").textContent = "按住说话";
  if (!rec || rec.state === "inactive") return;
  const blob = await new Promise((resolve) => {
    rec.onstop = () => resolve(new Blob(audioChunks, { type: rec.mimeType || "audio/webm" }));
    rec.stop();
    rec.stream.getTracks().forEach((track) => track.stop());
  });
  if (!blob.size) return;
  $("debugHint").textContent = "正在识别…";
  const body = new FormData();
  body.append("file", blob, "speech.webm");
  const res = await fetch("/api/voice/asr", { method: "POST", body });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    addMsg("sys", "识别失败: " + (data.detail || res.statusText));
    return;
  }
  const text = String(data.text || "").trim();
  if (!text) {
    addMsg("sys", "没有听清，请再说一次");
    return;
  }
  await send(text, { speak: true, channel: "asr" });
}

$("btnMic").addEventListener("pointerdown", (ev) => {
  ev.preventDefault();
  startTalk().catch((err) => addMsg("sys", "无法开麦: " + (err && err.message ? err.message : "失败")));
});
$("btnMic").addEventListener("pointerup", () => {
  stopTalk().catch(() => addMsg("sys", "识别失败"));
});
$("btnMic").addEventListener("pointerleave", () => {
  if (mediaRecorder) stopTalk().catch(() => {});
});

$("btnReport").onclick = () => {
  if (!sessionId) {
    alert("请先新建会话");
    return;
  }
  $("reportFile").click();
};
$("reportFile").addEventListener("change", async () => {
  const file = $("reportFile").files && $("reportFile").files[0];
  $("reportFile").value = "";
  if (!file) return;
  await uploadReport(file);
});
$("text").addEventListener("paste", async (ev) => {
  const items = ev.clipboardData && ev.clipboardData.items;
  if (!items) return;
  const imageItem = [...items].find((item) => item.type && item.type.startsWith("image/"));
  if (!imageItem) return;
  const file = imageItem.getAsFile();
  if (!file) return;
  ev.preventDefault();
  await uploadReport(file);
});

async function uploadReport(file) {
  if (!sessionId) {
    alert("请先新建会话");
    return;
  }
  addMsg("user", "【上传检查报告】" + file.name);
  addMsg("sys", "正在解析检查报告（PDF 或图片）…");
  $("btnReport").disabled = true;
  $("btnSend").disabled = true;
  $("debugHint").textContent = "解析报告中";
  try {
    const body = new FormData();
    body.append("file", file, file.name || "report.png");
    body.append("think", thinkOn() ? "true" : "false");
    const res = await fetch("/api/sessions/" + encodeURIComponent(userKey()) + "/report", {
      method: "POST",
      body,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      addMsg("sys", "解析失败: " + (data.detail || res.statusText));
      return;
    }
    addMsg("ai", [data.reply || "（无解读）", formatReportSummary(data.report)].filter(Boolean).join("\n\n"), data.actions);
    showRound(data);
    $("dumpPrep").textContent = pretty({ report: data.report, preprocess: data.preprocess });
    $("debugHint").textContent = "报告摘要 " + ((data.report && data.report.parse_source) || (data.file && data.file.role) || "");
  } catch (err) {
    addMsg("sys", "上传失败");
  } finally {
    $("btnReport").disabled = false;
    $("btnSend").disabled = false;
  }
}

(function bootThinkToggle() {
  const box = $("chkThink");
  if (!box) return;
  try {
    box.checked = localStorage.getItem("b2c-think-mode") === "1";
  } catch {
    box.checked = false;
  }
  box.addEventListener("change", persistThinkToggle);
})();

(function bootFromQuery() {
  const params = new URLSearchParams(location.search);
  let uid = (params.get("user_id") || "").trim();
  if (!uid) {
    try { uid = (localStorage.getItem("b2c-user-id") || "").trim(); } catch (e) { uid = ""; }
  }
  if (!uid) return;
  $("userId").value = uid;
  $("btnNew").click();
})();
