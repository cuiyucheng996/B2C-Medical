/**
 * C 端按 user_id 持久化 UI（localStorage + 服务端 draft）。
 */
(function (global) {
  const USER_KEY = "b2c-user-id";
  const DRAFT_PREFIX = "b2c-ui-draft:";

  function getUserId() {
    try {
      return (localStorage.getItem(USER_KEY) || "").trim();
    } catch (_e) {
      return "";
    }
  }

  function setUserId(uid) {
    const v = String(uid || "").trim();
    if (!v) return;
    try {
      localStorage.setItem(USER_KEY, v);
    } catch (_e) {}
  }

  function loadLocalDraft(userId) {
    const uid = String(userId || getUserId() || "").trim();
    if (!uid) return {};
    try {
      const raw = localStorage.getItem(DRAFT_PREFIX + uid);
      if (!raw) return {};
      const data = JSON.parse(raw);
      return data && typeof data === "object" ? data : {};
    } catch (_e) {
      return {};
    }
  }

  function saveLocalDraft(userId, payload) {
    const uid = String(userId || getUserId() || "").trim();
    if (!uid || !payload) return;
    try {
      localStorage.setItem(DRAFT_PREFIX + uid, JSON.stringify(payload));
    } catch (_e) {}
  }

  function mergeLocalDraft(userId, patch) {
    const uid = String(userId || getUserId() || "").trim();
    if (!uid) return {};
    const base = loadLocalDraft(uid);
    for (const key of Object.keys(patch || {})) {
      const val = patch[key];
      if (val && typeof val === "object" && !Array.isArray(val) && base[key] && typeof base[key] === "object") {
        base[key] = Object.assign({}, base[key], val);
      } else {
        base[key] = val;
      }
    }
    base.updated_at = new Date().toISOString();
    saveLocalDraft(uid, base);
    return base;
  }

  let _syncTimer = null;
  function scheduleServerSync(userId) {
    const uid = String(userId || getUserId() || "").trim();
    if (!uid) return;
    if (_syncTimer) clearTimeout(_syncTimer);
    _syncTimer = setTimeout(function () {
      _syncTimer = null;
      const body = loadLocalDraft(uid);
      fetch("/api/users/" + encodeURIComponent(uid) + "/ui-draft", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }).catch(function () {});
    }, 600);
  }

  function mergeDraft(userId, patch) {
    const merged = mergeLocalDraft(userId, patch);
    scheduleServerSync(userId);
    return merged;
  }

  async function fetchServerDraft(userId) {
    const uid = String(userId || getUserId() || "").trim();
    if (!uid) return {};
    try {
      const res = await fetch("/api/users/" + encodeURIComponent(uid) + "/ui-draft");
      const data = await res.json().catch(function () {
        return {};
      });
      if (!res.ok || !data || !data.draft) return {};
      mergeLocalDraft(uid, data.draft);
      return data.draft;
    } catch (_e) {
      return {};
    }
  }

  function mountUserBar(containerSelector) {
    const el = document.querySelector(containerSelector || ".b2c-user-bar");
    if (!el) return;
    const uid = getUserId();
    el.innerHTML =
      '当前用户 <strong>' +
      (uid ? escapeHtml(uid) : "未登录") +
      "</strong> · " +
      '<a href="/">切换账号</a>';
  }

  function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function saveReleasedChatDraft(userId, sessionPayload) {
    const uid = String(userId || getUserId() || "").trim();
    if (!uid || !sessionPayload) return;
    const messages = [];
    const rows = sessionPayload.user || sessionPayload.messages || [];
    if (Array.isArray(rows)) {
      rows.forEach(function (item) {
        if (!item) return;
        const role_type = item.role_type || item.role || "user";
        const content = String(item.content || "").trim();
        if (!content) return;
        messages.push({
          role_type: role_type,
          role: item.role || role_type,
          content: content,
        });
      });
    }
    mergeLocalDraft(uid, {
      chat: {
        session_id: sessionPayload.session_id || "",
        messages: messages,
        released_at: new Date().toISOString(),
      },
    });
    scheduleServerSync(uid);
  }

  async function releaseRemoteSession(userId) {
    const uid = String(userId || getUserId() || "").trim();
    if (!uid) return null;
    try {
      const res = await fetch("/api/sessions/" + encodeURIComponent(uid) + "/close", {
        method: "POST",
        keepalive: true,
      });
      const data = await res.json().catch(function () {
        return {};
      });
      if (data.session) saveReleasedChatDraft(uid, data.session);
      return data;
    } catch (_e) {
      return null;
    }
  }

  global.B2cPersist = {
    USER_KEY: USER_KEY,
    getUserId: getUserId,
    setUserId: setUserId,
    loadLocalDraft: loadLocalDraft,
    saveLocalDraft: saveLocalDraft,
    mergeDraft: mergeDraft,
    saveReleasedChatDraft: saveReleasedChatDraft,
    releaseRemoteSession: releaseRemoteSession,
    fetchServerDraft: fetchServerDraft,
    mountUserBar: mountUserBar,
  };
})(window);
