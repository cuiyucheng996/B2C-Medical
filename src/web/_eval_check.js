
    const $ = (id) => document.getElementById(id);
    const SAMPLES = [
      "头部常见有哪些病",
      "头痛是什么疾病",
      "胃炎有哪些症状",
      "血常规这个检查是查什么的",
      "转氨酶正常值是多少",
      "空腹血糖参考范围是多少",
    ];
    const pick = $("kbPick");
    SAMPLES.forEach((t, i) => {
      const o = document.createElement("option");
      o.value = t;
      o.textContent = t;
      if (i === 0) o.selected = true;
      pick.appendChild(o);
    });
    $("kbText").value = pick.value;
    pick.onchange = () => { $("kbText").value = pick.value; };

    document.querySelectorAll(".tabs button").forEach((btn) => {
      btn.onclick = () => {
        document.querySelectorAll(".tabs button").forEach((b) => b.classList.remove("on"));
        btn.classList.add("on");
        $("tab-kb").classList.toggle("hidden", btn.dataset.tab !== "kb");
        $("tab-ragflow").classList.toggle("hidden", btn.dataset.tab !== "ragflow");
        $("tab-file").classList.toggle("hidden", btn.dataset.tab !== "file");
        $("tab-coref").classList.toggle("hidden", btn.dataset.tab !== "coref");
        $("tab-other").classList.toggle("hidden", btn.dataset.tab !== "other");
      };
    });

    function esc(s) {
      return String(s ?? "").replace(/[&<>"']/g, (c) =>
        ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
      );
    }
    function renderLayoutEval(ids, data) {
      const phasesEl = $(ids.phases);
      const pagesEl = $(ids.pages);
      const blocksEl = $(ids.blocks);
      if (phasesEl) {
        phasesEl.innerHTML = (data.phases || []).length
          ? (data.phases || []).map((ph) => {
              const branches = (ph.branches || []).map((b) =>
                "<tr><td>" + esc(b.cond) + "</td><td>" + esc(b.handler) + "</td><td>" + esc(b.note) + "</td></tr>"
              ).join("");
              const facts = (ph.facts || []).map((f) =>
                "<div style='display:inline-block;margin:0 12px 4px 0'><b>" + esc(f[0]) + "</b> " + esc(f[1]) + "</div>"
              ).join("");
              return "<div style='margin:0 0 16px'>" +
                "<div style='font-weight:600;margin-bottom:2px'>" + esc(ph.title) + "</div>" +
                (ph.aim ? "<div class='hint' style='margin:0 0 8px'>" + esc(ph.aim) + "</div>" : "") +
                "<table><thead><tr><th>条件</th><th>怎么处理</th><th>说明</th></tr></thead><tbody>" +
                branches + "</tbody></table>" +
                "<div class='meta' style='margin-top:8px'>" + facts + "</div></div>";
            }).join("")
          : "<div class='hint'>本次没有分块前信息</div>";
      }
      if (pagesEl) {
        pagesEl.innerHTML = (data.page_rows || []).length
          ? (data.page_rows || []).map((p) =>
              "<tr><td>" + esc(p.page) + "</td><td>" + esc(p.chars) + "</td><td>" + esc(p.native) + "</td><td>" +
              esc(p.ocr) + "</td><td>" + esc(p.backend) + "</td><td>" + esc(p.regions) + "</td><td>" + esc(p.lines) + "</td></tr>"
            ).join("")
          : "<tr><td colspan='7'>无逐页信息</td></tr>";
      }
      if (blocksEl) {
        blocksEl.innerHTML = (data.block_samples || []).length
          ? (data.block_samples || []).map((p) =>
              "<tr><td>" + esc(p.type) + "</td><td>" + esc(p.page) + "</td><td>" + esc(p.text) + "</td></tr>"
            ).join("")
          : "<tr><td colspan='3'>无 content_blocks 样例</td></tr>";
      }
    }
    function fillChunkCfg(row) {
      if (!row) return;
      $("cfgEmbProv").value = row.embedding_provider || "";
      $("cfgEmbModel").value = row.embedding_model || "";
      $("cfgRfEmb").value = row.ragflow_embedding_model || "";
      $("cfgToken").value = row.chunk_token_num ?? "";
      $("cfgLocalMax").value = row.local_max_chars ?? "";
      $("cfgNaive").value = row.local_naive_chars ?? "";
      $("cfgOverlap").value = row.local_overlap ?? "";
      $("cfgMethod").value = row.chunk_method || "";
      const delim = row.delimiter === "\n" ? "\\n" : (row.delimiter || "");
      $("cfgDelim").value = delim;
      $("cfgLayout").value = row.layout_recognize || "";
    }

    function collectChunkCfg() {
      return {
        embedding_provider: $("cfgEmbProv").value.trim(),
        embedding_model: $("cfgEmbModel").value.trim(),
        ragflow_embedding_model: $("cfgRfEmb").value.trim(),
        chunk_token_num: Number($("cfgToken").value),
        local_max_chars: Number($("cfgLocalMax").value),
        local_naive_chars: Number($("cfgNaive").value),
        local_overlap: Number($("cfgOverlap").value),
        chunk_method: $("cfgMethod").value.trim(),
        delimiter: $("cfgDelim").value,
        layout_recognize: $("cfgLayout").value.trim(),
      };
    }

    async function loadChunkCfg() {
      try {
        const res = await fetch("/api/eval/chunk-config");
        const data = await res.json().catch(() => ({}));
        if (res.ok) fillChunkCfg(data);
      } catch (_err) {}
    }
    loadChunkCfg();

    $("btnRfCfg").onclick = async () => {
      $("rfCfgHint").textContent = "保存中…";
      try {
        const res = await fetch("/api/eval/chunk-config", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(collectChunkCfg()),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          $("rfCfgHint").textContent = data.detail || ("HTTP " + res.status);
          return;
        }
        fillChunkCfg(data);
        $("rfCfgHint").textContent = "已保存（当前进程）。下次入库会用这些值。";
      } catch (err) {
        $("rfCfgHint").textContent = String(err.message || err);
      }
    };

    function pill(ok) {
      return '<span class="pill ' + (ok ? "ok" : "bad") + '">' + (ok ? "通过" : "失败") + "</span>";
    }

    $("btnKb").onclick = async () => {
      const query = $("kbText").value.trim();
      if (!query) return;
      $("kbIntent").textContent = "请求中…";
      $("btnKb").disabled = true;
      try {
        const res = await fetch("/api/eval/kb", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          $("kbIntent").textContent = data.detail || ("HTTP " + res.status);
          return;
        }
      const kb = data.kb || {};
      $("kbHint").textContent =
        (kb.reachable ? "RAGFlow 通 · " : "") +
        (data.rerank_rule || "") +
        (kb.error ? " · " + kb.error : "");
      $("kbIntent").textContent =
        (data.major || "-") + " / " + (data.intent || "-") + " / " + (data.handler || "-") +
        (data.intent_source ? "（" + data.intent_source + "）" : "");
      $("kbTools").textContent = (data.tools || []).join("、") || "（空）";
      $("kbChan").textContent = data.channel || "-";
      const mentions = data.mentions || {};
      const extractBits = Object.keys(mentions).map((k) => {
        const vals = (mentions[k] || []).filter(Boolean);
        return vals.length ? k + "：" + vals.join("、") : "";
      }).filter(Boolean);
      const seedTxt = (data.seeds || []).filter(Boolean).join("、");
      $("kbExtract").textContent =
        (extractBits.join("；") || "未抽出") + (seedTxt ? " · 种子：" + seedTxt : "");
      const aligns = data.alignments || [];
      $("kbAlign").textContent = aligns.length
        ? aligns.map((a) =>
            (a.mention || "?") + " → " + (a.standard_name || a.mention || "?") +
            (a.node_label ? "（" + a.node_label + "）" : "")
          ).join("；")
        : "未对齐";
      const milvus = data.milvus || {};
      $("kbMilvus").innerHTML =
        '<span class="pill ' + (milvus.queried ? "ok" : "bad") + '">' +
        (milvus.queried ? "已查询" : "未查询") + "</span> " +
        esc(milvus.detail || "");
      const rows = data.rerank || [];
      $("kbBody").innerHTML = rows.length ? rows.map((r) => {
        const mv = Number(r.moved || 0);
        const mvHtml = mv > 0 ? '<span class="up">↑' + mv + "</span>"
          : mv < 0 ? '<span class="down">↓' + Math.abs(mv) + "</span>" : "—";
        return "<tr><td>" + esc(r.recall_rank) + "</td><td>" + esc(r.rerank_rank) +
          "</td><td>" + mvHtml + "</td><td>" + esc(r.score) + "</td><td>" +
          esc(r.keyword_hits) + "</td><td>" + esc(r.text) + "</td></tr>";
      }).join("") : "<tr><td colspan='6'>没有召回块"
        + (data.summary ? " · " + esc(data.summary) : "")
        + (data.seeds && data.seeds.length ? " · 种子 " + esc((data.seeds || []).join("、")) : "")
        + "</td></tr>";
      } catch (err) {
        $("kbIntent").textContent = String(err.message || err);
      } finally {
        $("btnKb").disabled = false;
      }
    };

    function slotParse(raw) {
      const out = {};
      const t = String(raw || "").trim();
      if (!t) return out;
      const m = t.match(/^([\u4e00-\u9fff]+)\s*[=:：]\s*(.+)$/);
      if (!m) return out;
      out[m[1]] = [m[2].trim()];
      return out;
    }

    $("btnCorefGold").onclick = async () => {
      $("corefSum").textContent = "运行中…";
      $("btnCorefGold").disabled = true;
      try {
        const res = await fetch("/api/eval/coref", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ gold: true }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          $("corefSum").textContent = data.detail || ("HTTP " + res.status);
          return;
        }
        $("corefSum").textContent = "通过 " + data.passed + " / " + data.total;
        $("corefBody").innerHTML = (data.rows || []).map((r) => {
          const slots = r.slots && Object.keys(r.slots).length
            ? Object.entries(r.slots).map(([k, v]) => k + "=" + (v || []).join("/")).join("；")
            : "（无槽）";
          return "<tr><td>" + pill(r.ok) + "</td><td>" + esc(r.title) +
            "</td><td>" + esc(r.prior) + "<br>" + esc(slots) +
            "</td><td>" + esc(r.query) +
            "</td><td>" + (r.got_changed ? "已改写" : "未改写") + " · " +
            esc(r.got_reason) + "<br>" + esc(r.got_text) +
            (r.fail && r.fail.length ? "<br>" + esc(r.fail.join("；")) : "") +
            "</td><td>" + esc(r.got_source || "-") + "</td></tr>";
        }).join("");
      } catch (err) {
        $("corefSum").textContent = String(err.message || err);
      } finally {
        $("btnCorefGold").disabled = false;
      }
    };

    $("btnCorefOne").onclick = async () => {
      const query = $("corefQuery").value.trim();
      if (!query) return;
      $("corefOne").textContent = "请求中…";
      try {
        const res = await fetch("/api/eval/coref", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            gold: false,
            query,
            prior: $("corefPrior").value.trim(),
            slot_memory: slotParse($("corefSlot").value),
          }),
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          $("corefOne").textContent = data.detail || ("HTTP " + res.status);
          return;
        }
        $("corefOne").innerHTML =
          (data.got_changed ? "已改写" : "未改写") + " · " + esc(data.got_source || "") +
          " · " + esc(data.got_reason) +
          "<br>结果：" + esc(data.got_text);
      } catch (err) {
        $("corefOne").textContent = String(err.message || err);
      }
    };

    $("btnRf").onclick = async () => {
      const input = $("rfPick");
      const file = input && input.files && input.files[0];
      if (!file) {
        $("rfHint").textContent = "先选 PDF";
        return;
      }
      $("rfHint").textContent = "入库评测中（本地版面识别约 1 分钟，写 RAGFlow 视服务快慢可能数分钟）…";
      $("btnRf").disabled = true;
      try {
        const body = new FormData();
        body.append("file", file, file.name);
        const src = $("rfSource").value || "local";
        const skip = $("rfSkip") && $("rfSkip").checked ? "1" : "0";
        const cfg = collectChunkCfg();
        const qs = new URLSearchParams({
          chunk_source: src,
          skip_ragflow: skip,
          chunk_token_num: String(cfg.chunk_token_num || ""),
          local_max_chars: String(cfg.local_max_chars || ""),
          local_naive_chars: String(cfg.local_naive_chars || ""),
          local_overlap: String(cfg.local_overlap || ""),
          embedding_model: cfg.embedding_model || "",
          ragflow_embedding_model: cfg.ragflow_embedding_model || "",
        });
        const res = await fetch("/api/eval/ragflow-ingest?" + qs.toString(), {
          method: "POST",
          body,
        });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          $("rfHint").textContent = data.detail || ("HTTP " + res.status);
          $("rfOk").innerHTML = pill(false);
          return;
        }
        $("rfOk").innerHTML = pill(!!data.ok) + (data.local_only ? " 仅本地" : "");
        $("rfMethod").textContent = data.method || src;
        const rf = data.ragflow || {};
        const web = (rf.web_url || (data.steps || []).map((s) => (s.extra || {}).web_url).filter(Boolean)[0] || "");
        $("rfWeb").innerHTML = web ? '<a href="' + esc(web) + '" target="_blank">' + esc(web) + "</a>" : "-";
        if (data.chunk_config) fillChunkCfg(data.chunk_config);
        $("rfHint").textContent = data.ok
          ? (data.local_only
              ? (data.skip_ragflow ? "本地分块完成，按选项跳过 RAGFlow 写入" : "本地分块完成，RAGFlow 未写入")
              : "链路完成")
          : "有步骤失败";
        renderLayoutEval({ phases: "rfPhases", pages: "rfPages", blocks: "rfBlocks" }, data);
        $("rfSteps").innerHTML = (data.steps || []).filter((s) => !s.extra || !s.extra.phase).map((s) => {
          const ops = (s.extra && s.extra.ops) || [];
          const opsHtml = ops.length
            ? "<ul style='margin:6px 0 0 18px;padding:0'>" + ops.map((o) => "<li>" + esc(o) + "</li>").join("") + "</ul>"
            : "";
          return "<tr><td>" + pill(!!s.ok) + "</td><td>" + esc(s.title) +
            "</td><td>" + esc(s.detail) + opsHtml + "</td></tr>";
        }).join("");
        $("rfPrev").innerHTML = (data.previews || []).length
          ? (data.previews || []).map((p) =>
              "<tr><td>" + esc(p.ord) + "</td><td>" + esc(p.page) + "</td><td>" + esc(p.text) + "</td></tr>"
            ).join("")
          : "<tr><td colspan='3'>无预览</td></tr>";
      } catch (err) {
        $("rfHint").textContent = String(err.message || err);
      } finally {
        $("btnRf").disabled = false;
      }
    };

    $("btnFile").onclick = async () => {
      const input = $("filePick");
      const file = input && input.files && input.files[0];
      if (!file) {
        $("fileHint").textContent = "先选文件";
        return;
      }
      $("fileHint").textContent = "评测中…";
      $("btnFile").disabled = true;
      try {
        const body = new FormData();
        body.append("file", file, file.name);
        const res = await fetch("/api/eval/file", { method: "POST", body });
        const data = await res.json().catch(() => ({}));
        if (!res.ok) {
          $("fileHint").textContent = data.detail || ("HTTP " + res.status);
          return;
        }
        const card = data.score_card || {};
        const intent = data.intent || {};
        const pre = data.preprocess || {};
        $("fileScore").textContent =
          String(card.score ?? "-") + " / 阈值 " + String(card.threshold ?? 2) +
          (card.lab_report ? " · 报告解读" : " · 非报告");
        $("fileIntent").textContent =
          (intent.major || "-") + " / " + (intent.intent || "-") + " / " + (intent.handler || "-");
        $("fileArt").textContent =
          (pre.artifact === "pdf" ? "PDF " + (pre.mineru_pdf_path || "") :
            pre.artifact === "structured_chunks" ? "结构化块 JSON" : "空") +
          (data.tools && data.tools.length ? " · 工具 " + data.tools.join("、") : "");
        $("fileHint").textContent = (pre.reasons || []).join("；") || data.filename || "完成";
        $("fileScoreBody").innerHTML = (card.points || []).map((p) =>
          "<tr><td>" + esc(p.name) + "</td><td>" + (p.hit ? "是" : "否") +
          "</td><td>" + esc(p.delta) + "</td><td>" + esc(p.detail) + "</td></tr>"
        ).join("");
        renderLayoutEval({ phases: "filePhases", pages: "filePages", blocks: "fileBlocks" }, data);
        const te = data.table_extract || {};
        $("fileTableHint").textContent = te.ok
          ? ("抽到 " + (te.table_count || 0) + " 张表")
          : (te.error || "未跑 Unstructured 拆表");
        $("fileTables").textContent = JSON.stringify(te.tables || [], null, 2);
        $("filePre").textContent = JSON.stringify(pre, null, 2);
        if (data.report_split) {
          $("fileSplitHint").textContent = data.report_split.summary || "已拆表";
          $("fileSplit").textContent = JSON.stringify(data.report_split, null, 2);
        } else {
          $("fileSplitHint").textContent = "本轮不是报告解读，不跑拆表";
          $("fileSplit").textContent = "-";
        }
      } catch (err) {
        $("fileHint").textContent = String(err.message || err);
      } finally {
        $("btnFile").disabled = false;
      }
    };

    $("btnEnv").onclick = async () => {
      $("envBox").textContent = "探测中…";
      const res = await fetch("/api/eval/env");
      const data = await res.json();
      $("envBox").innerHTML = (data.checks || []).map((c) =>
        pill(c.ok) + " " + esc(c.name) + " " + esc(c.detail)
      ).join("<br>");
    };
    $("btnRun").onclick = async () => {
      $("sum").textContent = "运行中…";
      const res = await fetch("/api/eval/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      const data = await res.json();
      $("sum").textContent = "通过 " + data.passed + " / " + data.total;
      $("goldBody").innerHTML = (data.gold || []).map((r) =>
        "<tr><td>" + pill(r.ok) + "</td><td>" + esc(r.text) + "</td><td>" +
        esc(r.expect_major) + " / " + esc(r.expect_handler) + "</td><td>" +
        esc(r.got_major) + " / " + esc(r.got_handler) + "</td><td>" +
        esc((r.got_tools || []).join("、") || "空") + "</td></tr>"
      ).join("");
      $("bindBody").innerHTML = (data.binds || []).map((r) =>
        "<tr><td>" + pill(r.ok) + "</td><td>" + esc(r.text) + "</td><td>" +
        esc((r.expect_tools || []).join("、") || "空") + "</td><td>" +
        esc((r.got_tools || []).join("、") || "空") + "</td></tr>"
      ).join("");
    };
  