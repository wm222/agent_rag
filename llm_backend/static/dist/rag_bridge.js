(function () {
  const oldFetch = window.fetch.bind(window);

  function getUserId() {
    return localStorage.getItem("user_id") || "1";
  }

  function getToken() {
    return localStorage.getItem("token") || "";
  }

  function makeJsonResponse(data, status = 200) {
    return new Response(JSON.stringify(data), {
      status,
      headers: {
        "Content-Type": "application/json"
      }
    });
  }

  window.fetch = async function (input, init = {}) {
    const url = typeof input === "string" ? input : input.url;

    // =========================
    // 1. 拦截原页面上传按钮
    // 原页面默认上传到 /upload
    // 这里改成上传到 /api/rag/upload
    // =========================
    if (
      url &&
      (
        url === "http://localhost:8000/upload" ||
        url.endsWith("/upload")
      ) &&
      !url.includes("/api/rag/upload")
    ) {
      try {
        const oldBody = init.body;
        const formData = new FormData();
        let uploadFile = null;

        if (oldBody instanceof FormData) {
          for (const [key, value] of oldBody.entries()) {
            formData.append(key, value);
            if (key === "file") {
              uploadFile = value;
            }
          }
        }

        if (!formData.has("user_id")) {
          formData.append("user_id", getUserId());
        }

        const res = await oldFetch("/api/rag/upload", {
          method: "POST",
          headers: {
            Authorization: `Bearer ${getToken()}`
          },
          body: formData
        });

        const data = await res.json();

        if (!res.ok) {
          return makeJsonResponse(data, res.status);
        }

        if (data.index_id) {
          localStorage.setItem("active_rag_index_id", data.index_id);
          localStorage.setItem("rag_enabled", "1");
        }

        return makeJsonResponse({
          status: "success",
          index_id: data.index_id,
          original_name: data.filename || (uploadFile && uploadFile.name) || "uploaded_file",
          size: (uploadFile && uploadFile.size) || 0,
          type: (uploadFile && uploadFile.type) || "",
          chunk_count: data.chunk_count,
          dimension: data.dimension
        });
      } catch (e) {
        console.error("[RAG Bridge] upload failed:", e);
        return makeJsonResponse({
          status: "error",
          detail: e.message || "RAG upload failed"
        }, 500);
      }
    }

    // =========================
    // 2. 拦截主聊天请求
    // 如果知识库问答开启，就自动携带 RAG 参数
    // =========================
    if (url && url.includes("/api/chat") && init && init.body) {
      try {
        const ragEnabled = localStorage.getItem("rag_enabled") === "1";
        const activeIndexId = localStorage.getItem("active_rag_index_id");

        if (ragEnabled && activeIndexId) {
          const body = JSON.parse(init.body);

          body.rag_enabled = true;
          body.rag_index_id = activeIndexId;
          body.top_k = Number(localStorage.getItem("rag_top_k") || 4);

          init = {
            ...init,
            body: JSON.stringify(body)
          };
        }
      } catch (e) {
        console.warn("[RAG Bridge] chat patch failed:", e);
      }
    }

    return oldFetch(input, init);
  };

  // =========================
  // 3. 监听原页面“知识库问答”按钮
  // =========================
  document.addEventListener("click", function (event) {
    const btn = event.target.closest("button");
    if (!btn) return;

    if (btn.innerText && btn.innerText.includes("知识库问答")) {
      setTimeout(() => {
        const isActive = btn.classList.contains("tool-btn-active");
        const activeIndexId = localStorage.getItem("active_rag_index_id");

        if (isActive) {
          localStorage.setItem("rag_enabled", "1");

          if (!activeIndexId) {
            alert("已开启知识库问答模式，但还没有可用索引。请先点击上传文件。");
          } else {
            console.log("[RAG Bridge] RAG enabled, index:", activeIndexId);
          }
        } else {
          localStorage.setItem("rag_enabled", "0");
          console.log("[RAG Bridge] RAG disabled.");
        }
      }, 80);
    }
  }, true);

  console.log("[RAG Bridge] loaded.");
})();
