import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

const root = dirname(fileURLToPath(import.meta.url));

const PROXY_TARGETS = {
  memory: {
    "/api": "http://127.0.0.1:8015",
  },
  report_read: {
    "/gate": "http://127.0.0.1:8012",
    "/facts": "http://127.0.0.1:8012",
    "/handbook": "http://127.0.0.1:8012",
    "/compose": "http://127.0.0.1:8012",
  },
  doc_ingest: {
    "/ingest": "http://127.0.0.1:8013",
  },
  directed_crawl: {
    "/fetch": "http://127.0.0.1:8014",
    "/catalog": "http://127.0.0.1:8014",
    "/webhook": "http://127.0.0.1:8014",
    "/jobs": "http://127.0.0.1:8014",
    "/compose": "http://127.0.0.1:8014",
    "/pipeline": "http://127.0.0.1:8014",
    "/rpa": "http://127.0.0.1:8014",
    "/ingest": "http://127.0.0.1:8014",
  },
  companion: {},
};

const MEMORY_API = "http://127.0.0.1:8015";
const B2C_API = "http://127.0.0.1:8002";

/** shell 开发：记忆与陪聊 API 分端口，避免 /api 整段打到 8015 */
function shellDevProxy() {
  const memoryRoutes = [
    "/api/me",
    "/api/login",
    "/api/logout",
    "/api/timeline",
    "/api/search",
    "/api/rag",
    "/api/card",
  ];
  const proxy = {
    "/b2c": {
      target: B2C_API,
      changeOrigin: true,
      rewrite: (path) => path.replace(/^\/b2c/, ""),
    },
    "/b2c_persist.js": { target: B2C_API, changeOrigin: true },
    "/api/eval": { target: B2C_API, changeOrigin: true },
    "/api/qa": { target: B2C_API, changeOrigin: true },
    "/api/tools": { target: B2C_API, changeOrigin: true },
    "/api/users/suggestions": { target: MEMORY_API, changeOrigin: true },
    "/api/users": { target: B2C_API, changeOrigin: true },
  };
  for (const prefix of memoryRoutes) {
    if (prefix === "/api/users") continue;
    proxy[prefix] = { target: MEMORY_API, changeOrigin: true };
  }
  const CRAWL_API = "http://127.0.0.1:8014";
  for (const prefix of ["/fetch", "/catalog", "/webhook", "/jobs", "/compose", "/pipeline", "/rpa"]) {
    proxy[prefix] = { target: CRAWL_API, changeOrigin: true };
  }
  proxy["/ingest/excel"] = { target: CRAWL_API, changeOrigin: true };
  return proxy;
}

function proxyForMode(mode) {
  const key = PROXY_TARGETS[mode] ? mode : "shell";
  if (key === "shell") {
    return shellDevProxy();
  }
  const targets = PROXY_TARGETS[key];
  const proxy = {};
  for (const [prefix, target] of Object.entries(targets)) {
    proxy[prefix] = { target, changeOrigin: true };
  }
  return proxy;
}

export default defineConfig(({ mode }) => ({
  root,
  base: "/",
  plugins: [vue()],
  server: {
    port: 5173,
    proxy: proxyForMode(mode),
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
    rollupOptions: {
      input: {
        console: resolve(root, "index.html"),
      },
    },
  },
}));
