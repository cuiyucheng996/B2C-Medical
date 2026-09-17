# B2C business

基于 **Spring AI**（`spring-ai-alibaba-starter-dashscope`，通义千问 DashScope）的 C 端业务服务。

**双栈架构**：生产环境只部署 **JDK + Spring Boot JAR**；Python 仅用于离线训练 BERT 纠错模型并导出 ONNX，不进生产服务器。Agent 开发规范见 `.agents/skills/b2c-business/SKILL.md`。

## 技术栈与版本

| 组件 | 版本 | 说明 |
|---|---|---|
| Java | 17 | 本机已装 JDK 17.0.12 LTS |
| Spring Boot | 3.5.10 | 与 dashscope starter 锁定版本对齐 |
| Spring AI | 1.1.2 | 由 `spring-ai-bom` 统一管理 |
| Spring AI Alibaba | 1.1.2.2 | `spring-ai-alibaba-starter-dashscope` |
| Maven | 3.9.16 | 通过 Maven Wrapper 使用 |

## 目录结构

```
B2C business/
├── pom.xml
├── mvnw / mvnw.cmd
├── .agents/skills/b2c-business/   # Agent 双栈开发规范
├── src/main/java/com/uvgraph/b2c/ # Java 业务（生产代码）
│   ├── controller/                # REST 入口
│   ├── service/                   # 业务编排（待扩展）
│   └── inference/                 # ONNX 推理（待扩展）
├── training/                      # Python 离线训练（目标位置，待迁移）
├── data/ checkpoint/ pretrained/ models/
└── src/main/resources/application.yml
```

当前 Python（`src/configs/`、`src/models/`）与 Maven 标准目录混用，后续迁入 `training/`。详见 `.agents/skills/b2c-business/architecture.md`。

## 缓存策略

Maven 发行版与依赖仓库全部放在项目内 `.m2` 目录（`mvnw` 已内置 `MAVEN_USER_HOME` 指向），**不写 C 盘**。如需换位置，修改 `mvnw.cmd` / `mvnw` 顶部的 `MAVEN_USER_HOME` 即可。

## 运行

```powershell
# 1. 准备 API Key（与主项目 uvgraph 共用 .env 里的 DASHSCOPE_API_KEY）
$env:DASHSCOPE_API_KEY = "sk-xxxx"

# 2. 首次构建（自动下载 Maven + 依赖，进 .m2）
.\mvnw.cmd clean package -DskipTests

# 3. 启动
.\mvnw.cmd spring-boot:run
```

启动后：

- 健康检查：http://127.0.0.1:8080/actuator/health
- Chat 示例：`GET http://127.0.0.1:8080/api/chat?message=你好`

## 常用命令

```powershell
.\mvnw.cmd -v                    # 查看 Maven 版本
.\mvnw.cmd compile               # 编译
.\mvnw.cmd clean package -DskipTests   # 打包
.\mvnw.cmd spring-boot:run       # 运行
```

## 双栈分工

| 层 | 技术 | 环境 |
|---|---|---|
| 业务 API、LLM 对话 | Java Spring Boot + Spring AI | JDK 生产 |
| BERT 纠错推理 | Java + ONNX Runtime | JDK 生产 |
| 模型训练、导出 ONNX | Python PyTorch | 开发机 / CI |

## 后续扩展方向

1. 迁移 Python 到 `training/`，补 `export_onnx.py`
2. Java 增加 `SpellCheckInferenceEngine` + `SpellCheckController`
3. 问诊 Agent：`ChatClient` + Tool Calling（可调 uvgraph 主服务 RAG）
4. Neo4j / Redis / MySQL 按需在 `pom.xml` 追加
