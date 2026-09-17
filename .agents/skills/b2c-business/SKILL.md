---
name: b2c-business
description: >-
  B2C business 双栈架构规范：JDK 生产环境跑 Spring Boot 业务，Python 仅离线训练与导出模型。
  Use when editing B2C business/, spell check BERT, Spring AI DashScope, Maven/Java 业务层,
  Python 训练脚本, ONNX 导出, JDK 部署, or planning controller/service/inference 分层.
---

# B2C business 双栈规范

## 核心原则

| 层 | 语言 | 运行环境 | 职责 |
|---|---|---|---|
| **业务运行时** | Java 17 | JDK 生产服务器 | REST API、会话、Spring AI 对话、ONNX 推理 |
| **模型训练** | Python | 开发机 / CI（不进生产） | 数据预处理、BERT 微调、导出 ONNX |
| **产物交接** | 文件 | `models/` 目录 | Python 导出 → Java 加载，不经 HTTP 调 Python |

**生产 JDK 环境只部署 Spring Boot fat JAR + ONNX 权重，不部署 Python 运行时。**

## 目标目录结构

```
B2C business/
├── pom.xml
├── mvnw / mvnw.cmd
├── src/main/java/com/uvgraph/b2c/     # Java 业务（唯一生产代码）
│   ├── B2cBusinessApplication.java
│   ├── config/                        # AppProperties、路径、DashScope
│   ├── controller/                    # REST 入口，薄层
│   ├── service/                       # 业务编排（问诊、纠错、对话）
│   ├── inference/                     # ONNX Runtime 加载 spell_check 模型
│   └── client/                        # 外部 HTTP 客户端（如需调 uvgraph 主服务）
├── src/main/resources/
│   ├── application.yml
│   └── models/                        # 打包进 JAR 或外挂目录（spell_check.onnx）
├── training/                          # Python 离线训练（不进生产）
│   ├── configs/config.py
│   ├── models/spell_check_bert.py
│   ├── preprocess/
│   ├── runner/                        # train.py、export_onnx.py
│   └── requirements.txt
├── data/                              # 训练数据（gitignore 大文件）
├── checkpoint/                        # PyTorch 训练 checkpoint
├── pretrained/                        # bert-base-chinese 等预训练权重
├── models/                            # 导出产物（ONNX + tokenizer），供 Java 读取
└── logs/
```

### 当前待迁移

Python 暂在 `src/configs/`、`src/models/`（与 Maven `src/main/` 混用）。**新代码一律放 `training/`**；改旧文件时顺手迁移，不在 `src/` 下新增 Python。

## Java 分层约定

```
Controller → Service → (InferenceEngine | ChatClient)
```

| 包 | 职责 | 禁止 |
|---|---|---|
| `controller` | 参数校验、HTTP 映射、返回 DTO | 业务逻辑、直接调 ONNX |
| `service` | 业务流程编排 | PyTorch 概念、训练逻辑 |
| `inference` | ONNX 加载、tokenize、predict | REST、会话管理 |
| `config` | `application.yml` 绑定 | 散落 `System.getenv` |

Spring AI（DashScope）对话走 `ChatClient`；BERT 纠错走 `inference` 包 ONNX，**不混在一个 Service 里硬编码两种后端**。

## Python 训练约定

| 目录 | 职责 |
|---|---|
| `training/configs/` | 路径常量（ROOT_DIR、DATA_DIR、CHECKPOINT_DIR） |
| `training/models/` | `nn.Module` 定义（如 `SpellCheckBert`） |
| `training/preprocess/` | 原始数据 → tensor 数据集 |
| `training/runner/train.py` | 训练入口 |
| `training/runner/export_onnx.py` | 导出 ONNX → `models/spell_check.onnx` |

训练脚本只做 ML，**不写 HTTP 接口、不做业务编排**。

## 模型交接流程

```
1. Python 训练 → checkpoint/best.pt
2. export_onnx.py → models/spell_check.onnx (+ tokenizer 文件)
3. Java SpellCheckInferenceEngine 加载 models/
4. mvn package → 部署 JAR
```

Java 侧重载 ONNX，不加载 PyTorch。改模型结构后必须重新 export，并同步 Java 侧输入输出 shape。

## 配置

| 配置项 | 位置 | 说明 |
|---|---|---|
| DashScope API Key | `application.yml` + 环境变量 `DASHSCOPE_API_KEY` | 与 uvgraph 主项目共用 |
| 模型路径 | `application.yml` `app.models.spell-check-dir` | 默认 `models/` |
| Python 路径 | `training/configs/config.py` | 仅训练用，Java 不读 |

## Agent 行为规范

### 必须

1. **先判断栈**：业务/API/部署 → Java；训练/预处理/导出 → Python
2. **单一维护入口**：路径常量 Java 用 `AppProperties`，Python 用 `training/configs/config.py`
3. **最小改动**：只改任务相关文件，不顺手重构无关模块
4. **保留已有注释**（含 Java/Python 行内注释）
5. 改 Java 后跑 `.\mvnw.cmd compile`；改 Python 导出脚本后说明需 re-export ONNX
6. Maven 缓存继续用项目内 `.m2`，不写 C 盘

### 禁止

1. 在生产 Java 代码中引入 PyTorch、transformers、`ProcessBuilder` 调 Python 推理
2. 在 Python `runner/` 里写 Spring Controller 或业务会话逻辑
3. 把 `RouterState`、LangGraph 状态放进 Java DTO 以外的地方硬编码
4. 在 `src/main/java` 以外放 Java 源码
5. 新建与 `training/` 平行的第三套 Python 目录

### 新增功能决策

| 需求 | 做法 |
|---|---|
| 新 REST 接口 | Java `controller` + `service` |
| LLM 对话 / Agent | Java `ChatClient` + Spring AI |
| BERT 纠错推理 | Java `inference` + ONNX |
| 新训练任务 / 换模型 | Python `training/runner/` |
| 复杂 NLP 暂无法 ONNX 化 | 先评估 Spring AI 能否替代；不行再设计独立推理服务（非默认） |

## 部署清单（JDK 生产）

```powershell
# 构建
cd "B2C business"
$env:DASHSCOPE_API_KEY = "sk-xxxx"
.\mvnw.cmd clean package -DskipTests

# 运行（仅需 JRE 17+）
java -jar target/b2c-business-0.0.1-SNAPSHOT.jar
```

部署包：`target/*.jar` + `models/`（若未打进 JAR）+ 环境变量。**无需 Python、无需 GPU。**

## 常用命令

```powershell
# Java
.\mvnw.cmd compile
.\mvnw.cmd spring-boot:run

# Python（仅开发机，在 training 同级或 training 内执行）
cd training
pip install -r requirements.txt
python runner/train.py
python runner/export_onnx.py
```

## 延伸阅读

- 架构与接口约定详见 [architecture.md](architecture.md)
