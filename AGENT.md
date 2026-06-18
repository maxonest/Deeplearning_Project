# AGENT.md

本文件供参与本仓库开发的 AI Agent 和开发者使用。修改代码前先阅读本文件与 `README.md`，并以当前仓库实现为准，不要仅依据通用模板推断项目行为。

## 项目目标

这是一个面向本地专业知识领域的问答系统，主要运行环境为 Windows 10/11、Python 3.10+、Node.js 18+ 和 NVIDIA RTX 4090 D。

核心链路：

1. 将原始资料整理到 `data/processed/`。
2. 使用 sentence-transformers 生成向量，并持久化为 FAISS 索引。
3. FastAPI 根据问题检索 top-k 文档、拼接多轮记忆和 RAG 提示词。
4. Transformers 从本地目录加载 Qwen 模型，并通过 SSE 流式输出。
5. React 前端展示 Markdown、检索来源和 `<think>...</think>` 思考内容。
6. `models/train_lora.py` 使用 PyTorch 自定义训练循环和 PEFT 完成 LoRA/QLoRA SFT。

## 目录职责

- `backend/`：FastAPI API、模型加载、RAG、会话记忆和请求模型。
- `embeddings/`：语料切块、embedding、FAISS 建库和查询。
- `frontend/`：React 18 + Vite 前端。
- `models/`：本地推理入口、LoRA/QLoRA 训练脚本和本地模型目录。
- `utils/`：配置、数据加载、语料清洗和日志工具。
- `tests/`：不依赖真实模型权重或已有 FAISS 索引的单元测试。
- `data/raw/`：未经处理的原始资料。
- `data/processed/`：清洗后、可直接参与知识库构建的资料。
- `data/finetune/`：监督微调数据集。
- `embeddings/faiss_index/`：生成的 FAISS 索引和元数据，不提交 Git。

## 关键约束

### 路径与配置

- 所有项目路径应优先基于 `pathlib.Path` 和项目根目录解析，不能依赖当前工作目录。
- 本地配置从根目录 `.env` 读取；新增配置时同步更新 `.env.example` 和 `README.md`。
- Windows 默认模型路径为 `models/qwen/Qwen3.5-9B`。
- 不要将模型权重、LoRA checkpoint、FAISS 索引、`.env`、日志或 `node_modules` 提交到 Git。
- 需要直接运行子目录脚本时，确保项目根目录可被 Python 导入，避免出现 `No module named 'utils'`。

### 本地模型

- 模型必须延迟加载，不能让 FastAPI 导入阶段直接占满显存。
- 优先使用本地文件，保留 `local_files_only` 配置。
- Transformers 模型加载参数使用 `dtype`，不要重新使用已弃用的 `torch_dtype`。
- CPU 回退可以用于功能验证，但正式推理和训练应检查 CUDA。
- 流式生成必须维持 `TextIteratorStreamer` 和 SSE 接口兼容性。
- 深度思考由请求字段 `enable_thinking` 控制；不要强制所有请求启用思考。

### RAG 与会话

- 普通寒暄不应提示“未检索到知识库”，也不应强行走专业 RAG 回答。
- 专业问题才检索知识库，并将文档、历史上下文和当前问题组装为提示词。
- FAISS 索引是持久化产物。只有语料、切块参数或 embedding 模型改变时才重建，不要在每次服务启动时自动重建。
- 构建索引与查询索引必须使用相同的 embedding 模型。
- 会话内存目前保存在进程内；修改时保持线程安全，并注意服务重启后历史不会保留。

### 微调

- `models/train_lora.py` 当前使用 PyTorch 自定义训练循环和 PEFT。
- 不要轻易改回 Hugging Face `Trainer`、`datasets` 或 `pyarrow` 训练链路。该组合曾在 Windows 环境导入时触发 access violation。
- 训练只对 assistant 输出计算 loss，用户提示词和系统提示词标签必须保持为 `-100`。
- 优先使用 tokenizer 自带的 `chat_template`，只有缺失时才使用 ChatML 回退格式。
- 保持梯度累积、warmup、梯度裁剪、验证、checkpoint 和 SwanLab 指标上报功能。
- 训练日志中的 loss 应为未除以梯度累积步数的实际平均 loss。
- `grad_norm` 应为有限值且大于 0；出现 `0`、`NaN` 或持续异常增大时应立即排查。
- 正式训练前先执行 dry run 和小样本过拟合测试。

### 前端

- 保持界面简洁、干净、适合重复使用的工作台，不照搬任何现有产品的品牌视觉。
- `Enter` 发送，`Shift+Enter` 换行。
- 助手消息支持 Markdown；`<think>...</think>` 内容使用独立的较暗背景块显示。
- 前端默认调用 `/api/chat/stream`，修改后端事件格式时必须同步更新 SSE 解析逻辑。
- 保持 Vite 的相对资源路径配置，以便 GitHub Pages 静态预览。
- GitHub Pages 只能预览静态前端，不能托管本地模型或 FastAPI 后端。

## 数据格式

SFT 数据支持 JSON 数组或 JSONL，标准字段为：

```json
{
  "instruction": "根据专业知识回答用户问题",
  "input": "用户问题",
  "output": "标准答案",
  "metadata": {
    "source": "manual"
  }
}
```

修改数据处理逻辑时，应继续兼容 `question/query`、`answer/response` 等已有别名，并拒绝缺失 input/output 的记录。

## 常用命令

Windows 环境：

```powershell
conda activate local-domain-qa
python models/run_local_model.py --prompt "你好，请介绍一下你自己"
python start_windows.py
```

语料与知识库：

```powershell
python utils/prepare_corpus.py
python embeddings/embed_utils.py build --input data/processed --output embeddings/faiss_index
python embeddings/embed_utils.py query "测试问题" --top_k 5
```

训练前验证与正式训练：

```powershell
python models/train_lora.py --dry_run --max_samples 3
python models/train_lora.py --no_swanlab --max_samples 32 --epochs 3 --logging_steps 1
python models/train_lora.py
```

测试和前端构建：

```powershell
pytest
cd frontend
npm install
npm run build
```

macOS/Linux 可使用 `./run.sh`；Windows 必须使用 `python start_windows.py`。

## 修改后的验证要求

- 修改 Python 公共逻辑：运行 `pytest`。
- 修改单个 Python 脚本：至少运行 `python -m compileall <文件或目录>` 和对应的 `--help`/dry run。
- 修改 RAG 或 API：运行相关 pytest，并验证 `/health`、`/api/chat` 或 `/api/chat/stream`。
- 修改 embedding：运行切块相关测试，并至少执行一次小语料 build/query。
- 修改训练：先 dry run，再运行 32 条样本的小规模训练，确认 loss、learning rate、grad_norm 和 checkpoint 正常。
- 修改前端：运行 `npm run build`，并检查桌面与窄屏布局、流式输出、Markdown、思考块和键盘发送。
- 不要求测试真实 9B 模型的改动时，应明确说明没有进行完整 GPU 验证。

## 开发原则

- 优先沿用现有结构和依赖，避免无关重构。
- 新增复杂依赖前先确认 Windows、CUDA 和离线运行兼容性。
- 错误信息应包含实际路径、缺失依赖或失败阶段，不能静默退出。
- 终端只输出必要进度；详细训练诊断写入日志文件或 SwanLab。
- 不在代码中写入访问令牌、绝对用户目录或机器专属信息。
- 不删除或覆盖用户数据、模型、索引和训练产物。
- 不自动执行 Git commit、push、reset 或 checkout，除非用户明确要求。
