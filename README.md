# Local Domain QA System

本项目是一套本地专业知识领域问答系统，包含数据集处理、FAISS 向量知识库、RAG 后端、本地大模型推理、LoRA/QLoRA 微调、React 前端、Windows 启动器和自动化测试。

## 功能模块

- 数据集构建：JSON/JSONL 数据加载、字段标准化、训练/验证/测试划分。
- 知识库构建：读取 `txt/md/csv` 语料，切块后用 sentence-transformers 生成 embedding，并保存 FAISS 索引。
- 本地 RAG：FastAPI 接收问题，管理多轮会话，检索 top-k 文档，拼接上下文后调用本地模型。
- 本地模型：支持通过 Transformers 加载 `Qwen3.5-9B` 等本地 causal LM，并可通过 PEFT 挂载 LoRA adapter。
- 微调训练：基于 PyTorch 自定义训练循环 + PEFT 进行 LoRA/QLoRA SFT，避免 Windows 环境中 `Trainer/datasets/pyarrow` 的兼容性问题。
- 前端页面：React 18 + Vite 多轮对话 UI，支持 top-k 参数和召回来源展示。
- 测试：覆盖数据处理、多轮记忆、RAG 提示词链路和 FastAPI 接口。

## 目录结构

```text
project_root/
├─ backend/                 # FastAPI、RAG、多轮记忆、本地模型客户端
├─ data/
│  ├─ raw/                  # 原始语料，不提交大文件
│  ├─ processed/            # 清洗后语料，不提交大文件
│  └─ dataset.json          # SFT 示例数据集
├─ embeddings/              # FAISS 知识库构建与检索代码
├─ frontend/                # React 18 + Vite 前端
├─ models/
│  ├─ qwen/                 # Qwen 本地权重目录，不提交 GitHub
│  ├─ deepseek/             # DeepSeek 本地权重目录，不提交 GitHub
│  ├─ run_local_model.py    # 本地模型单独推理测试
│  └─ train_lora.py         # LoRA/QLoRA 训练脚本
├─ tests/                   # pytest 测试
├─ utils/                   # 配置、数据处理、日志工具
├─ .env.example             # Windows/本地模型配置样例
├─ requirements.txt         # Linux/通用依赖，含训练依赖
├─ requirements-windows.txt # Windows 推理部署依赖
├─ run.sh                   # macOS/Linux 启动脚本
└─ start_windows.py         # Windows 启动脚本
```

## Windows 机房部署

假设项目在：

```text
D:/lyx/Deeplearning_Project
```

模型在：

```text
D:/lyx/Deeplearning_Project/models/qwen/Qwen3.5-9B
```

### 1. 创建环境

在 Anaconda Prompt 或 PowerShell 中执行：

```powershell
cd D:/lyx/Deeplearning_Project
conda create -n local-domain-qa python=3.10 -y
conda activate local-domain-qa

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install -c conda-forge faiss-cpu -y
pip install -r requirements-windows.txt
```

Windows 上建议用 Conda 安装 `faiss-cpu`，其余 Python 包再用 `requirements-windows.txt` 安装。模型提示缺少 fast path 可选加速库时可以先忽略，系统会回退到 PyTorch 实现。

### 2. 配置模型路径

```powershell
copy .env.example .env
```

确认 `.env` 中：

```env
USE_LOCAL_MODEL=true
LOCAL_MODEL_PATH=models/qwen/Qwen3.5-9B
LOCAL_LORA_ADAPTER_PATH=models/qwen/lora_adapter
LOCAL_MODEL_MAX_NEW_TOKENS=2048
LOCAL_MODEL_TEMPERATURE=0.2
LOCAL_MODEL_TOP_P=0.9
LOCAL_MODEL_ENABLE_THINKING=false
```

### 3. 单独测试模型

```powershell
python models/run_local_model.py --prompt "你好，请介绍一下你自己"
```

配置了 `LOCAL_LORA_ADAPTER_PATH` 后，上述命令会先加载 `LOCAL_MODEL_PATH`
中的基础模型，再挂载 LoRA adapter。也可以在命令行中显式指定：

```powershell
python models/run_local_model.py --model_path models/qwen/Qwen3.5-9B --lora_adapter_path models/qwen/lora_adapter --prompt "你好，请介绍一下你自己"
```

从其他机器复制 adapter 时，目录中至少需要包含
`adapter_config.json` 和 `adapter_model.safetensors`（或 `adapter_model.bin`），
并且基础模型必须与训练 LoRA 时使用的模型兼容。不使用 LoRA 时，从 `.env`
中删除 `LOCAL_LORA_ADAPTER_PATH` 或将其留空。

如果能正常输出，再启动系统。

### 4. 启动后端测试模型

只启动 FastAPI，不启动前端：

```powershell
python start_windows.py --backend-only
```

当 `USE_LOCAL_MODEL=true` 时，FastAPI 启动后会在后台加载基础模型并挂载 LoRA。
等待 `/health` 返回 `startup_ready=true` 后，再在另一个 PowerShell 窗口执行：

```powershell
$body = @{
  prompt = "请用一句话说明你擅长的专业领域"
  enable_thinking = $false
} | ConvertTo-Json

Invoke-RestMethod `
  -Uri "http://localhost:8000/api/model/test" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

模型加载耗时会转移到后端启动阶段，第一次请求无需再等待权重加载。返回结果中的
`model_loaded` 应为 `true`，`local_lora_adapter_path` 应指向
`models/qwen/lora_adapter`。也可以访问 `http://localhost:8000/health`
确认模型是否已经加载。

FastAPI 启动后会立即开放 `/health`，知识库和模型在后台继续初始化。前端每
1.5 秒轮询健康状态，并依次显示“知识库构建中”“模型加载中”“系统就绪”。
初始化完成前输入框和发送按钮保持禁用，完成后自动解锁，无需刷新页面。

如果基础模型或 LoRA 路径错误、adapter 文件不完整，`/health` 会返回
`startup_phase=failed` 和具体的 `startup_error`，聊天接口保持不可用。不要使用多个
Uvicorn worker，否则每个 worker 都会各自加载一份模型并占用显存。

Windows 后端启动日志会同时写入 `logs/backend_startup.log`。如果后端进程出现
`3221225725 (0xC00000FD)`，这是 Windows 原生栈溢出，而不是普通 Python 异常。
请提供日志中最后一条 `Model load stage` 以及其后的 faulthandler 调用栈，用于判断
崩溃发生在依赖导入、tokenizer、基础模型还是 LoRA adapter 加载阶段。

`start_windows.py` 不会在终端实时输出后端日志；Uvicorn、知识库和模型加载日志只写入
`logs/backend_startup.log`。终端仅显示服务地址、日志路径和进程退出摘要。
需要首次构建或更新知识库时，终端会额外显示一条单行刷新的编码进度条，例如：

```text
Knowledge-base encoding [██████████████░░░░░░░░░░░░░░] 50.00%  3200/6400
```

索引未变化并被直接复用时不会显示该进度条。

macOS/Linux 可使用：

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

然后测试：

```bash
curl -X POST http://localhost:8000/api/model/test \
  -H "Content-Type: application/json" \
  -d '{"prompt":"请用一句话说明你擅长的专业领域","enable_thinking":false}'
```

### 5. 启动前后端

Windows 不使用 `run.sh`：

```powershell
python start_windows.py
```

访问：

```text
http://localhost:5173
```

后端地址：

```text
http://localhost:8000
```

## macOS/Linux 快速开始

```bash
conda create -n local-domain-qa python=3.10 -y
conda activate local-domain-qa
pip install -r requirements.txt

cd frontend
npm install
cd ..

./run.sh
```

## 语料整理与 FAISS 知识库

推荐目录职责：

```text
data/raw/        # 原始长文本、手工收集资料
data/processed/  # 清洗后的文本、可直接检索的 JSON/JSONL 问答数据
```

当前项目支持把 `.txt/.md/.csv/.json/.jsonl` 放入 `data/processed` 参与建库。其中 JSON 问答数据会自动转成“来源/问题/答案”的可检索文本。

后端启动时还会把 `data/finetune/sft_dataset_clean.json` 一并加入知识库。
该文件保持在微调目录中，不需要复制到 `data/processed`。每条
`instruction/input/output` 记录会被转换为带来源的“问题 + 答案”检索文本。

如果你把原始文本放在 `data/raw`，先同步清洗到 `data/processed`：

```bash
python utils/prepare_corpus.py
```

构建索引：

```bash
python embeddings/embed_utils.py build \
  --input data/processed \
  --output embeddings/faiss_index
```

测试检索：

```bash
python embeddings/embed_utils.py query "你的问题" --top_k 5
```

FAISS 索引会保存到 `embeddings/faiss_index`，构建完成后可以反复直接使用，不需要每次启动都重建。只有当你新增、删除或修改 `data/processed` 中的语料后，才需要重新运行 build 命令。

当前 `.env.example` 会在启动时检查知识库是否过期：

```env
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32
FAISS_THREADS=1
FINETUNE_DATASET_PATH=data/finetune/sft_dataset_clean.json
REBUILD_KNOWLEDGE_BASE_ON_STARTUP=true
KNOWLEDGE_BASE_SELF_TEST_QUERY=什么是体适能？
ISOLATE_RETRIEVAL_PROCESS=true
RETRIEVAL_FAILURE_FALLBACK=true
```

系统会对语料内容、embedding 模型和切块参数计算指纹。只有这些内容发生变化、
索引不存在或索引校验失败时才重建；未变化时直接复用持久化 FAISS 索引。
旧版本 `metadata.json` 没有 `format_version` 时会被视为不可用索引并强制重建，
即使 `REBUILD_KNOWLEDGE_BASE_ON_STARTUP=false` 也不会继续使用旧格式索引。

FAISS 和 SentenceTransformer 默认运行在独立的 CPU 子进程中，避免与加载到 CUDA
的 9B 模型共享同一组原生运行库。即使检索子进程发生 Windows 原生崩溃，FastAPI
主进程仍能继续运行；当 `RETRIEVAL_FAILURE_FALLBACK=true` 时，本次请求会退化为
不带知识库文档的模型回答，并在后端日志中记录检索异常。

`/health` 会返回 `knowledge_base_ready`、`knowledge_base_chunks` 和
`knowledge_base_error`。检索不可用但模型仍可回答时，健康状态为 `degraded`。
如果启动重建失败，当前进程会熔断知识库检索，不会在每个专业问题上重复加载同一个
损坏索引；修复语料或环境后重启后端即可重新尝试。

启动顺序为：检查或更新知识库、执行检索自检、加载基础模型、挂载 LoRA、开始接受
请求。设置 `REBUILD_KNOWLEDGE_BASE_ON_STARTUP=false` 后，不再自动更新过期索引，
但仍会校验并加载现有索引。

## 数据集格式

`data/dataset.json` 支持 JSON Array 或 JSON Lines。推荐格式：

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

划分数据集：

```bash
python utils/data_loader.py --input data/dataset.json --output_dir data
```

## LoRA/QLoRA 微调

安装训练依赖后，先登录 SwanLab 以便在浏览器里实时查看 loss、learning rate、训练步数等曲线：

```powershell
python -m pip install swanlab
swanlab login
```

当前训练参数已经写入 `models/train_lora.py`，默认使用：

- 模型：`models/qwen/Qwen3.5-9B`
- 数据：`data/finetune/sft_dataset_clean.json`
- 输出：`models/qwen/lora_adapter`
- LoRA：`r=16, alpha=32, dropout=0.05`
- 训练：`epochs=1, max_seq_length=1024, batch_size=1, gradient_accumulation_steps=8, bf16=True`
- 监控：默认启用 SwanLab，同时在控制台打印 step、loss、learning rate 和显存

正式训练前建议先预览数据和 prompt：

```powershell
python models/train_lora.py --dry_run --max_samples 3
```

开始 LoRA 微调：

```powershell
python models/train_lora.py
```

禁用 SwanLab，仅保留控制台进度：

```powershell
python models/train_lora.py --no_swanlab
```

自定义 SwanLab 项目名和实验名：

```powershell
python models/train_lora.py --swanlab_project local-domain-qa --swanlab_run_name qwen3.5-9b-run01
```

如果 `bf16` 报错，可以改用：

```powershell
python models/train_lora.py --no_bf16 --fp16
```

QLoRA：

```powershell
python models/train_lora.py --use_qlora
```

Windows 上 QLoRA 依赖 `bitsandbytes`，如果安装困难，优先使用 LoRA 或在 Linux/CUDA 环境训练。

## 测试

单元测试不需要真实模型权重或 FAISS 索引：

```bash
pytest
```

测试覆盖：

- `utils.data_loader`
- `backend.memory`
- `backend.rag`
- `backend.app`

## API

健康检查：

```bash
curl http://localhost:8000/health
```

直接测试当前配置的基础模型和 LoRA（不经过知识库检索）：

```bash
curl -X POST http://localhost:8000/api/model/test \
  -H "Content-Type: application/json" \
  -d '{"prompt":"测试微调后的模型","enable_thinking":false}'
```

问答接口：

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"什么是本地RAG？","top_k":5}'
```

流式问答接口：

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question":"请用Markdown列出本系统模块","top_k":5}'
```

前端默认使用流式接口，回答会逐步显示。输入框中直接按 `Enter` 发送，`Shift+Enter` 换行；助手回答支持常见 Markdown 渲染，包括标题、列表、粗体、行内代码和代码块。

前端输入框右侧有“深度思考”按钮。开启后，本次请求会发送 `enable_thinking=true`；关闭后会发送 `enable_thinking=false`。如果模型输出 `<think>...</think>`，前端会把思考内容放进背景色稍暗的圆角块，正常回答仍按普通 Markdown 展示。

模型输出长度由 `.env` 中的 `LOCAL_MODEL_MAX_NEW_TOKENS` 控制，默认是 `2048`。如果回答经常没说完，可以尝试调到 `3072` 或 `4096`；数值越大，单次生成耗时和显存占用越高。

如果模型支持 Qwen thinking 模板，默认通过 `LOCAL_MODEL_ENABLE_THINKING=false` 关闭思考模式，以减少首 token 等待时间。需要启用时改成：

```env
LOCAL_MODEL_ENABLE_THINKING=true
```

## GitHub 上传

模型权重、FAISS 索引、语料、`node_modules` 和构建产物已被 `.gitignore` 忽略。

```bash
git add .
git commit -m "Rebuild local domain QA system"
git push
```

GitHub Pages 只能预览前端静态页面；完整问答功能需要本地或公网后端。
