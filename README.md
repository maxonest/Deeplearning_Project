# Local Domain QA System

本项目是一套本地专业知识领域问答系统，包含数据集处理、FAISS 向量知识库、RAG 后端、本地大模型推理、LoRA/QLoRA 微调、React 前端、Windows 启动器和自动化测试。

## 功能模块

- 数据集构建：JSON/JSONL 数据加载、字段标准化、训练/验证/测试划分。
- 知识库构建：读取 `txt/md/csv` 语料，切块后用 Transformers mean pooling 生成 embedding，并保存 FAISS 索引。
- 本地 RAG：FastAPI 接收问题，管理多轮会话，检索 top-k 文档，拼接上下文后调用本地模型。
- 本地模型：支持通过 Transformers 加载 `Qwen3.5-9B` 等本地 causal LM，并可通过 PEFT 挂载 LoRA adapter。
- 专家角色：训练、直接推理和 RAG 共用统一的运动健康垂直领域专家系统提示词。
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
Windows 下固定使用 `numpy==1.26.4`，避免 pip/conda 混装后出现
`_multiarray_umath` DLL 或 NumPy 二进制 ABI 冲突。
RAG 不再导入 `sentence-transformers` Python 包，而是直接用已有的
`transformers + torch` 加载 MiniLM 并执行标准 attention-mask mean pooling，从而
完全绕开 `datasets/aiohttp` 和损坏的 Windows 证书库导入链。

### 2. 配置模型路径

```powershell
copy .env.example .env
```

确认 `.env` 中：

```env
USE_LOCAL_MODEL=true
LOCAL_MODEL_PATH=models/qwen/Qwen3.5-9B
USE_LORA_ADAPTER=true
LOCAL_LORA_ADAPTER_PATH=models/qwen/lora_adapter
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32
LOCAL_MODEL_MAX_NEW_TOKENS=2048
LOCAL_MODEL_TEMPERATURE=0.2
LOCAL_MODEL_TOP_P=0.9
LOCAL_MODEL_ENABLE_THINKING=false
```

### 3. 单独测试模型

```powershell
python models/run_local_model.py --prompt "你好，请介绍一下你自己"
```

当 `USE_LORA_ADAPTER=true` 且配置了 `LOCAL_LORA_ADAPTER_PATH` 时，上述命令会先加载
`LOCAL_MODEL_PATH` 中的基础模型，再挂载 LoRA adapter。设置
`USE_LORA_ADAPTER=false` 后只加载基础模型，无需删除 adapter 文件或路径。也可以在
命令行中显式指定：

```powershell
python models/run_local_model.py --model_path models/qwen/Qwen3.5-9B --lora_adapter_path models/qwen/lora_adapter --prompt "你好，请介绍一下你自己"
```

从其他机器复制 adapter 时，目录中至少需要包含
`adapter_config.json` 和 `adapter_model.safetensors`（或 `adapter_model.bin`），
并且基础模型必须与训练 LoRA 时使用的模型兼容。不使用 LoRA 时，在 `.env` 中设置
`USE_LORA_ADAPTER=false`。

如果能正常输出，再启动系统。

### 4. 启动后端测试模型

首次部署或更新语料后，先构建并验证本地知识库：

```powershell
python embeddings/embed_utils.py build
python embeddings/embed_utils.py status
python embeddings/embed_utils.py query "什么是体适能？" --top_k 3
```

三条命令均成功后再启动系统。以后只要语料、embedding 模型和切块参数没有变化，
无需再次构建。

只启动 FastAPI，不启动前端：

```powershell
python start_windows.py --backend-only
```

当 `USE_LOCAL_MODEL=true` 时，FastAPI 启动后会在后台加载基础模型；仅当
`USE_LORA_ADAPTER=true` 时挂载 LoRA。
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
`model_loaded` 应为 `true`。启用 LoRA 时，`use_lora_adapter` 应为 `true`，
`local_lora_adapter_path` 应指向 `models/qwen/lora_adapter`。也可以访问
`http://localhost:8000/health` 确认模型是否已经加载。

FastAPI 启动后会立即开放 `/health`，知识库和模型在后台继续初始化。前端每
1.5 秒轮询健康状态，并依次显示“知识库校验中”“模型加载中”“系统就绪”。
初始化完成前输入框和发送按钮保持禁用，完成后自动解锁，无需刷新页面。

### 运动健康专家角色

系统角色统一定义在 `utils/prompts.py` 的 `SPORTS_HEALTH_SYSTEM_PROMPT` 中，并同时
用于直接模型推理、RAG 提示词、SFT 数据格式化和 LoRA 训练。修改专家定位时只需要
调整这一处。现有 LoRA 无需重新训练即可在推理阶段使用新角色；以后重新训练时，
训练模板也会自动使用同一份角色提示词。

如果基础模型或 LoRA 路径错误、adapter 文件不完整，`/health` 会返回
`startup_phase=failed` 和具体的 `startup_error`，聊天接口保持不可用。不要使用多个
Uvicorn worker，否则每个 worker 都会各自加载一份模型并占用显存。

Windows 后端启动日志会同时写入 `logs/backend_startup.log`。如果后端进程出现
`3221225725 (0xC00000FD)`，这是 Windows 原生栈溢出，而不是普通 Python 异常。
请提供日志中最后一条 `Model load stage` 以及其后的 faulthandler 调用栈，用于判断
崩溃发生在依赖导入、tokenizer、基础模型还是 LoRA adapter 加载阶段。

`start_windows.py` 不会在终端实时输出后端日志；Uvicorn、知识库和模型加载日志只写入
`logs/backend_startup.log`。终端仅显示服务地址、日志路径和进程退出摘要。
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

默认知识库构建命令还会把 `data/finetune/sft_dataset_clean.json` 一并加入知识库。
该文件保持在微调目录中，不需要复制到 `data/processed`。每条
`instruction/input/output` 记录会被转换为带来源的“问题 + 答案”检索文本。

如果你把原始文本放在 `data/raw`，先同步清洗到 `data/processed`：

```bash
python utils/prepare_corpus.py
```

默认使用轻量多语言模型
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`。文档和用户问题采用
完全相同的 Transformers mean pooling 编码流程，并统一归一化后写入或查询 FAISS。
这里的 `sentence-transformers/` 是 Hugging Face 模型仓库命名空间，不表示程序会
导入同名 Python 包。
第一次运行需要联网下载模型；以后会直接使用 Hugging Face 本地缓存。

首次构建或语料更新后，单独执行知识库构建命令。该命令默认同时读取
`data/processed/` 和 `data/finetune/sft_dataset_clean.json`：

```powershell
python embeddings/embed_utils.py build
```

从 Qwen3-Embedding-4B 配置切换回来后，旧索引不兼容，需要执行一次上述命令重新构建。

默认在 CPU 上构建，不占用对话模型显存。数据量较大时可显式使用
`--device cuda` 加速。

编码时会显示单行进度条。索引保存在 `embeddings/faiss_index/index.faiss`，文本和
模型信息保存在同目录的 `metadata.json`。

需要指定语料源时，可重复使用 `--input`：

```bash
python embeddings/embed_utils.py build \
  --input data/processed \
  --input data/finetune/sft_dataset_clean.json
```

检查本地索引状态：

```bash
python embeddings/embed_utils.py status
```

测试检索：

```bash
python embeddings/embed_utils.py query "你的问题" --top_k 5
```

FAISS 索引持久化在 `embeddings/faiss_index`。构建成功后，后端启动只读取、校验并
执行一次检索自检，不再编码或修改索引。只有语料、embedding 模型或切块参数变化时，
才需要重新运行 `build`。

推荐配置：

```env
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32
FAISS_THREADS=1
FINETUNE_DATASET_PATH=data/finetune/sft_dataset_clean.json
KNOWLEDGE_BASE_SELF_TEST_QUERY=什么是体适能？
RETRIEVAL_FAILURE_FALLBACK=true
```

RAG 使用普通的进程内调用链：Transformers 编码问题、FAISS 检索 top-k、
将命中文档拼接进提示词，再交给本地模型回答。索引构建和查询必须使用同一个
Embedding 模型。

`/health` 会返回 `knowledge_base_ready`、`knowledge_base_chunks` 和
`knowledge_base_error`。默认 `RETRIEVAL_FAILURE_FALLBACK=true`，因此索引缺失或损坏
时后端仍会加载模型并进入降级状态，普通聊天仍可使用；重新运行 `build` 并重启后端
即可恢复 RAG。

稳定启动顺序为：读取本地索引、完整性校验、检索自检、加载基础模型、按开关决定
是否挂载 LoRA、后端就绪、前端自动解锁。

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
