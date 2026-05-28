# Local Domain QA System

本项目是一套本地专业知识领域问答系统，包含数据集处理、FAISS 向量知识库、RAG 后端、本地大模型推理、LoRA/QLoRA 微调、React 前端、Windows 启动器和自动化测试。

## 功能模块

- 数据集构建：JSON/JSONL 数据加载、字段标准化、训练/验证/测试划分。
- 知识库构建：读取 `txt/md/csv` 语料，切块后用 sentence-transformers 生成 embedding，并保存 FAISS 索引。
- 本地 RAG：FastAPI 接收问题，管理多轮会话，检索 top-k 文档，拼接上下文后调用本地模型。
- 本地模型：支持通过 Transformers 加载 `Qwen3.5-9B` 等本地 causal LM。
- 微调训练：基于 Transformers Trainer + PEFT 进行 LoRA/QLoRA SFT。
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
LOCAL_MODEL_MAX_NEW_TOKENS=2048
LOCAL_MODEL_TEMPERATURE=0.2
LOCAL_MODEL_TOP_P=0.9
LOCAL_MODEL_ENABLE_THINKING=false
```

### 3. 单独测试模型

```powershell
python models/run_local_model.py --prompt "你好，请介绍一下你自己"
```

如果能正常输出，再启动系统。

### 4. 启动前后端

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

## 构建 FAISS 知识库

把语料放入：

```text
data/processed/
```

支持 `.txt`、`.md`、`.csv`。

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

LoRA：

```bash
python models/train_lora.py \
  --model_name_or_path models/qwen/Qwen3.5-9B \
  --dataset_path data/dataset.json \
  --output_dir models/qwen/lora_adapter \
  --local_files_only \
  --bf16
```

QLoRA：

```bash
python models/train_lora.py \
  --model_name_or_path models/qwen/Qwen3.5-9B \
  --dataset_path data/dataset.json \
  --output_dir models/qwen/qlora_adapter \
  --local_files_only \
  --use_qlora \
  --bf16
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
