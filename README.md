# Local Domain QA System

本项目是一个本地专业知识领域问答系统骨架，覆盖数据集构建、本地向量知识库、LoRA/QLoRA 微调、本地 RAG、多轮对话记忆、FastAPI 后端和 React 前端。

## 技术栈

- Python 3.10+
- FastAPI 后端 API
- React 18 + Vite 前端
- FAISS 本地向量检索
- sentence-transformers 文本向量化
- Transformers + PEFT + TRL 进行 LoRA/QLoRA 微调
- 推荐 GPU: RTX 4090 24GB 或更高

## 目录结构

```text
project_root/
├─ data/
│  ├─ raw/              # 原始语料文件
│  ├─ processed/        # 清洗后的文本
│  └─ dataset.json      # 生成的 JSON 数据集
├─ embeddings/
│  ├─ faiss_index/      # FAISS 索引文件
│  └─ embed_utils.py    # Embedding 生成脚本
├─ models/
│  ├─ qwen/             # Qwen 模型权重
│  ├─ deepseek/         # DeepSeek 模型权重
│  └─ train_lora.py     # LoRA/QLoRA 微调脚本
├─ backend/
│  ├─ app.py            # FastAPI 后端主入口
│  ├─ rag.py            # RAG 检索逻辑
│  └─ memory.py         # 多轮对话上下文管理
├─ frontend/
│  ├─ src/
│  │  ├─ components/    # React UI 组件
│  │  └─ pages/         # 页面文件
│  └─ package.json
├─ utils/
│  ├─ data_loader.py    # 数据加载与处理
│  ├─ logging_utils.py  # 日志工具
│  └─ config.py         # 全局配置
├─ requirements.txt
└─ run.sh
```

## 快速开始

```bash
# 创建并激活 Conda 环境
conda create -n local-domain-qa python=3.10 -y
conda activate local-domain-qa

# 安装 Python 依赖
pip install -r requirements.txt

# 安装前端依赖
cd frontend
npm install
cd ..

# 启动前后端服务
./run.sh
```

也可以使用项目内置的 Conda 环境文件：

```bash
conda env create -f environment.yml
conda activate local-domain-qa
cd frontend && npm install && cd ..
./run.sh
```

后端默认运行在 `http://localhost:8000`，前端默认运行在 `http://localhost:5173`。

## GitHub 上传

上传前建议只提交源码、配置和示例数据，不提交 `node_modules`、`dist`、模型权重、FAISS 索引、原始语料和 Python 缓存文件。项目已提供 `.gitignore` 过滤这些生成文件。

```bash
git init
git add .
git commit -m "Initial local domain QA system"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

如果你已经在 GitHub 创建了空仓库，并安装了 GitHub CLI，也可以用：

```bash
gh repo create <your-repo> --public --source=. --remote=origin --push
```

## 前端页面预览

GitHub 仓库页面本身只能预览 README 和代码，不能直接运行 React 前端。要在线预览前端页面，可以使用 GitHub Pages 部署 `frontend` 的 Vite 构建产物。

本地构建：

```bash
cd frontend
npm install
npm run build
```

然后在 GitHub 仓库中进入 `Settings -> Pages`，选择 GitHub Actions 或部署分支，将 `frontend/dist` 发布为静态站点。

注意：当前前端会调用本地后端 `http://localhost:8000`。部署到 GitHub Pages 后，页面可以打开，但问答接口只有在你的本机同时运行后端、或你把后端部署到公网并设置 `VITE_API_BASE_URL` 时才可用。

## 数据集格式

`data/dataset.json` 使用 JSON Lines 或 JSON Array 均可。推荐字段：

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

## 构建向量知识库

```bash
python embeddings/embed_utils.py \
  --input data/processed \
  --output embeddings/faiss_index
```

## LoRA/QLoRA 微调

```bash
python models/train_lora.py \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset_path data/dataset.json \
  --output_dir models/qwen/lora_adapter \
  --use_qlora
```

## 后续开发建议

- 将 `backend/rag.py` 中的 `LocalLLMClient` 替换为 vLLM、Ollama、llama.cpp 或 Transformers 本地推理服务。
- 根据专业领域补充 `data/raw`，再清洗到 `data/processed`。
- 微调前先验证样本质量、回答风格和拒答边界。
- 生产部署时增加鉴权、限流、审计日志和模型输出安全过滤。
