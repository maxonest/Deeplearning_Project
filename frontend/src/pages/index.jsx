import React, { useEffect, useMemo, useState } from "react";
import {
  Brain,
  CircleAlert,
  LoaderCircle,
  RotateCcw,
  SendHorizontal,
  SlidersHorizontal,
} from "lucide-react";
import ChatMessage from "../components/ChatMessage.jsx";
import SourceList from "../components/SourceList.jsx";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const ASSISTANT_STREAM_KEY = "assistant-streaming";
const INITIAL_HEALTH = {
  status: "connecting",
  startup_phase: "connecting",
  startup_ready: false,
  startup_message: "正在连接后端",
  startup_error: null,
  model_loaded: false,
  use_lora_adapter: true,
  use_rag: true,
  knowledge_base_ready: false,
  knowledge_base_chunks: 0,
};

function getStatusPresentation(health, loading) {
  if (loading) {
    return { label: "生成中", detail: "模型正在组织回答", tone: "busy" };
  }
  if (health.status === "offline") {
    return { label: "等待后端", detail: "正在尝试连接本地服务", tone: "waiting" };
  }
  if (health.startup_phase === "knowledge_base") {
    return { label: "知识库校验中", detail: "正在读取并测试本地索引", tone: "waiting" };
  }
  if (health.startup_phase === "model") {
    return {
      label: "模型加载中",
      detail: health.use_lora_adapter === false ? "正在加载基础模型" : "正在加载基础模型与 LoRA",
      tone: "waiting",
    };
  }
  if (health.startup_phase === "failed") {
    return { label: "初始化失败", detail: health.startup_error || "请检查后端日志", tone: "error" };
  }
  if (health.status === "degraded") {
    return { label: "模型已就绪", detail: "知识库暂不可用，将使用模型直接回答", tone: "warning" };
  }
  if (health.startup_ready && health.use_rag === false) {
    return { label: "模型已就绪", detail: "RAG 已关闭，当前使用模型直接回答", tone: "ready" };
  }
  if (health.startup_ready) {
    return { label: "系统就绪", detail: "模型与知识库均可用", tone: "ready" };
  }
  return { label: "后端启动中", detail: health.startup_message || "请稍候", tone: "waiting" };
}

function parseSseChunk(buffer) {
  const events = [];
  let remaining = buffer;
  let boundary = remaining.indexOf("\n\n");

  while (boundary !== -1) {
    const block = remaining.slice(0, boundary);
    remaining = remaining.slice(boundary + 2);
    const lines = block.split("\n");
    const event = lines.find((line) => line.startsWith("event:"))?.slice(6).trim() || "message";
    const data = lines
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    events.push({ event, data });
    boundary = remaining.indexOf("\n\n");
  }

  return { events, remaining };
}

export default function IndexPage() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "正在连接本地问答服务。系统准备完成后，你就可以开始提问。",
    },
  ]);
  const [question, setQuestion] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [topK, setTopK] = useState(5);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [enableThinking, setEnableThinking] = useState(false);
  const [health, setHealth] = useState(INITIAL_HEALTH);

  const turnCount = useMemo(
    () => messages.filter((message) => message.role === "user").length,
    [messages],
  );
  const statusPresentation = getStatusPresentation(health, loading);
  const systemReady = health.startup_ready && health.startup_phase !== "failed";

  useEffect(() => {
    let active = true;
    let timerId;

    async function checkHealth() {
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(), 2500);
      try {
        const response = await fetch(`${API_BASE_URL}/health`, {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        if (active) {
          setHealth(payload);
        }
      } catch {
        if (active) {
          setHealth((current) => ({
            ...current,
            status: "offline",
            startup_phase: "connecting",
            startup_ready: false,
            startup_message: "正在连接后端",
          }));
        }
      } finally {
        window.clearTimeout(timeoutId);
        if (active) {
          timerId = window.setTimeout(checkHealth, 1500);
        }
      }
    }

    checkHealth();
    return () => {
      active = false;
      window.clearTimeout(timerId);
    };
  }, []);

  async function sendMessage(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading || !systemReady) return;

    setMessages((current) => [...current, { role: "user", content: trimmed }]);
    setQuestion("");
    setLoading(true);

    try {
      setMessages((current) => [
        ...current,
        { role: "assistant", content: "", streamKey: ASSISTANT_STREAM_KEY },
      ]);

      const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmed,
          session_id: sessionId,
          top_k: topK,
          enable_thinking: enableThinking,
        }),
      });

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}));
        throw new Error(errorPayload.detail || `HTTP ${response.status}`);
      }

      if (!response.body) {
        throw new Error("浏览器不支持流式响应");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parsed = parseSseChunk(buffer);
        buffer = parsed.remaining;

        for (const item of parsed.events) {
          const payload = item.data ? JSON.parse(item.data) : {};
          if (item.event === "meta") {
            setSessionId(payload.session_id);
            setDocuments(payload.documents || []);
          }
          if (item.event === "delta") {
            setMessages((current) =>
              current.map((message) =>
                message.streamKey === ASSISTANT_STREAM_KEY
                  ? { ...message, content: message.content + (payload.text || "") }
                  : message,
              ),
            );
          }
          if (item.event === "error") {
            throw new Error(payload.message || "流式生成失败");
          }
        }
      }

      setMessages((current) =>
        current.map((message) =>
          message.streamKey === ASSISTANT_STREAM_KEY ? { role: "assistant", content: message.content } : message,
        ),
      );
    } catch (error) {
      checkHealthAfterRequestFailure();
      setMessages((current) => [
        ...current.filter((message) => message.streamKey !== ASSISTANT_STREAM_KEY),
        {
          role: "assistant",
          content: `请求未完成：${error.message}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  }

  async function checkHealthAfterRequestFailure() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, { cache: "no-store" });
      if (response.ok) {
        setHealth(await response.json());
      }
    } catch {
      setHealth((current) => ({
        ...current,
        status: "offline",
        startup_phase: "connecting",
        startup_ready: false,
        startup_message: "后端连接已中断",
      }));
    }
  }

  async function resetConversation() {
    if (sessionId) {
      await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/clear`, { method: "POST" }).catch(() => {});
    }
    setSessionId(null);
    setDocuments([]);
    setMessages([
      {
        role: "assistant",
        content: systemReady ? "上下文已清空，可以开始新的专业问答。" : "上下文已清空，正在等待系统就绪。",
      },
    ]);
  }

  return (
    <main className="workspace">
      <aside className="sidePanel">
        <div className="brandBlock">
          <div className="brandIcon">
            <Brain size={22} />
          </div>
          <div>
            <h1>本地领域问答</h1>
            <p>RAG / LoRA / Memory</p>
          </div>
        </div>

        <section className="controlGroup">
          <div className="controlTitle">
            <SlidersHorizontal size={16} />
            <span>检索参数</span>
          </div>
          <label className="rangeLabel" htmlFor="topK">
            Top-K 文档
            <strong>{topK}</strong>
          </label>
          <input
            id="topK"
            min="1"
            max="10"
            value={topK}
            type="range"
            onChange={(event) => setTopK(Number(event.target.value))}
          />
        </section>

        <section className="statsGrid">
          <div>
            <span>对话轮次</span>
            <strong>{turnCount}</strong>
          </div>
          <div>
            <span>召回来源</span>
            <strong>{documents.length}</strong>
          </div>
        </section>

        <button className="ghostButton" type="button" onClick={resetConversation}>
          <RotateCcw size={16} />
          清空上下文
        </button>
      </aside>

      <section className="chatPanel">
        <div className="chatHeader">
          <div>
            <span>多轮对话工作台</span>
            <h2>今天想查询什么？</h2>
          </div>
          <div className={`statusCluster status-${statusPresentation.tone}`}>
            <div className="statusIcon" aria-hidden="true">
              {statusPresentation.tone === "error" ? (
                <CircleAlert size={16} />
              ) : statusPresentation.tone === "waiting" || statusPresentation.tone === "busy" ? (
                <LoaderCircle className="statusSpinner" size={16} />
              ) : (
                <span className="statusPulse" />
              )}
            </div>
            <div>
              <strong>{statusPresentation.label}</strong>
              <span>{statusPresentation.detail}</span>
            </div>
          </div>
        </div>

        {!systemReady && (
          <div className={`startupNotice startupNotice-${statusPresentation.tone}`} role="status">
            <div className="startupNoticeMark">
              {statusPresentation.tone === "error" ? <CircleAlert size={18} /> : <LoaderCircle size={18} />}
            </div>
            <div>
              <strong>{statusPresentation.label}</strong>
              <p>{statusPresentation.detail}。系统准备完成后，输入框会自动解锁。</p>
            </div>
          </div>
        )}

        <div className="messageList">
          {messages.map((message, index) => (
            <ChatMessage key={`${message.role}-${index}`} role={message.role} content={message.content} />
          ))}
        </div>

        <form className="composer" onSubmit={sendMessage}>
          <textarea
            value={question}
            placeholder={systemReady ? "输入你的领域问题..." : statusPresentation.label}
            rows={3}
            disabled={!systemReady || loading}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                sendMessage(event);
              }
            }}
          />
          <div className="composerActions">
            <button
              className={`thinkingButton ${enableThinking ? "thinkingButtonActive" : ""}`}
              type="button"
              disabled={loading || !systemReady}
              onClick={() => setEnableThinking((current) => !current)}
              title="切换深度思考"
            >
              <Brain size={16} />
              深度思考
            </button>
            <button
              className="sendButton"
              type="submit"
              disabled={loading || !systemReady || !question.trim()}
              title={systemReady ? "发送" : statusPresentation.label}
            >
              <SendHorizontal size={18} />
              发送
            </button>
          </div>
        </form>
      </section>

      <aside className="sourcePanel">
        <div className="panelTitle">
          <span>知识库召回</span>
          <strong>{health.use_rag === false ? "已关闭" : "FAISS"}</strong>
        </div>
        <SourceList documents={documents} />
      </aside>
    </main>
  );
}
