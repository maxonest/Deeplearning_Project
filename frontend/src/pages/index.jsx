import React, { useMemo, useState } from "react";
import { Brain, RotateCcw, SendHorizontal, SlidersHorizontal } from "lucide-react";
import ChatMessage from "../components/ChatMessage.jsx";
import SourceList from "../components/SourceList.jsx";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const ASSISTANT_STREAM_KEY = "assistant-streaming";

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
      content: "本地专业知识问答系统已就绪。你可以输入领域问题，我会结合多轮上下文和 FAISS 知识库进行回答。",
    },
  ]);
  const [question, setQuestion] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [topK, setTopK] = useState(5);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [enableThinking, setEnableThinking] = useState(false);

  const turnCount = useMemo(
    () => messages.filter((message) => message.role === "user").length,
    [messages],
  );

  async function sendMessage(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || loading) return;

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
        throw new Error(`HTTP ${response.status}`);
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
      setMessages((current) => [
        ...current.filter((message) => message.streamKey !== ASSISTANT_STREAM_KEY),
        {
          role: "assistant",
          content: `请求失败：${error.message}。请确认后端服务已启动。`,
        },
      ]);
    } finally {
      setLoading(false);
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
        content: "上下文已清空，可以开始新的专业问答。",
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
          <div className={`statusDot ${loading ? "statusBusy" : ""}`}>{loading ? "生成中" : "在线"}</div>
        </div>

        <div className="messageList">
          {messages.map((message, index) => (
            <ChatMessage key={`${message.role}-${index}`} role={message.role} content={message.content} />
          ))}
        </div>

        <form className="composer" onSubmit={sendMessage}>
          <textarea
            value={question}
            placeholder="输入你的领域问题..."
            rows={3}
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
              disabled={loading}
              onClick={() => setEnableThinking((current) => !current)}
              title="切换深度思考"
            >
              <Brain size={16} />
              深度思考
            </button>
            <button className="sendButton" type="submit" disabled={loading || !question.trim()} title="发送">
              <SendHorizontal size={18} />
              发送
            </button>
          </div>
        </form>
      </section>

      <aside className="sourcePanel">
        <div className="panelTitle">
          <span>知识库召回</span>
          <strong>FAISS</strong>
        </div>
        <SourceList documents={documents} />
      </aside>
    </main>
  );
}
