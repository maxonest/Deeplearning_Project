import React, { useMemo, useState } from "react";
import { Brain, RotateCcw, SendHorizontal, SlidersHorizontal } from "lucide-react";
import ChatMessage from "../components/ChatMessage.jsx";
import SourceList from "../components/SourceList.jsx";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

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
      const response = await fetch(`${API_BASE_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmed,
          session_id: sessionId,
          top_k: topK,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const payload = await response.json();
      setSessionId(payload.session_id);
      setDocuments(payload.documents || []);
      setMessages((current) => [...current, { role: "assistant", content: payload.answer }]);
    } catch (error) {
      setMessages((current) => [
        ...current,
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
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                sendMessage(event);
              }
            }}
          />
          <button type="submit" disabled={loading || !question.trim()} title="发送">
            <SendHorizontal size={18} />
            发送
          </button>
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
