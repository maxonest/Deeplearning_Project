import React from "react";
import { Bot, UserRound } from "lucide-react";
import MarkdownMessage from "./MarkdownMessage.jsx";

export default function ChatMessage({ role, content }) {
  const isUser = role === "user";

  return (
    <article className={`message ${isUser ? "messageUser" : "messageAssistant"}`}>
      <div className="messageIcon" aria-hidden="true">
        {isUser ? <UserRound size={16} /> : <Bot size={16} />}
      </div>
      <div className="messageBody">
        <div className="messageRole">{isUser ? "用户" : "助手"}</div>
        <div className="messageContent">
          {isUser ? content : <MarkdownMessage content={content} />}
        </div>
      </div>
    </article>
  );
}
