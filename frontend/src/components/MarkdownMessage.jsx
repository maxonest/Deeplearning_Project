import React from "react";

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderInline(value) {
  return escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function markdownToHtml(markdown) {
  const lines = markdown.split(/\r?\n/);
  const html = [];
  let inCodeBlock = false;
  let codeLines = [];
  let inList = false;

  function closeList() {
    if (inList) {
      html.push("</ul>");
      inList = false;
    }
  }

  function closeCodeBlock() {
    if (inCodeBlock) {
      html.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      codeLines = [];
      inCodeBlock = false;
    }
  }

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      if (inCodeBlock) {
        closeCodeBlock();
      } else {
        closeList();
        inCodeBlock = true;
        codeLines = [];
      }
      continue;
    }

    if (inCodeBlock) {
      codeLines.push(line);
      continue;
    }

    if (!line.trim()) {
      closeList();
      html.push("<br />");
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      closeList();
      const level = heading[1].length;
      html.push(`<h${level}>${renderInline(heading[2])}</h${level}>`);
      continue;
    }

    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    if (bullet) {
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${renderInline(bullet[1])}</li>`);
      continue;
    }

    closeList();
    html.push(`<p>${renderInline(line)}</p>`);
  }

  closeCodeBlock();
  closeList();
  return html.join("");
}

function splitThinkSegments(content) {
  const segments = [];
  let cursor = 0;
  const lower = content.toLowerCase();

  while (cursor < content.length) {
    const start = lower.indexOf("<think>", cursor);
    if (start === -1) {
      segments.push({ type: "answer", text: content.slice(cursor) });
      break;
    }

    if (start > cursor) {
      segments.push({ type: "answer", text: content.slice(cursor, start) });
    }

    const thinkStart = start + "<think>".length;
    const end = lower.indexOf("</think>", thinkStart);
    if (end === -1) {
      segments.push({ type: "think", text: content.slice(thinkStart), open: true });
      break;
    }

    segments.push({ type: "think", text: content.slice(thinkStart, end), open: false });
    cursor = end + "</think>".length;
  }

  return segments.filter((segment) => segment.text.length > 0);
}

export default function MarkdownMessage({ content }) {
  return (
    <div className="markdownContent">
      {splitThinkSegments(content || "").map((segment, index) => (
        <section className={segment.type === "think" ? "thinkBlock" : "answerBlock"} key={index}>
          {segment.type === "think" && <div className="thinkLabel">深度思考{segment.open ? "中" : ""}</div>}
          <div dangerouslySetInnerHTML={{ __html: markdownToHtml(segment.text) }} />
        </section>
      ))}
    </div>
  );
}
