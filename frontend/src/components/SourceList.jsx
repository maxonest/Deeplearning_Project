import React from "react";
import { FileText } from "lucide-react";

export default function SourceList({ documents }) {
  if (!documents?.length) {
    return <div className="emptySources">暂无检索来源</div>;
  }

  return (
    <div className="sourceList">
      {documents.map((doc, index) => (
        <section className="sourceItem" key={`${doc.source}-${doc.id ?? index}`}>
          <div className="sourceHeader">
            <FileText size={14} />
            <span>文档 {index + 1}</span>
            {typeof doc.score === "number" && <strong>{doc.score.toFixed(3)}</strong>}
          </div>
          <p>{doc.text}</p>
          <small>{doc.source}</small>
        </section>
      ))}
    </div>
  );
}
