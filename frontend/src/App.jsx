
import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const INITIAL_MESSAGE = {
  role: "assistant",
  content:
    "Hello! I'm Mini GPT, your local AI assistant. What would you like to explore today?",
};

export default function App() {
  // 1. Declare all state and refs before using them in effects.
  const [messages, setMessages] = useState([INITIAL_MESSAGE]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const bottomRef = useRef(null);
  const conversationRef = useRef(null);

  // 2. Scroll to the latest message as the response streams.
  useEffect(() => {
    const container = conversationRef.current;
    const bottom = bottomRef.current;

    if (!container || !bottom) return;

    const distanceFromBottom =
      container.scrollHeight -
      container.scrollTop -
      container.clientHeight;

    // Follow the conversation while already near its bottom.
    if (distanceFromBottom < 180 || loading) {
      bottom.scrollIntoView({
        behavior: "auto",
        block: "end",
      });
    }
  }, [messages, loading]);

  async function sendMessage(event) {
    event.preventDefault();

    const message = input.trim();

    if (!message || loading) return;

    const history = [
      ...messages,
      { role: "user", content: message },
    ];

    setMessages([
      ...history,
      { role: "assistant", content: "" },
    ]);

    setInput("");
    setError("");
    setLoading(true);

    try {
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({
          messages: history.map(({ role, content }) => ({
            role,
            content,
          })),
        }),
      });

      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.detail || `Request failed (${response.status})`);
      }

      if (!response.body) {
        throw new Error("The server did not provide a response stream.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = "";
      let finished = false;

      function appendToken(token) {
        setMessages((current) => {
          const updated = [...current];
          const lastIndex = updated.length - 1;

          if (
            lastIndex < 0 ||
            updated[lastIndex].role !== "assistant"
          ) {
            return current;
          }

          updated[lastIndex] = {
            ...updated[lastIndex],
            content: updated[lastIndex].content + token,
          };

          return updated;
        });
      }

      function processEvent(eventText) {
        for (const line of eventText.split(/\r?\n/)) {
          if (!line.startsWith("data:")) continue;

          const value = line.slice(5).trimStart();

          if (value === "[DONE]") {
            finished = true;
            continue;
          }

          if (!value) continue;

          const data = JSON.parse(value);

          if (data.error) {
            throw new Error(data.error);
          }

          if (typeof data.token === "string") {
            appendToken(data.token);
          }
        }
      }

      while (!finished) {
        const { value, done } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const events = buffer.split(/\r?\n\r?\n/);
        buffer = events.pop() ?? "";

        for (const eventText of events) {
          processEvent(eventText);

          if (finished) break;
        }
      }

      buffer += decoder.decode();

      if (buffer.trim()) {
        processEvent(buffer);
      }

      if (!finished) {
        throw new Error("The response stream ended unexpectedly.");
      }
    } catch (err) {
      setError(
        `${err.message}. Check that FastAPI and Ollama are running.`
      );
    } finally {
      setLoading(false);
    }
  }

  function newChat() {
    if (loading) return;

    setMessages([INITIAL_MESSAGE]);
    setInput("");
    setError("");
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">✳</div>
          <span>
            mini<span className="brand-accent">GPT</span>
          </span>
        </div>

        <button
          className="new-chat"
          onClick={newChat}
          disabled={loading}
        >
          <span>＋</span> New chat
        </button>

        <div className="sidebar-label">YOUR MODEL</div>

        <div className="model-card">
          <div className="model-dot" />
          <div>
            <strong>Qwen 2.5</strong>
            <small>1.5B · Ollama local</small>
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="online-dot" />
          Runs on your machine
        </div>
      </aside>

      <main className="chat-layout">
        <header className="topbar">
          <div>
            <strong>Mini GPT</strong>
            <span className="topbar-subtitle">
              Personal AI assistant
            </span>
          </div>

          <div className="local-badge">
            <span className="online-dot" /> LOCAL MODEL
          </div>
        </header>

        <section className="conversation" ref={conversationRef}>
          <div className="messages">
            {messages.map((message, index) => (
              <article
                className={`message ${message.role}`}
                key={`${index}-${message.role}`}
              >
                <div className="avatar">
                  {message.role === "assistant" ? "✳" : "Y"}
                </div>

                <div className="message-body">
                  <div className="message-author">
                    {message.role === "assistant" ? "Mini GPT" : "You"}
                  </div>

                  <div className="message-text">
                    {message.content}
                    {loading &&
                      index === messages.length - 1 &&
                      message.role === "assistant" && (
                        <span className="stream-cursor">▍</span>
                      )}
                  </div>
                </div>
              </article>
            ))}

            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}

            <div ref={bottomRef} />
          </div>
        </section>

        <div className="composer-area">
          <form className="composer" onSubmit={sendMessage}>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !event.shiftKey &&
                  !event.nativeEvent.isComposing
                ) {
                  event.preventDefault();
                  event.currentTarget.form.requestSubmit();
                }
              }}
              placeholder="Message Mini GPT..."
              rows={1}
              maxLength={4000}
              disabled={loading}
            />

            <button
              className="send-button"
              type="submit"
              disabled={loading || !input.trim()}
              aria-label="Send message"
            >
              {loading ? "…" : "↑"}
            </button>
          </form>

          <p className="disclaimer">
            Responses stream live from your local Ollama model.
          </p>
        </div>
      </main>
    </div>
  );
}
