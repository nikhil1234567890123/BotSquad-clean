import { useState } from "react";
import { FiSend } from "react-icons/fi";
import {
  FaThumbsUp,
  FaThumbsDown,
  FaCommentDots,
  FaTimes,
} from "react-icons/fa";
import "./index.css";

const App = () => {
  const [messages, setMessages] = useState([
    {
      role: "bot",
      text: "Hi! I'm PU Admission Assistant. How can I help you today?",
    },
  ]);
  const [input, setInput] = useState("");
  const [open, setOpen] = useState(false); // toggle chat panel

  const handleSend = () => {
    if (!input.trim()) return;
    setMessages([...messages, { role: "user", text: input }]);
    setInput("");
  };

  return (
    <div className="relative">
      {/* Floating Chat Button */}
      {!open && (
        <button onClick={() => setOpen(true)} className="fixed-btn">
          <FaCommentDots size={20} />
        </button>
      )}

      {/* Chat Panel */}
      {open && (
        <div className="chat-panel">
          {/* Header */}
          <header className="chat-header">
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div className="pu-logo-wrapper">
                <img src="pu-logo.png" alt="PU Logo" className="pu-logo" />
              </div>
              <h1>PU AI Assistant</h1>
            </div>
            <button
              onClick={() => setOpen(false)}
              style={{
                background: "none",
                border: "none",
                color: "white",
                fontSize: "18px",
              }}
            >
              <FaTimes />
            </button>
          </header>

          {/* Chat Body */}
          <main className="chat-body">
            {messages.map((msg, i) => (
              <div key={i} className={`message-row ${msg.role}`}>
                <div className={`message-bubble ${msg.role}`}>
                  {msg.text}
                  {msg.role === "bot" && i === messages.length - 1 && (
                    <div className="feedback-buttons">
                      <button className="text-green-600">
                        <FaThumbsUp />
                      </button>
                      <button className="text-red-500">
                        <FaThumbsDown />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </main>

          {/* Input Area */}
          <div className="chat-footer">
            <div className="input-box">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleSend();
                }}
                placeholder="Type your message..."
              />
              <button onClick={handleSend}>
                <FiSend />
              </button>
            </div>

            {/* Quick Tags */}
            <div className="quick-tags">
              <span className="bg-red">Admission Process</span>
              <span className="bg-blue">Fee Structure</span>
              <span className="bg-pink">Course Details</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
