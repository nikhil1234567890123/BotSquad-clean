import { useState, useRef, useEffect } from "react";
import { FiSend } from "react-icons/fi";
import { FaCommentDots, FaTimes, FaArrowDown } from "react-icons/fa";
import "./index.css";
  
type Message = {
  role: string;
  text: string;
  time: string;
  quickTags?: string[];
};

const App = () => {
  const [messages, setMessages] = useState<Message[]>([
    { 
      role: "bot", 
      text: "Hi! I'm PU Assistant. How can I help you today?", 
      time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      quickTags: ["Admission Inquiry", "General Inquiry"]
    },
  ]);
  const [input, setInput] = useState("");
  const [open, setOpen] = useState(false);
  const [isTyping, setIsTyping] = useState(false);

  const chatBodyRef = useRef<HTMLDivElement>(null);
  const [showScrollButton, setShowScrollButton] = useState(false);

  useEffect(() => {
    if (chatBodyRef.current) {
      chatBodyRef.current.scrollTo({ top: chatBodyRef.current.scrollHeight, behavior: "smooth" });
    }
  }, [messages, isTyping]);

  useEffect(() => {
    const handleScroll = () => {
      const chat = chatBodyRef.current;
      if (chat) {
        const atBottom = chat.scrollHeight - chat.scrollTop - chat.clientHeight < 10;
        setShowScrollButton(!atBottom);
      }
    };
    const chat = chatBodyRef.current;
    if (chat) {
      chat.addEventListener("scroll", handleScroll);
    }
    return () => {
      if (chat) {
        chat.removeEventListener("scroll", handleScroll);
      }
    };
  }, []);

  const handleSend = async (msgText?: string) => {
    const textToSend = msgText || input;
    if (!textToSend.trim()) return;

    const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute:'2-digit' });
    setMessages(prev => [...prev, { role: "user", text: textToSend, time: now }]);
    setInput("");
    setIsTyping(true);

    try {
      const res = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: textToSend }),
      });
      const data = await res.json();
      const botTime = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      setMessages(prev => [...prev, { role: "bot", text: data.reply, time: botTime }]);
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: "bot", text: "Please visit the admin office for more information.", time: now }]);
    }
    setIsTyping(false);
  };

  const handleSendTag = (tagText: string) => {
    handleSend(tagText);
  };

  return (
    <div className="relative">
      {!open && (
        <div className="chat-trigger" onClick={() => setOpen(true)}>
          <div className="chat-bubble">
            <div className="chat-bubble-title">Hello! How can I help?</div>
            <div className="chat-bubble-sub">Professor</div>
          </div>
          <button className="fixed-btn">
            <FaCommentDots size={20} />
          </button>
        </div>
      )}

      {open && (
        <div className="chat-panel">
          <header className="chat-header">
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div className="pu-logo-wrapper">
                <img src="pu-logo.png" alt="PU Logo" className="pu-logo" />
              </div>
              <h1>PU AI Assistant</h1>
            </div>
            <button onClick={() => setOpen(false)} style={{ color: "white", fontSize: "18px" }}>
              <FaTimes />
            </button>
          </header>

          <main className="chat-body" ref={chatBodyRef}>
            {messages.map((msg, i) => (
              <div key={i} className={`message-row ${msg.role}`}>
                <div className={`message-bubble ${msg.role}`}>
                  {msg.text}
                  {msg.quickTags && (
                    <div className="quick-tags-inline">
                      {msg.quickTags.map((tag, idx) => (
                        <span key={idx} className="quick-tag" onClick={() => handleSend(tag)}>
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                  <div className="message-time">{msg.time}</div>
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="message-row bot">
                <div className="message-bubble bot">
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}

            {showScrollButton && (
              <button
                className="scroll-to-bottom"
                onClick={() => {
                  chatBodyRef.current?.scrollTo({ top: chatBodyRef.current.scrollHeight, behavior: "smooth" });
                }}
              >
                <FaArrowDown />
              </button>
            )}
          </main>

          <div className="chat-footer">
            <div className="input-box">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
                placeholder="Type your message..."
              />
              <button onClick={() => handleSend()}>
                <FiSend />
              </button>
            </div>

            <div className="quick-tags">
              <span className="bg-red" onClick={() => handleSendTag("Admission Process")}>Admission Process</span>
              <span className="bg-blue" onClick={() => handleSendTag("Fee Structure")}>Fee Structure</span>
              <span className="bg-pink" onClick={() => handleSendTag("Course Details")}>Course Details</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
