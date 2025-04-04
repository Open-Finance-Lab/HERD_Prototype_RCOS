import { useState } from 'react'
import "./Chat.css"
import Main from '../assets/Main'

function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (input.trim() === "") return;
    setMessages([...messages, input]);
    setInput("");
  };

  return (
    <>
      <Main>
        <div className="title">
          <h1>Heterogenous Experts with Routing Decisions</h1>
        </div>

        <div className="content">
          <div className="chatContent">
            <div className="chatBox">
              <div className="chatMessages">
                {messages.map((msg, index) => (
                  <div key={index} className="chatMessage">{msg}</div>
                ))}
              </div>
              <div className="chatInputSection">
                <input 
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Type your message..."
                  className="chatInput"
                />
                <button onClick={handleSend} className="sendButton">Send</button>
              </div>
            </div>
          </div>
        </div>
      </Main>
    </>
  )
}

export default Chat;
