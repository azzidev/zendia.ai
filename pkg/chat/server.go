package chat

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/azzidev/zendia.ai/pkg/dialogue"
	"github.com/azzidev/zendia.ai/pkg/model"
	"github.com/azzidev/zendia.ai/pkg/tools"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

type Server struct {
	Model *model.LLM
	Tools *tools.Registry
	port  int
	mu    sync.Mutex
}

type ChatMessage struct {
	Type    string `json:"type"`
	Content string `json:"content"`
	From    string `json:"from"`
	Time    string `json:"time"`
}

func NewServer(llm *model.LLM, port int) *Server {
	return &Server{Model: llm, Tools: tools.NewRegistry(), port: port}
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("ws error:", err)
		return
	}
	defer conn.Close()

	// manda info do modelo
	info := ChatMessage{
		Type:    "info",
		Content: s.Model.String(),
		From:    "system",
		Time:    time.Now().Format("15:04:05"),
	}
	sendJSON(conn, info)

	for {
		_, msg, err := conn.ReadMessage()
		if err != nil {
			break
		}

		var incoming ChatMessage
		if err := json.Unmarshal(msg, &incoming); err != nil {
			continue
		}

		// tenta tool calling primeiro
		var response string
		if toolResult, ok := s.Tools.TryExecute(incoming.Content); ok {
			response = toolResult
		} else {
			// gera resposta com o modelo
			s.mu.Lock()
			prompt := dialogue.UserToken + " " + incoming.Content + " " + dialogue.BotToken
			generated := s.Model.Generate(prompt, 50, 0.7)
			response = dialogue.ParseBotResponse(generated)
			s.mu.Unlock()
		}

		if response == "" {
			response = "hmm, não sei o que dizer sobre isso ainda."
		}

		reply := ChatMessage{
			Type:    "message",
			Content: response,
			From:    "bot",
			Time:    time.Now().Format("15:04:05"),
		}
		sendJSON(conn, reply)
	}
}

func sendJSON(conn *websocket.Conn, v interface{}) {
	data, _ := json.Marshal(v)
	conn.WriteMessage(websocket.TextMessage, data)
}

func (s *Server) Start() {
	mux := http.NewServeMux()
	mux.HandleFunc("/ws", s.handleChat)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, chatHTML)
	})

	addr := fmt.Sprintf(":%d", s.port)
	fmt.Printf("💬 Chat rodando em http://localhost%s\n", addr)
	http.ListenAndServe(addr, mux)
}

var chatHTML = `<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Zendia.AI — Chat</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

  :root {
    --bg: #08080d;
    --bg2: #0e0e16;
    --bg3: #14142a;
    --border: #1e1e32;
    --text: #e8e8f0;
    --text2: #8888a8;
    --text3: #555570;
    --accent: #6c5ce7;
    --accent2: #a29bfe;
    --green: #00cec9;
    --user-bg: #1a1a3e;
    --bot-bg: #12121e;
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
  }

  #header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 24px;
    background: var(--bg2);
    border-bottom: 1px solid var(--border);
  }

  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .logo-icon {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, var(--accent), var(--green));
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
  }

  .logo h1 {
    font-size: 16px;
    font-weight: 600;
    color: var(--text);
  }

  .logo h1 span { color: var(--accent2); }

  #status {
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: var(--text3);
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--green);
    animation: blink 2s ease-in-out infinite;
  }

  @keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  #messages {
    flex: 1;
    overflow-y: auto;
    padding: 20px 24px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  #messages::-webkit-scrollbar { width: 4px; }
  #messages::-webkit-scrollbar-track { background: transparent; }
  #messages::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  .msg {
    max-width: 75%;
    padding: 12px 16px;
    border-radius: 14px;
    font-size: 14px;
    line-height: 1.5;
    animation: fadeIn 0.2s ease;
  }

  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .msg-user {
    align-self: flex-end;
    background: var(--user-bg);
    border: 1px solid rgba(108, 92, 231, 0.2);
    border-bottom-right-radius: 4px;
  }

  .msg-bot {
    align-self: flex-start;
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }

  .msg-system {
    align-self: center;
    background: transparent;
    color: var(--text3);
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    padding: 6px 12px;
    border: 1px solid var(--border);
    border-radius: 20px;
  }

  .msg-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
    font-size: 11px;
    color: var(--text3);
  }

  .msg-header .name {
    font-weight: 600;
    color: var(--accent2);
  }

  .msg-bot .msg-header .name {
    color: var(--green);
  }

  .typing {
    align-self: flex-start;
    padding: 12px 16px;
    background: var(--bot-bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    border-bottom-left-radius: 4px;
    display: none;
  }

  .typing-dots {
    display: flex;
    gap: 4px;
  }

  .typing-dots span {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--text3);
    animation: typingDot 1.2s ease-in-out infinite;
  }

  .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
  .typing-dots span:nth-child(3) { animation-delay: 0.4s; }

  @keyframes typingDot {
    0%, 100% { opacity: 0.3; transform: scale(0.8); }
    50% { opacity: 1; transform: scale(1); }
  }

  #input-area {
    padding: 16px 24px;
    background: var(--bg2);
    border-top: 1px solid var(--border);
    display: flex;
    gap: 10px;
  }

  #input {
    flex: 1;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 12px 16px;
    color: var(--text);
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    outline: none;
    transition: border-color 0.2s;
  }

  #input:focus {
    border-color: var(--accent);
  }

  #input::placeholder {
    color: var(--text3);
  }

  #send {
    background: linear-gradient(135deg, var(--accent), #5a4bd1);
    border: none;
    border-radius: 12px;
    padding: 12px 20px;
    color: white;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.15s, transform 0.1s;
  }

  #send:hover { opacity: 0.9; }
  #send:active { transform: scale(0.97); }
</style>
</head>
<body>

<div id="header">
  <div class="logo">
    <div class="logo-icon">🧠</div>
    <h1>zendia<span>.ai</span></h1>
  </div>
  <div id="status">
    <div class="dot"></div>
    <span id="status-text">conectando...</span>
  </div>
</div>

<div id="messages">
  <div class="msg msg-system">iniciando conexão...</div>
</div>

<div class="typing" id="typing">
  <div class="typing-dots">
    <span></span><span></span><span></span>
  </div>
</div>

<div id="input-area">
  <input type="text" id="input" placeholder="Digite sua mensagem..." autocomplete="off">
  <button id="send">Enviar</button>
</div>

<script>
const messages = document.getElementById('messages');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const typing = document.getElementById('typing');
const statusText = document.getElementById('status-text');

function addMessage(content, from, type) {
  const div = document.createElement('div');

  if (type === 'info' || from === 'system') {
    div.className = 'msg msg-system';
    div.textContent = content;
  } else {
    const cls = from === 'user' ? 'msg-user' : 'msg-bot';
    div.className = 'msg ' + cls;

    const header = document.createElement('div');
    header.className = 'msg-header';
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = from === 'user' ? 'você' : 'zendia';
    const time = document.createElement('span');
    time.textContent = new Date().toLocaleTimeString('pt-BR', {hour:'2-digit', minute:'2-digit'});
    header.appendChild(name);
    header.appendChild(time);

    const body = document.createElement('div');
    body.textContent = content;

    div.appendChild(header);
    div.appendChild(body);
  }

  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
}

const ws = new WebSocket('ws://' + location.host + '/ws');

ws.onopen = () => {
  statusText.textContent = 'online';
  addMessage('conectado! pode conversar.', 'system', 'info');
};

ws.onclose = () => {
  statusText.textContent = 'desconectado';
  document.querySelector('.dot').style.background = '#ff7675';
  addMessage('conexão perdida.', 'system', 'info');
};

ws.onmessage = (e) => {
  typing.style.display = 'none';
  const data = JSON.parse(e.data);

  if (data.type === 'info') {
    addMessage(data.content, 'system', 'info');
  } else {
    addMessage(data.content, data.from, 'message');
  }
};

function send() {
  const text = input.value.trim();
  if (!text) return;

  addMessage(text, 'user', 'message');
  input.value = '';

  typing.style.display = 'block';
  messages.scrollTop = messages.scrollHeight;

  ws.send(JSON.stringify({ type: 'message', content: text, from: 'user' }));
}

sendBtn.addEventListener('click', send);
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') send();
});

input.focus();
</script>
</body>
</html>`
