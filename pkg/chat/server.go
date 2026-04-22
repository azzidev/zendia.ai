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

		var response string
		if toolResult, ok := s.Tools.TryExecute(incoming.Content); ok {
			response = toolResult
		} else {
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
	mux.HandleFunc("/ws/chat", s.handleChat)
	mux.Handle("/", http.FileServer(http.Dir("web")))

	addr := fmt.Sprintf(":%d", s.port)
	fmt.Printf("💬 Chat rodando em http://localhost%s/chat.html\n", addr)
	http.ListenAndServe(addr, mux)
}
