package visualizer

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

type Server struct {
	clients   map[*websocket.Conn]bool
	mu        sync.Mutex
	port      int
	Connected chan struct{}
	connOnce  sync.Once
}

func NewServer(port int) *Server {
	return &Server{
		clients:   make(map[*websocket.Conn]bool),
		port:      port,
		Connected: make(chan struct{}),
	}
}

func (s *Server) handleWS(w http.ResponseWriter, r *http.Request) {
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Println("websocket upgrade error:", err)
		return
	}
	s.mu.Lock()
	s.clients[conn] = true
	s.mu.Unlock()
	s.connOnce.Do(func() { close(s.Connected) })
	defer func() {
		s.mu.Lock()
		delete(s.clients, conn)
		s.mu.Unlock()
		conn.Close()
	}()
	for {
		if _, _, err := conn.ReadMessage(); err != nil {
			break
		}
	}
}

func (s *Server) Broadcast(data interface{}) {
	msg, err := json.Marshal(data)
	if err != nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for conn := range s.clients {
		if err := conn.WriteMessage(websocket.TextMessage, msg); err != nil {
			conn.Close()
			delete(s.clients, conn)
		}
	}
}

func (s *Server) Start() {
	mux := http.NewServeMux()
	mux.HandleFunc("/ws", s.handleWS)
	// serve arquivos estáticos da pasta web/
	mux.Handle("/", http.FileServer(http.Dir("web")))

	addr := fmt.Sprintf(":%d", s.port)
	fmt.Printf("🧠 Visualizador rodando em http://localhost%s\n", addr)
	go http.ListenAndServe(addr, mux)
}
