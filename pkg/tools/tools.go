package tools

import (
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

type Tool struct {
	Name        string
	Description string
	Execute     func(args string) string
}

type Registry struct {
	Tools map[string]Tool
}

func NewRegistry() *Registry {
	r := &Registry{Tools: make(map[string]Tool)}
	r.registerDefaults()
	return r
}

func (r *Registry) Register(t Tool) {
	r.Tools[t.Name] = t
}

func (r *Registry) registerDefaults() {
	r.Register(Tool{
		Name:        "hora",
		Description: "retorna a hora atual",
		Execute: func(args string) string {
			return time.Now().Format("15:04:05 de 02/01/2006")
		},
	})

	r.Register(Tool{
		Name:        "data",
		Description: "retorna a data atual",
		Execute: func(args string) string {
			dias := []string{"domingo", "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado"}
			meses := []string{"", "janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro", "novembro", "dezembro"}
			now := time.Now()
			return fmt.Sprintf("%s, %d de %s de %d", dias[now.Weekday()], now.Day(), meses[now.Month()], now.Year())
		},
	})

	r.Register(Tool{
		Name:        "sistema",
		Description: "retorna informações do sistema",
		Execute: func(args string) string {
			return fmt.Sprintf("OS: %s | Arch: %s | CPUs: %d | Goroutines: %d",
				runtime.GOOS, runtime.GOARCH, runtime.NumCPU(), runtime.NumGoroutine())
		},
	})

	r.Register(Tool{
		Name:        "cmd",
		Description: "executa um comando do sistema",
		Execute: func(args string) string {
			args = strings.TrimSpace(args)
			if args == "" {
				return "nenhum comando fornecido"
			}

			// bloqueia comandos perigosos
			dangerous := []string{"rm -rf", "del /f", "format", "shutdown", "reboot", "mkfs", "dd if"}
			lower := strings.ToLower(args)
			for _, d := range dangerous {
				if strings.Contains(lower, d) {
					return "comando bloqueado por segurança"
				}
			}

			var cmd *exec.Cmd
			if runtime.GOOS == "windows" {
				cmd = exec.Command("cmd", "/c", args)
			} else {
				cmd = exec.Command("sh", "-c", args)
			}

			output, err := cmd.CombinedOutput()
			if err != nil {
				return fmt.Sprintf("erro: %v\n%s", err, string(output))
			}

			result := strings.TrimSpace(string(output))
			if len(result) > 500 {
				result = result[:500] + "... (truncado)"
			}
			return result
		},
	})

	r.Register(Tool{
		Name:        "calc",
		Description: "calcula expressões matemáticas simples",
		Execute: func(args string) string {
			// parser simples de expressões
			args = strings.TrimSpace(args)
			args = strings.ReplaceAll(args, " ", "")

			// tenta avaliar via cmd
			var cmd *exec.Cmd
			if runtime.GOOS == "windows" {
				cmd = exec.Command("powershell", "-Command", fmt.Sprintf("[math]::Round((%s), 6)", args))
			} else {
				cmd = exec.Command("bc", "-l")
				cmd.Stdin = strings.NewReader(args + "\n")
			}

			output, err := cmd.CombinedOutput()
			if err != nil {
				return "não consegui calcular: " + args
			}
			return strings.TrimSpace(string(output))
		},
	})
}

// TryExecute tenta detectar e executar um tool call na mensagem do usuário
func (r *Registry) TryExecute(message string) (string, bool) {
	lower := strings.ToLower(strings.TrimSpace(message))

	// hora
	if containsAny(lower, []string{"que horas", "hora atual", "que hora", "horas são"}) {
		return r.Tools["hora"].Execute(""), true
	}

	// data
	if containsAny(lower, []string{"que dia", "data de hoje", "dia é hoje", "qual a data"}) {
		return r.Tools["data"].Execute(""), true
	}

	// sistema
	if containsAny(lower, []string{"info do sistema", "informações do sistema", "meu sistema", "meu computador", "meu pc"}) {
		return r.Tools["sistema"].Execute(""), true
	}

	// calc
	if strings.HasPrefix(lower, "calcul") || strings.HasPrefix(lower, "quanto é") || strings.HasPrefix(lower, "quanto e") {
		expr := lower
		for _, prefix := range []string{"calcule ", "calcula ", "quanto é ", "quanto e "} {
			expr = strings.TrimPrefix(expr, prefix)
		}
		return r.Tools["calc"].Execute(expr), true
	}

	// cmd explícito
	if strings.HasPrefix(lower, "execute ") || strings.HasPrefix(lower, "roda ") || strings.HasPrefix(lower, "rode ") {
		cmd := message
		for _, prefix := range []string{"execute ", "Execute ", "roda ", "Roda ", "rode ", "Rode "} {
			cmd = strings.TrimPrefix(cmd, prefix)
		}
		return r.Tools["cmd"].Execute(cmd), true
	}

	return "", false
}

func containsAny(s string, patterns []string) bool {
	for _, p := range patterns {
		if strings.Contains(s, p) {
			return true
		}
	}
	return false
}

// ListTools retorna descrição de todas as tools disponíveis
func (r *Registry) ListTools() string {
	var lines []string
	for name, tool := range r.Tools {
		lines = append(lines, fmt.Sprintf("• %s: %s", name, tool.Description))
	}
	return strings.Join(lines, "\n")
}
