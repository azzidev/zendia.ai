package datasource

import (
	"bufio"
	"compress/gzip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
)

const (
	SubtitlesURL     = "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/pt_br.txt.gz"
	MinLineLen       = 10
	MaxLineLen       = 200
	MaxStorageMB     = 500
	AvgPairBytes     = 150 // estimativa média de bytes por par de diálogo no Mongo
)

var (
	reSubTags    = regexp.MustCompile(`<[^>]+>`)
	reSubBracket = regexp.MustCompile(`\[[^\]]*\]`)
	reSubParens  = regexp.MustCompile(`\([^)]*\)`)
	reSubDash    = regexp.MustCompile(`^[-–—]\s*`)
	reMultiSpace = regexp.MustCompile(`\s{2,}`)
	reNumbers    = regexp.MustCompile(`^\d+$`)
	reTimestamp  = regexp.MustCompile(`\d{2}:\d{2}`)
)

// DownloadSubtitles baixa o corpus OpenSubtitles PT-BR
func DownloadSubtitles(destDir string) (string, error) {
	destPath := filepath.Join(destDir, "opensubtitles-pt.gz")

	if info, err := os.Stat(destPath); err == nil && info.Size() > 0 {
		fmt.Printf("📦 Subtitles já existe: %s (%.0f MB)\n", destPath, float64(info.Size())/1024/1024)
		return destPath, nil
	}

	fmt.Println("⬇️  Baixando OpenSubtitles PT-BR...")
	fmt.Printf("   URL: %s\n", SubtitlesURL)
	fmt.Println("   Isso pode demorar uns minutos...")

	resp, err := http.Get(SubtitlesURL)
	if err != nil {
		return "", fmt.Errorf("erro no download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("status HTTP: %d", resp.StatusCode)
	}

	totalSize := resp.ContentLength
	if totalSize > 0 {
		fmt.Printf("   Tamanho: %.0f MB\n", float64(totalSize)/1024/1024)
	}

	file, err := os.Create(destPath)
	if err != nil {
		return "", fmt.Errorf("erro ao criar arquivo: %w", err)
	}
	defer file.Close()

	written := int64(0)
	buf := make([]byte, 1024*1024)
	lastPrint := time.Now()

	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			file.Write(buf[:n])
			written += int64(n)

			if time.Since(lastPrint) > 5*time.Second && totalSize > 0 {
				pct := float64(written) / float64(totalSize) * 100
				fmt.Printf("   📥 %.1f%% (%.0f / %.0f MB)\n", pct, float64(written)/1024/1024, float64(totalSize)/1024/1024)
				lastPrint = time.Now()
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", fmt.Errorf("erro durante download: %w", err)
		}
	}

	fmt.Printf("   ✅ Download completo: %.0f MB\n", float64(written)/1024/1024)
	return destPath, nil
}

// ProcessSubtitles lê o .gz, limpa e cria pares de diálogo no MongoDB
// Para quando atingir ~500MB de dados inseridos
func ProcessSubtitles(gzPath string, store *MongoStore) (*PipelineStats, error) {
	start := time.Now()
	stats := &PipelineStats{}

	file, err := os.Open(gzPath)
	if err != nil {
		return nil, fmt.Errorf("erro ao abrir: %w", err)
	}
	defer file.Close()

	fmt.Println("\n🔄 Processando legendas (descompactando + criando pares)...")
	fmt.Printf("   Limite: %d MB no MongoDB\n", MaxStorageMB)

	gzReader, err := gzip.NewReader(file)
	if err != nil {
		return nil, fmt.Errorf("erro ao descompactar: %w", err)
	}
	defer gzReader.Close()

	scanner := bufio.NewScanner(gzReader)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024)

	var batch []DialogueDoc
	batchSize := 500
	prevLine := ""
	estimatedBytes := int64(0)
	maxBytes := int64(MaxStorageMB) * 1024 * 1024

	for scanner.Scan() {
		line := scanner.Text()
		stats.Parsed++

		cleaned := cleanSubtitleLine(line)
		if cleaned == "" {
			prevLine = ""
			continue
		}

		if prevLine != "" && prevLine != cleaned {
			doc := DialogueDoc{User: prevLine, Bot: cleaned}
			batch = append(batch, doc)
			estimatedBytes += int64(len(prevLine) + len(cleaned) + 50) // +50 overhead BSON
			stats.Cleaned++
		}

		prevLine = cleaned

		if len(batch) >= batchSize {
			if err := store.InsertDialogueBatch(batch); err != nil {
				fmt.Printf("   ⚠️  Erro ao inserir: %v\n", err)
			} else {
				stats.Inserted += len(batch)
			}
			batch = batch[:0]

			if stats.Inserted%10000 == 0 {
				mb := float64(estimatedBytes) / 1024 / 1024
				fmt.Printf("   💬 %d pares | ~%.0f MB / %d MB\n",
					stats.Inserted, mb, MaxStorageMB)
			}
		}

		if estimatedBytes >= maxBytes {
			fmt.Printf("   🛑 Limite de %d MB atingido\n", MaxStorageMB)
			break
		}
	}

	if len(batch) > 0 {
		if err := store.InsertDialogueBatch(batch); err == nil {
			stats.Inserted += len(batch)
		}
	}

	stats.ElapsedTime = time.Since(start)
	return stats, nil
}

func cleanSubtitleLine(line string) string {
	line = strings.TrimSpace(line)

	// pula linhas vazias, números puros, timestamps
	if line == "" || reNumbers.MatchString(line) || reTimestamp.MatchString(line) {
		return ""
	}

	// remove tags HTML/XML
	line = reSubTags.ReplaceAllString(line, "")
	// remove [música], [risos], etc
	line = reSubBracket.ReplaceAllString(line, "")
	// remove (narrador), etc
	line = reSubParens.ReplaceAllString(line, "")
	// remove traço de diálogo
	line = reSubDash.ReplaceAllString(line, "")
	// limpa espaços
	line = reMultiSpace.ReplaceAllString(line, " ")
	line = strings.TrimSpace(line)

	// validações
	if len(line) < MinLineLen || len(line) > MaxLineLen {
		return ""
	}

	// pula se não é UTF-8 válido
	if !utf8.ValidString(line) {
		return ""
	}

	// pula linhas que são só pontuação
	letters := 0
	for _, r := range line {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= 'à' && r <= 'ÿ') {
			letters++
		}
	}
	if letters < 5 {
		return ""
	}

	// pula linhas com muitas maiúsculas (gritos, títulos)
	upper := 0
	for _, r := range line {
		if r >= 'A' && r <= 'Z' {
			upper++
		}
	}
	if len(line) > 10 && float64(upper)/float64(len(line)) > 0.5 {
		return ""
	}

	return line
}
