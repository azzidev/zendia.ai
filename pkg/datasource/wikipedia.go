package datasource

import (
	"compress/bzip2"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	WikiDumpURL  = "https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2"
	MinTextChars = 200 // ignora artigos muito curtos
)

// wikiDump representa a estrutura do XML da Wikipedia
type wikiPage struct {
	Title    string       `xml:"title"`
	Revision wikiRevision `xml:"revision"`
	NS       int          `xml:"ns"`
}

type wikiRevision struct {
	Text string `xml:"text"`
}

type PipelineStats struct {
	Downloaded  bool
	Parsed      int
	Cleaned     int
	Inserted    int
	Skipped     int
	ElapsedTime time.Duration
}

// DownloadDump baixa o dump da Wikipedia PT-BR
func DownloadDump(destDir string) (string, error) {
	destPath := filepath.Join(destDir, "ptwiki-latest.xml.bz2")

	// se já existe, pula
	if info, err := os.Stat(destPath); err == nil && info.Size() > 0 {
		fmt.Printf("📦 Dump já existe: %s (%.0f MB)\n", destPath, float64(info.Size())/1024/1024)
		return destPath, nil
	}

	fmt.Println("⬇️  Baixando dump da Wikipedia PT-BR...")
	fmt.Printf("   URL: %s\n", WikiDumpURL)
	fmt.Println("   Isso pode demorar uns minutos...")

	resp, err := http.Get(WikiDumpURL)
	if err != nil {
		return "", fmt.Errorf("erro no download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("status HTTP: %d", resp.StatusCode)
	}

	totalSize := resp.ContentLength
	fmt.Printf("   Tamanho: %.0f MB\n", float64(totalSize)/1024/1024)

	file, err := os.Create(destPath)
	if err != nil {
		return "", fmt.Errorf("erro ao criar arquivo: %w", err)
	}
	defer file.Close()

	// download com progresso
	written := int64(0)
	buf := make([]byte, 1024*1024) // 1MB buffer
	lastPrint := time.Now()

	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			file.Write(buf[:n])
			written += int64(n)

			if time.Since(lastPrint) > 5*time.Second {
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

// ProcessDump lê o dump bz2, limpa e insere no MongoDB
func ProcessDump(dumpPath string, store *MongoStore, maxArticles int) (*PipelineStats, error) {
	start := time.Now()
	stats := &PipelineStats{}

	file, err := os.Open(dumpPath)
	if err != nil {
		return nil, fmt.Errorf("erro ao abrir dump: %w", err)
	}
	defer file.Close()

	fmt.Println("\n🔄 Processando dump (descompactando + parseando XML)...")

	bzReader := bzip2.NewReader(file)
	decoder := xml.NewDecoder(bzReader)

	var batch []Document
	batchSize := 100

	for {
		token, err := decoder.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			// XML da wikipedia pode ter encoding issues, pula
			continue
		}

		se, ok := token.(xml.StartElement)
		if !ok || se.Name.Local != "page" {
			continue
		}

		var page wikiPage
		if err := decoder.DecodeElement(&page, &se); err != nil {
			continue
		}

		stats.Parsed++

		// pula namespaces que não são artigos (talk, user, etc)
		if page.NS != 0 {
			stats.Skipped++
			continue
		}

		// pula redirects
		if strings.HasPrefix(strings.ToLower(page.Revision.Text), "#redirect") ||
			strings.HasPrefix(strings.ToLower(page.Revision.Text), "#redirecionamento") {
			stats.Skipped++
			continue
		}

		// limpa o texto
		cleaned := CleanWikiText(page.Revision.Text)
		if len(cleaned) < MinTextChars {
			stats.Skipped++
			continue
		}

		stats.Cleaned++

		batch = append(batch, Document{
			Title: page.Title,
			Text:  cleaned,
			Chars: len(cleaned),
		})

		// insere em batch
		if len(batch) >= batchSize {
			if err := store.InsertBatch(batch); err != nil {
				fmt.Printf("   ⚠️  Erro ao inserir batch: %v\n", err)
			} else {
				stats.Inserted += len(batch)
			}
			batch = batch[:0]

			if stats.Inserted%1000 == 0 {
				fmt.Printf("   📝 %d artigos inseridos (parsed: %d, skipped: %d)\n",
					stats.Inserted, stats.Parsed, stats.Skipped)
			}
		}

		if maxArticles > 0 && stats.Inserted >= maxArticles {
			fmt.Printf("   🛑 Limite de %d artigos atingido\n", maxArticles)
			break
		}
	}

	// insere resto
	if len(batch) > 0 {
		if err := store.InsertBatch(batch); err == nil {
			stats.Inserted += len(batch)
		}
	}

	stats.ElapsedTime = time.Since(start)
	return stats, nil
}
