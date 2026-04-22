package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/azzidev/zendia.ai/pkg/chat"
	"github.com/azzidev/zendia.ai/pkg/datasource"
	"github.com/azzidev/zendia.ai/pkg/dialogue"
	"github.com/azzidev/zendia.ai/pkg/gpu"
	"github.com/azzidev/zendia.ai/pkg/matrix"
	"github.com/azzidev/zendia.ai/pkg/model"
	"github.com/azzidev/zendia.ai/pkg/tokenizer"
	"github.com/azzidev/zendia.ai/pkg/visualizer"
)

const defaultModelPath = "datasets/zendia-model.gob"

type VisData struct {
	Epoch        int                        `json:"epoch"`
	Step         int                        `json:"step"`
	Loss         float64                    `json:"loss"`
	AvgLoss      float64                    `json:"avgLoss"`
	TokensSeen   int                        `json:"tokensSeen"`
	Architecture []int                      `json:"architecture"`
	Sample       string                     `json:"sample"`
	Phase        string                     `json:"phase"`
	Activations  []model.ActivationSnapshot `json:"activations"`
}

func main() {
	modelPath := flag.String("model", defaultModelPath, "Caminho do modelo (.gob)")
	mode := flag.String("mode", "all", "Modo: download, download-dialogues, train, finetune, chat, all")
	mongoURI := flag.String("mongo", "mongodb://localhost:27017", "URI do MongoDB")
	maxArticles := flag.Int("max-articles", 10000, "Máximo de artigos Wikipedia")
	epochs := flag.Int("epochs", 5, "Epochs de treinamento base")
	ftEpochs := flag.Int("ft-epochs", 5, "Epochs de fine-tuning")
	flag.Parse()

	// inicializa GPU
	if err := gpu.Init(); err != nil {
		fmt.Printf("⚠️  GPU indisponível: %v (usando CPU paralela)\n", err)
		// sem GPU, usa CPU paralela
		matrix.MulFunc = func(a, b *matrix.Matrix) (*matrix.Matrix, error) {
			return a.MulParallel(b)
		}
	} else {
		defer gpu.Cleanup()
		matrix.MulFunc = func(a, b *matrix.Matrix) (*matrix.Matrix, error) {
			return gpu.MatMul(a, b)
		}
	}

	switch *mode {
	case "download":
		runDownloadWiki(*mongoURI, *maxArticles)
	case "download-dialogues":
		runDownloadDialogues(*mongoURI)
	case "train":
		runTrain(*mongoURI, *epochs, *modelPath)
	case "finetune":
		runFineTune(*mongoURI, *ftEpochs, *modelPath)
	case "chat":
		runChat(*modelPath)
	case "all":
		runDownloadWiki(*mongoURI, *maxArticles)
		runDownloadDialogues(*mongoURI)
		runTrain(*mongoURI, *epochs, *modelPath)
		runFineTune(*mongoURI, *ftEpochs, *modelPath)
		runChat(*modelPath)
	default:
		fmt.Println("Modos: download, download-dialogues, train, finetune, chat, all")
	}
}

func runDownloadWiki(mongoURI string, maxArticles int) {
	fmt.Println("╔══════════════════════════════════════════╗")
	fmt.Println("║  🌐 ZENDIA.AI — Download Wikipedia      ║")
	fmt.Println("╚══════════════════════════════════════════╝\n")

	store, err := datasource.NewMongoStore(mongoURI, "zendia")
	if err != nil {
		fmt.Printf("❌ %v\n", err)
		os.Exit(1)
	}
	defer store.Close()

	count, _ := store.Count()
	if count > 0 {
		fmt.Printf("📊 MongoDB já tem %d artigos. Pulando.\n", count)
		return
	}

	dataDir := filepath.Join(".", "datasets")
	os.MkdirAll(dataDir, 0755)

	dumpPath, err := datasource.DownloadDump(dataDir)
	if err != nil {
		fmt.Printf("❌ %v\n", err)
		os.Exit(1)
	}

	stats, err := datasource.ProcessDump(dumpPath, store, maxArticles)
	if err != nil {
		fmt.Printf("❌ %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n✅ %d artigos inseridos em %s\n", stats.Inserted, stats.ElapsedTime.Round(time.Second))
}

func runDownloadDialogues(mongoURI string) {
	fmt.Println("\n╔══════════════════════════════════════════╗")
	fmt.Println("║  🎬 ZENDIA.AI — Download Diálogos       ║")
	fmt.Println("║  OpenSubtitles PT-BR                    ║")
	fmt.Println("╚══════════════════════════════════════════╝\n")

	store, err := datasource.NewMongoStore(mongoURI, "zendia")
	if err != nil {
		fmt.Printf("❌ %v\n", err)
		os.Exit(1)
	}
	defer store.Close()

	count, _ := store.DialogueCount()
	if count > 0 {
		fmt.Printf("📊 MongoDB já tem %d pares de diálogo. Pulando.\n", count)
		return
	}

	dataDir := filepath.Join(".", "datasets")
	os.MkdirAll(dataDir, 0755)

	gzPath, err := datasource.DownloadSubtitles(dataDir)
	if err != nil {
		fmt.Printf("❌ %v\n", err)
		os.Exit(1)
	}

	stats, err := datasource.ProcessSubtitles(gzPath, store)
	if err != nil {
		fmt.Printf("❌ %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n✅ %d pares de diálogo inseridos em %s\n", stats.Inserted, stats.ElapsedTime.Round(time.Second))
	finalCount, _ := store.DialogueCount()
	fmt.Printf("📊 Total no MongoDB: %d diálogos\n", finalCount)
}

func runTrain(mongoURI string, numEpochs int, modelPath string) {
	fmt.Println("\n╔══════════════════════════════════════════╗")
	fmt.Println("║  🧠 ZENDIA.AI — Treinamento             ║")
	fmt.Println("╚══════════════════════════════════════════╝\n")

	store, err := datasource.NewMongoStore(mongoURI, "zendia")
	if err != nil {
		fmt.Printf("❌ %v\n", err)
		os.Exit(1)
	}
	defer store.Close()

	count, _ := store.Count()
	if count == 0 {
		fmt.Println("⚠️  Rode -mode=download primeiro.")
		return
	}

	var llm *model.LLM
	var tok *tokenizer.Tokenizer

	// tenta carregar checkpoint existente
	if existing, err := model.Load(modelPath); err == nil {
		fmt.Println("📂 Checkpoint encontrado! Continuando treinamento...")
		llm = existing
		tok = existing.Tokenizer
		fmt.Printf("   %s\n\n", llm)
	} else {
		// treina tokenizer e cria modelo do zero
		fmt.Println("🔤 Treinando tokenizer (2000 merges, 3000 textos)...")
		texts, _ := store.GetAllText(3000)
		tok = tokenizer.New()
		tok.Train(texts, 2000)
		dialogue.RegisterSpecialTokens(tok)
		fmt.Printf("   %s\n\n", tok)

		cfg := model.Config{
			VocabSize:    tok.VocabSize(),
			EmbedDim:     64,
			NumHeads:     4,
			NumLayers:    2,
			FFDim:        128,
			MaxSeqLen:    64,
			LearningRate: 0.0005,
		}

		llm = model.New(cfg, tok)
		fmt.Printf("🏗️  %s\n\n", llm)
	}

	// visualizador
	viz := visualizer.NewServer(8080)
	viz.Start()
	fmt.Println("⏳ Abra http://localhost:8080 ...")
	<-viz.Connected
	fmt.Println("✅ Conectado!\n")
	time.Sleep(300 * time.Millisecond)

	// prepara dados
	fmt.Println("📦 Preparando sequências...")
	trainTexts, _ := store.GetAllText(2000)

	var sequences [][]int
	for _, text := range trainTexts {
		for _, p := range strings.Split(text, "\n") {
			p = strings.TrimSpace(p)
			if len(p) < 40 || len(p) > 500 {
				continue
			}
			ids := tok.Encode(p)
			if len(ids) >= 4 && len(ids) <= llm.Config.MaxSeqLen {
				sequences = append(sequences, ids)
			}
		}
	}
	fmt.Printf("   %d sequências\n\n", len(sequences))

	// treina
	fmt.Println("🚀 Treinando...\n")
	arch := []int{llm.Config.VocabSize, llm.Config.EmbedDim, llm.Config.FFDim, llm.Config.EmbedDim, llm.Config.VocabSize}
	totalSteps := 0
	tokensSeen := 0
	start := time.Now()

	for epoch := 0; epoch < numEpochs; epoch++ {
		epochLoss := 0.0
		epochSteps := 0

		for i, seq := range sequences {
			loss := llm.TrainStep(seq)
			if math.IsNaN(loss) || math.IsInf(loss, 0) {
				continue
			}

			epochLoss += loss
			epochSteps++
			totalSteps++
			tokensSeen += len(seq)

			if totalSteps%10 == 0 {
				sample := ""
				if totalSteps%200 == 0 {
					sample = llm.Generate("o brasil", 30, 0.8)
				}
				viz.Broadcast(VisData{
					Epoch: epoch + 1, Step: totalSteps,
					Loss: loss, AvgLoss: epochLoss / float64(epochSteps),
					TokensSeen: tokensSeen, Architecture: arch,
					Sample: sample, Phase: "pre-training v2",
					Activations: llm.GetActivations(),
				})
			}

			if totalSteps%100 == 0 {
				avg := epochLoss / float64(epochSteps)
				tps := float64(tokensSeen) / time.Since(start).Seconds()
				fmt.Printf("   [E%d] %d/%d | Loss: %.4f | Avg: %.4f | %.0f tok/s\n",
					epoch+1, i+1, len(sequences), loss, avg, tps)
			}
		}

		avg := epochLoss / float64(maxInt(epochSteps, 1))
		fmt.Printf("\n   📊 Epoch %d | Avg Loss: %.4f\n", epoch+1, avg)

		// amostras
		for _, p := range []string{"o brasil", "a cidade de", "no ano de"} {
			fmt.Printf("   💬 \"%s\" → \"%s\"\n", p, llm.Generate(p, 30, 0.7))
		}
		fmt.Println()

		// SALVA A CADA EPOCH
		llm.Save(modelPath)
		fmt.Printf("   💾 Checkpoint salvo (epoch %d)\n\n", epoch+1)
	}

	fmt.Printf("\n✅ Treinamento base concluído em %s\n", time.Since(start).Round(time.Second))
}

func runFineTune(mongoURI string, numEpochs int, modelPath string) {
	fmt.Println("\n╔══════════════════════════════════════════╗")
	fmt.Println("║  💬 ZENDIA.AI — Fine-Tuning Diálogos    ║")
	fmt.Println("╚══════════════════════════════════════════╝\n")

	llm, err := model.Load(modelPath)
	if err != nil {
		fmt.Printf("❌ Modelo não encontrado. Rode -mode=train primeiro.\n")
		return
	}

	store, err := datasource.NewMongoStore(mongoURI, "zendia")
	if err != nil {
		fmt.Printf("⚠️  Mongo indisponível, usando só diálogos hardcoded.\n")
		store = nil
	}
	if store != nil {
		defer store.Close()
	}

	fmt.Println("📝 Carregando diálogos...")
	pairs := dialogue.FullDataset(store, 50000)

	ft := dialogue.NewFineTuner(pairs, llm.Tokenizer, llm.Config.MaxSeqLen)
	fmt.Printf("\n   %s\n\n", ft)

	if len(ft.Sequences) == 0 {
		fmt.Println("⚠️  Nenhuma sequência válida.")
		return
	}

	llm.Config.LearningRate = 0.0001

	fmt.Println("🚀 Fine-tuning...\n")
	start := time.Now()
	stepsPerEpoch := len(ft.Sequences)

	for epoch := 0; epoch < numEpochs; epoch++ {
		epochLoss := 0.0
		validSteps := 0

		for step := 0; step < stepsPerEpoch; step++ {
			result := ft.Step(llm.TrainStep)
			if result != nil && !math.IsNaN(result.Loss) && !math.IsInf(result.Loss, 0) {
				epochLoss += result.Loss
				validSteps++
			}

			if validSteps > 0 && validSteps%1000 == 0 {
				avg := epochLoss / float64(validSteps)
				fmt.Printf("   [E%d] Step %d/%d | Avg Loss: %.4f\n",
					epoch+1, step+1, stepsPerEpoch, avg)
			}
		}

		avg := epochLoss / float64(maxInt(validSteps, 1))
		fmt.Printf("\n   📊 Epoch %d/%d | Avg Loss: %.4f\n", epoch+1, numEpochs, avg)

		// testa
		fmt.Println("   💬 Testando:")
		tests := []string{"oi", "quem é você", "o que é ia", "me ajuda", "tchau"}
		for _, t := range tests {
			prompt := dialogue.UserToken + " " + t + " " + dialogue.BotToken
			resp := dialogue.ParseBotResponse(llm.Generate(prompt, 40, 0.7))
			fmt.Printf("      Você: %s\n      Bot:  %s\n\n", t, resp)
		}

		// SALVA A CADA EPOCH
		llm.Save(modelPath)
		fmt.Printf("   💾 Checkpoint salvo (ft epoch %d)\n\n", epoch+1)
	}

	fmt.Printf("\n✅ Fine-tuning concluído em %s\n", time.Since(start).Round(time.Second))
}

func runChat(modelPath string) {
	fmt.Println("\n╔══════════════════════════════════════════╗")
	fmt.Println("║  💬 ZENDIA.AI — Chat                    ║")
	fmt.Println("╚══════════════════════════════════════════╝\n")

	llm, err := model.Load(modelPath)
	if err != nil {
		fmt.Println("❌ Modelo não encontrado. Rode o pipeline completo:")
		fmt.Println("   go run ./cmd/main.go -mode=all")
		return
	}

	server := chat.NewServer(llm, 8080)
	fmt.Println("🌐 Abra http://localhost:8080 pra conversar!\n")
	server.Start()
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
