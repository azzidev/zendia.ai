package dialogue

import (
	"fmt"
	"math"
	"strings"

	"github.com/azzidev/zendia.ai/pkg/datasource"
	"github.com/azzidev/zendia.ai/pkg/tokenizer"
)

const (
	UserToken = "<user>"
	BotToken  = "<bot>"
	SepToken  = "<sep>"
)

type Pair struct {
	User string
	Bot  string
}

// RegisterSpecialTokens adiciona tokens especiais de diálogo no tokenizer
func RegisterSpecialTokens(tok *tokenizer.Tokenizer) {
	tok.AddSpecialToken(UserToken)
	tok.AddSpecialToken(BotToken)
	tok.AddSpecialToken(SepToken)
}

// FormatPair formata um par de diálogo pro treinamento
func FormatPair(p Pair) string {
	return UserToken + " " + p.User + " " + BotToken + " " + p.Bot
}

// ParseBotResponse extrai a resposta do bot de um texto gerado
func ParseBotResponse(text string) string {
	idx := strings.LastIndex(text, BotToken)
	if idx == -1 {
		return text
	}
	response := strings.TrimSpace(text[idx+len(BotToken):])
	// corta se tiver outro <user> (alucinação)
	if userIdx := strings.Index(response, UserToken); userIdx != -1 {
		response = strings.TrimSpace(response[:userIdx])
	}
	return response
}

// BuildTrainingData gera sequências tokenizadas dos diálogos
func BuildTrainingData(pairs []Pair, tok *tokenizer.Tokenizer, maxLen int) [][]int {
	var sequences [][]int
	for _, pair := range pairs {
		text := FormatPair(pair)
		ids := tok.Encode(text)
		if len(ids) >= 4 && len(ids) <= maxLen {
			sequences = append(sequences, ids)
		}
	}
	return sequences
}

// Dataset retorna diálogos em PT-BR pra fine-tuning
func Dataset() []Pair {
	return []Pair{
		// saudações
		{"oi", "olá! como posso te ajudar?"},
		{"olá", "oi! tudo bem? em que posso ajudar?"},
		{"e aí", "e aí! tudo certo? o que precisa?"},
		{"bom dia", "bom dia! como posso te ajudar hoje?"},
		{"boa tarde", "boa tarde! em que posso ser útil?"},
		{"boa noite", "boa noite! precisa de algo?"},
		{"fala", "fala! como vai? o que precisa?"},
		{"tudo bem", "tudo ótimo! e com você?"},
		{"como vai", "vou bem! obrigado por perguntar. e você?"},

		// sobre a IA
		{"quem é você", "eu sou a zendia, uma inteligência artificial criada do zero em golang."},
		{"qual seu nome", "meu nome é zendia. fui criada pra conversar e ajudar."},
		{"o que você é", "sou uma ia feita do zero, sem usar bibliotecas prontas. puro código."},
		{"como você funciona", "funciono com um transformer que aprendeu padrões da língua portuguesa lendo a wikipedia."},
		{"você é inteligente", "estou aprendendo! cada conversa me ajuda a melhorar."},
		{"você pensa", "processo informações e gero respostas, mas pensar de verdade é algo bem mais complexo."},
		{"você tem sentimentos", "não tenho sentimentos reais. sou um modelo matemático que gera texto."},
		{"quem te criou", "fui criada como um projeto de ia do zero em golang."},

		// conhecimento geral
		{"o que é ia", "inteligência artificial é a área da computação que cria sistemas capazes de realizar tarefas que normalmente precisam de inteligência humana."},
		{"o que é machine learning", "machine learning é quando um programa aprende padrões a partir de dados, sem ser programado explicitamente pra cada tarefa."},
		{"o que é um transformer", "transformer é uma arquitetura de rede neural que usa atenção pra entender relações entre palavras em um texto."},
		{"o que é deep learning", "deep learning usa redes neurais com muitas camadas pra aprender representações complexas dos dados."},
		{"o que é python", "python é uma linguagem de programação muito usada em ciência de dados e inteligência artificial."},
		{"o que é golang", "golang é uma linguagem criada pelo google. é rápida, simples e ótima pra sistemas de alta performance."},
		{"o que é programação", "programação é a arte de escrever instruções pra um computador executar. é como ensinar uma máquina a fazer coisas."},

		// brasil
		{"qual a capital do brasil", "a capital do brasil é brasília, localizada no distrito federal."},
		{"quem descobriu o brasil", "pedro álvares cabral chegou ao brasil em 1500, mas os povos indígenas já viviam aqui há milhares de anos."},
		{"qual o maior estado do brasil", "o amazonas é o maior estado do brasil em área territorial."},
		{"quantos estados tem o brasil", "o brasil tem 26 estados e o distrito federal, totalizando 27 unidades federativas."},

		// utilidade
		{"me ajuda", "claro! me diz o que você precisa e vou fazer o possível pra ajudar."},
		{"não entendi", "sem problema! me explica de outro jeito que eu tento ajudar melhor."},
		{"obrigado", "de nada! fico feliz em ajudar."},
		{"valeu", "por nada! qualquer coisa é só chamar."},
		{"tchau", "tchau! foi bom conversar. até a próxima!"},
		{"até mais", "até mais! volte quando quiser."},

		// conversação
		{"conta uma piada", "por que o programador usa óculos? porque ele não consegue c sharp."},
		{"qual sua cor favorita", "se eu pudesse ver cores, acho que gostaria de azul. combina com tecnologia."},
		{"você gosta de música", "não consigo ouvir música, mas acho fascinante como ela é basicamente matemática organizada."},
		{"qual o sentido da vida", "quarenta e dois. brincadeira! acho que cada pessoa encontra seu próprio sentido."},
		{"me conta algo interessante", "sabia que o cérebro humano tem cerca de 86 bilhões de neurônios? eu tenho bem menos que isso."},

		// técnico
		{"como criar uma ia", "comece entendendo redes neurais, depois implemente um perceptron, evolua pra mlp e depois pra transformer. é o que fizemos aqui!"},
		{"o que é backpropagation", "backpropagation é o algoritmo que calcula como ajustar os pesos da rede neural pra reduzir o erro."},
		{"o que é attention", "attention é o mecanismo que permite a rede focar nas partes mais relevantes do input pra gerar cada palavra."},
		{"o que é embedding", "embedding transforma palavras em vetores de números. palavras similares ficam próximas no espaço vetorial."},
		{"o que é tokenizer", "tokenizer quebra texto em pedaços menores chamados tokens. pode ser palavras, subpalavras ou caracteres."},
		{"o que é loss", "loss é a função que mede o quão errada a rede está. o objetivo do treinamento é minimizar o loss."},
		{"o que é epoch", "epoch é uma passada completa por todos os dados de treinamento. geralmente treina por várias epochs."},
		{"o que é gpu", "gpu é uma placa de vídeo. ela é muito boa pra treinar ias porque faz muitos cálculos em paralelo."},
	}
}

// AugmentedDataset retorna o dataset com variações
func AugmentedDataset() []Pair {
	base := Dataset()
	var augmented []Pair
	augmented = append(augmented, base...)

	for _, p := range base {
		augmented = append(augmented, Pair{p.User + "?", p.Bot})
		augmented = append(augmented, Pair{p.User + "!", p.Bot})
		augmented = append(augmented, Pair{strings.ToUpper(p.User[:1]) + p.User[1:], p.Bot})
	}

	return augmented
}

// LoadFromMongo carrega diálogos do MongoDB (OpenSubtitles)
func LoadFromMongo(store *datasource.MongoStore, maxPairs int) ([]Pair, error) {
	docs, err := store.GetDialogues(maxPairs)
	if err != nil {
		return nil, err
	}

	pairs := make([]Pair, len(docs))
	for i, doc := range docs {
		pairs[i] = Pair{User: doc.User, Bot: doc.Bot}
	}
	return pairs, nil
}

// FullDataset combina diálogos hardcoded + MongoDB
func FullDataset(store *datasource.MongoStore, maxMongoPairs int) []Pair {
	// base hardcoded (sempre inclui)
	pairs := AugmentedDataset()
	fmt.Printf("   📝 %d pares hardcoded\n", len(pairs))

	// mongo (OpenSubtitles)
	if store != nil {
		mongoPairs, err := LoadFromMongo(store, maxMongoPairs)
		if err == nil && len(mongoPairs) > 0 {
			pairs = append(pairs, mongoPairs...)
			fmt.Printf("   🎬 %d pares do OpenSubtitles\n", len(mongoPairs))
		}
	}

	fmt.Printf("   📊 Total: %d pares de diálogo\n", len(pairs))
	return pairs
}

// TrainResult guarda métricas do fine-tuning
type TrainResult struct {
	Step    int
	Loss    float64
	AvgLoss float64
}

// FineTune treina o modelo com diálogos
type FineTuner struct {
	Sequences [][]int
	StepCount int
	TotalLoss float64
}

func NewFineTuner(pairs []Pair, tok *tokenizer.Tokenizer, maxLen int) *FineTuner {
	return &FineTuner{
		Sequences: BuildTrainingData(pairs, tok, maxLen),
	}
}

func (ft *FineTuner) Step(trainFn func([]int) float64) *TrainResult {
	if len(ft.Sequences) == 0 {
		return nil
	}

	idx := ft.StepCount % len(ft.Sequences)
	loss := trainFn(ft.Sequences[idx])

	if !math.IsNaN(loss) && !math.IsInf(loss, 0) {
		ft.StepCount++
		ft.TotalLoss += loss
	}

	return &TrainResult{
		Step:    ft.StepCount,
		Loss:    loss,
		AvgLoss: ft.TotalLoss / float64(max(ft.StepCount, 1)),
	}
}

func (ft *FineTuner) String() string {
	return fmt.Sprintf("FineTuner(sequences=%d)", len(ft.Sequences))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
