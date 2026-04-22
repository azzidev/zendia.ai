package model

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/azzidev/zendia.ai/pkg/embedding"
	"github.com/azzidev/zendia.ai/pkg/matrix"
	"github.com/azzidev/zendia.ai/pkg/tokenizer"
	"github.com/azzidev/zendia.ai/pkg/transformer"
)

type Config struct {
	VocabSize  int
	EmbedDim   int
	NumHeads   int
	NumLayers  int
	FFDim      int
	MaxSeqLen  int
	LearningRate float64
}

func DefaultConfig(vocabSize int) Config {
	return Config{
		VocabSize:    vocabSize,
		EmbedDim:     64,
		NumHeads:     4,
		NumLayers:    2,
		FFDim:        128,
		MaxSeqLen:    128,
		LearningRate: 0.001,
	}
}

type LLM struct {
	Config    Config
	EmbedTable *embedding.Table
	PosEnc     *embedding.PositionalEncoding
	Blocks     []*transformer.Block
	FinalNorm  *transformer.LayerNorm
	LMHead     *matrix.Matrix // [embedDim x vocabSize] — projeta pra logits
	LMBias     []float64
	Tokenizer  *tokenizer.Tokenizer
}

func New(cfg Config, tok *tokenizer.Tokenizer) *LLM {
	blocks := make([]*transformer.Block, cfg.NumLayers)
	for i := range blocks {
		blocks[i] = transformer.NewBlock(cfg.EmbedDim, cfg.NumHeads, cfg.FFDim)
	}

	return &LLM{
		Config:     cfg,
		EmbedTable: embedding.NewTable(cfg.VocabSize, cfg.EmbedDim),
		PosEnc:     embedding.NewPositionalEncoding(cfg.MaxSeqLen, cfg.EmbedDim),
		Blocks:     blocks,
		FinalNorm:  transformer.NewLayerNorm(cfg.EmbedDim),
		LMHead:     matrix.Xavier(cfg.EmbedDim, cfg.VocabSize),
		LMBias:     make([]float64, cfg.VocabSize),
		Tokenizer:  tok,
	}
}

// Forward: tokens → embeddings → transformer blocks → logits
// input: []int (token IDs) → output: [seqLen x vocabSize] (logits)
func (m *LLM) Forward(tokenIDs []int) *matrix.Matrix {
	// embeddings + positional encoding
	embeds := m.EmbedTable.LookupBatch(tokenIDs)
	x := m.PosEnc.Apply(embeds)

	// transformer blocks
	for _, block := range m.Blocks {
		x = block.Forward(x)
	}

	// final layer norm
	x = m.FinalNorm.Forward(x)

	// LM head: projeta pra vocabulário
	logits, _ := matrix.MatMul(x, m.LMHead)
	logits = logits.AddRow(m.LMBias)

	return logits
}

// TrainStep treina um passo com uma sequência de tokens
// Tarefa: dado tokens[0..n-1], prever tokens[1..n]
func (m *LLM) TrainStep(tokenIDs []int) float64 {
	if len(tokenIDs) < 2 {
		return 0
	}

	seqLen := len(tokenIDs) - 1
	input := tokenIDs[:seqLen]
	targets := tokenIDs[1 : seqLen+1]

	// forward
	logits := m.Forward(input)

	// cross-entropy loss + softmax
	loss := 0.0
	probs := matrix.New(seqLen, m.Config.VocabSize)

	for i := 0; i < seqLen; i++ {
		// softmax estável
		maxLogit := logits.Data[i][0]
		for j := 1; j < m.Config.VocabSize; j++ {
			if logits.Data[i][j] > maxLogit {
				maxLogit = logits.Data[i][j]
			}
		}
		expSum := 0.0
		for j := 0; j < m.Config.VocabSize; j++ {
			probs.Data[i][j] = math.Exp(logits.Data[i][j] - maxLogit)
			expSum += probs.Data[i][j]
		}
		for j := 0; j < m.Config.VocabSize; j++ {
			probs.Data[i][j] /= expSum
		}

		// cross-entropy: -log(prob do token correto)
		targetProb := probs.Data[i][targets[i]]
		if targetProb < 1e-10 {
			targetProb = 1e-10
		}
		loss -= math.Log(targetProb)
	}
	loss /= float64(seqLen)

	// backward: grad do cross-entropy + softmax = probs - one_hot(target)
	grad := matrix.New(seqLen, m.Config.VocabSize)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < m.Config.VocabSize; j++ {
			grad.Data[i][j] = probs.Data[i][j]
		}
		grad.Data[i][targets[i]] -= 1.0
		// normaliza pelo seqLen
		for j := 0; j < m.Config.VocabSize; j++ {
			grad.Data[i][j] /= float64(seqLen)
		}
	}

	lr := m.Config.LearningRate

	// grad do LM head
	finalOut := m.FinalNorm.Forward(m.lastTransformerOutput())
	finalOutT := finalOut.Transpose()
	dLMHead, _ := matrix.MatMul(finalOutT, grad)

	// atualiza LM head
	for i := 0; i < m.LMHead.Rows; i++ {
		for j := 0; j < m.LMHead.Cols; j++ {
			m.LMHead.Data[i][j] -= lr * dLMHead.Data[i][j]
		}
	}
	dLMBias := colSum(grad)
	for j := range m.LMBias {
		m.LMBias[j] -= lr * dLMBias[j]
	}

	// propaga pelo LM head
	lmHeadT := m.LMHead.Transpose()
	blockGrad, _ := matrix.MatMul(grad, lmHeadT)

	// propaga pela final norm
	blockGrad = m.FinalNorm.Backward(blockGrad, lr)

	// propaga pelos transformer blocks (reverso)
	for i := len(m.Blocks) - 1; i >= 0; i-- {
		blockGrad = m.Blocks[i].Backward(blockGrad, lr)
	}

	// atualiza embeddings
	for i, id := range input {
		if id >= 0 && id < m.Config.VocabSize {
			for j := 0; j < m.Config.EmbedDim; j++ {
				m.EmbedTable.Weights.Data[id][j] -= lr * blockGrad.Data[i][j]
			}
		}
	}

	return loss
}

func (m *LLM) lastTransformerOutput() *matrix.Matrix {
	if len(m.Blocks) > 0 {
		lastBlock := m.Blocks[len(m.Blocks)-1]
		return lastBlock.FFNInput // output do último bloco antes do FFN residual
	}
	return nil
}

// Generate gera texto a partir de um prompt
func (m *LLM) Generate(prompt string, maxTokens int, temperature float64) string {
	ids := m.Tokenizer.Encode(prompt)
	// remove <eos> do final pra continuar gerando
	if len(ids) > 1 && ids[len(ids)-1] == m.Tokenizer.Vocab["<eos>"] {
		ids = ids[:len(ids)-1]
	}

	for i := 0; i < maxTokens; i++ {
		// trunca se passar do maxSeqLen
		input := ids
		if len(input) > m.Config.MaxSeqLen {
			input = input[len(input)-m.Config.MaxSeqLen:]
		}

		logits := m.Forward(input)

		// pega logits do último token
		lastLogits := logits.Data[logits.Rows-1]

		// amostragem com temperatura
		nextToken := sampleWithTemperature(lastLogits, temperature)

		// para se gerou <eos>
		if nextToken == m.Tokenizer.Vocab["<eos>"] {
			break
		}

		ids = append(ids, nextToken)
	}

	return m.Tokenizer.Decode(ids)
}

func sampleWithTemperature(logits []float64, temperature float64) int {
	if temperature <= 0 {
		temperature = 0.01
	}

	// aplica temperatura
	scaled := make([]float64, len(logits))
	maxVal := logits[0]
	for _, v := range logits {
		if v > maxVal {
			maxVal = v
		}
	}

	expSum := 0.0
	for i, v := range logits {
		scaled[i] = math.Exp((v - maxVal) / temperature)
		expSum += scaled[i]
	}

	// normaliza pra probabilidades
	for i := range scaled {
		scaled[i] /= expSum
	}

	// amostra
	r := rand.Float64()
	cumSum := 0.0
	for i, p := range scaled {
		cumSum += p
		if r <= cumSum {
			return i
		}
	}
	return len(scaled) - 1
}

// ParamCount retorna o número total de parâmetros do modelo
func (m *LLM) ParamCount() int {
	count := 0

	// embeddings
	count += m.Config.VocabSize * m.Config.EmbedDim

	// transformer blocks
	for _, block := range m.Blocks {
		// attention: Q, K, V pra cada head + Wo
		headDim := m.Config.EmbedDim / m.Config.NumHeads
		perHead := 3 * (m.Config.EmbedDim*headDim + headDim) // Wq,Wk,Wv + biases
		count += m.Config.NumHeads * perHead
		count += m.Config.NumHeads*headDim*m.Config.EmbedDim + m.Config.EmbedDim // Wo + Bo

		// FFN
		count += m.Config.EmbedDim*m.Config.FFDim + m.Config.FFDim   // W1 + B1
		count += m.Config.FFDim*m.Config.EmbedDim + m.Config.EmbedDim // W2 + B2

		// layer norms
		count += 2 * (2 * m.Config.EmbedDim) // gamma + beta x2

		_ = block
	}

	// final norm + LM head
	count += 2 * m.Config.EmbedDim
	count += m.Config.EmbedDim*m.Config.VocabSize + m.Config.VocabSize

	return count
}

func (m *LLM) String() string {
	return fmt.Sprintf("LLM(vocab=%d, embed=%d, heads=%d, layers=%d, ff=%d, params=%d)",
		m.Config.VocabSize, m.Config.EmbedDim, m.Config.NumHeads,
		m.Config.NumLayers, m.Config.FFDim, m.ParamCount())
}

// ActivationSnapshot retorna as ativações de cada camada do último forward
// Cada camada é um []float64 com a média das ativações por neurônio
type ActivationSnapshot struct {
	LayerName   string    `json:"name"`
	Size        int       `json:"size"`
	Activations []float64 `json:"activations"` // média por neurônio
	Min         float64   `json:"min"`
	Max         float64   `json:"max"`
	Mean        float64   `json:"mean"`
}

func (m *LLM) GetActivations() []ActivationSnapshot {
	var snaps []ActivationSnapshot

	for i, block := range m.Blocks {
		// attention output
		if block.AttnInput != nil {
			snaps = append(snaps, matrixToSnapshot(
				fmt.Sprintf("block%d_attn", i), block.AttnInput))
		}
		// FFN output
		if block.FFNInput != nil {
			snaps = append(snaps, matrixToSnapshot(
				fmt.Sprintf("block%d_ffn", i), block.FFNInput))
		}
	}

	return snaps
}

func matrixToSnapshot(name string, m *matrix.Matrix) ActivationSnapshot {
	// média absoluta por coluna (neurônio)
	acts := make([]float64, m.Cols)
	min, max, sum := 999.0, -999.0, 0.0

	for j := 0; j < m.Cols; j++ {
		colSum := 0.0
		for i := 0; i < m.Rows; i++ {
			v := math.Abs(m.Data[i][j])
			colSum += v
		}
		acts[j] = colSum / float64(m.Rows)
		if acts[j] < min { min = acts[j] }
		if acts[j] > max { max = acts[j] }
		sum += acts[j]
	}

	return ActivationSnapshot{
		LayerName:   name,
		Size:        m.Cols,
		Activations: acts,
		Min:         min,
		Max:         max,
		Mean:        sum / float64(m.Cols),
	}
}

func colSum(m *matrix.Matrix) []float64 {
	sums := make([]float64, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			sums[j] += m.Data[i][j]
		}
	}
	return sums
}
