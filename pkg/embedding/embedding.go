package embedding

import (
	"math"
	"math/rand"

	"github.com/azzidev/zendia.ai/pkg/matrix"
)

// Table guarda os vetores de embedding pra cada token
type Table struct {
	VocabSize int
	EmbedDim  int
	Weights   *matrix.Matrix // [vocabSize x embedDim]
}

func NewTable(vocabSize, embedDim int) *Table {
	// inicialização Xavier
	scale := math.Sqrt(2.0 / float64(vocabSize+embedDim))
	weights := matrix.New(vocabSize, embedDim)
	for i := 0; i < vocabSize; i++ {
		for j := 0; j < embedDim; j++ {
			weights.Data[i][j] = rand.NormFloat64() * scale
		}
	}
	return &Table{
		VocabSize: vocabSize,
		EmbedDim:  embedDim,
		Weights:   weights,
	}
}

// Lookup retorna o vetor de embedding pra um token ID
func (t *Table) Lookup(id int) []float64 {
	if id < 0 || id >= t.VocabSize {
		return make([]float64, t.EmbedDim)
	}
	vec := make([]float64, t.EmbedDim)
	copy(vec, t.Weights.Data[id])
	return vec
}

// LookupBatch retorna embeddings pra uma sequência de IDs
// Retorna matrix [seqLen x embedDim]
func (t *Table) LookupBatch(ids []int) *matrix.Matrix {
	result := matrix.New(len(ids), t.EmbedDim)
	for i, id := range ids {
		if id >= 0 && id < t.VocabSize {
			copy(result.Data[i], t.Weights.Data[id])
		}
	}
	return result
}

// Update ajusta o embedding de um token (usado no treinamento)
func (t *Table) Update(id int, grad []float64, lr float64) {
	if id < 0 || id >= t.VocabSize {
		return
	}
	for j := 0; j < t.EmbedDim; j++ {
		t.Weights.Data[id][j] -= lr * grad[j]
	}
}

// Positional Encoding — adiciona informação de posição (usado no Transformer)
type PositionalEncoding struct {
	MaxLen   int
	EmbedDim int
	Encoding *matrix.Matrix
}

func NewPositionalEncoding(maxLen, embedDim int) *PositionalEncoding {
	enc := matrix.New(maxLen, embedDim)

	for pos := 0; pos < maxLen; pos++ {
		for i := 0; i < embedDim; i++ {
			angle := float64(pos) / math.Pow(10000, float64(2*(i/2))/float64(embedDim))
			if i%2 == 0 {
				enc.Data[pos][i] = math.Sin(angle)
			} else {
				enc.Data[pos][i] = math.Cos(angle)
			}
		}
	}

	return &PositionalEncoding{
		MaxLen:   maxLen,
		EmbedDim: embedDim,
		Encoding: enc,
	}
}

// Apply soma o positional encoding aos embeddings
// input: [seqLen x embedDim], output: [seqLen x embedDim]
func (pe *PositionalEncoding) Apply(embeddings *matrix.Matrix) *matrix.Matrix {
	result := matrix.New(embeddings.Rows, embeddings.Cols)
	for i := 0; i < embeddings.Rows && i < pe.MaxLen; i++ {
		for j := 0; j < embeddings.Cols; j++ {
			result.Data[i][j] = embeddings.Data[i][j] + pe.Encoding.Data[i][j]
		}
	}
	return result
}

// CosineSimilarity calcula similaridade entre dois vetores
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
