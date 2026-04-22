package attention

import (
	"math"

	"github.com/azzidev/zendia.ai/pkg/matrix"
)

// Head é uma cabeça de atenção (single head)
// Q, K, V são projeções lineares aprendidas
type Head struct {
	Wq, Wk, Wv *matrix.Matrix // [embedDim x headDim]
	Bq, Bk, Bv []float64      // bias [headDim]
	HeadDim    int

	// cache do forward
	Q, K, V    *matrix.Matrix
	Scores     *matrix.Matrix
	Input      *matrix.Matrix

	// gradientes
	DWq, DWk, DWv *matrix.Matrix
	DBq, DBk, DBv []float64
}

func NewHead(embedDim, headDim int) *Head {
	return &Head{
		Wq:      matrix.Xavier(embedDim, headDim),
		Wk:      matrix.Xavier(embedDim, headDim),
		Wv:      matrix.Xavier(embedDim, headDim),
		Bq:      make([]float64, headDim),
		Bk:      make([]float64, headDim),
		Bv:      make([]float64, headDim),
		HeadDim: headDim,
	}
}

// Forward calcula self-attention: softmax(Q*K^T / sqrt(d)) * V
// input: [seqLen x embedDim] → output: [seqLen x headDim]
func (h *Head) Forward(input *matrix.Matrix, causalMask bool) *matrix.Matrix {
	h.Input = input

	// projeções lineares: Q = input * Wq + Bq
	h.Q, _ = matrix.MatMul(input, h.Wq)
	h.Q = h.Q.AddRow(h.Bq)

	h.K, _ = matrix.MatMul(input, h.Wk)
	h.K = h.K.AddRow(h.Bk)

	h.V, _ = matrix.MatMul(input, h.Wv)
	h.V = h.V.AddRow(h.Bv)

	// scores = Q * K^T / sqrt(headDim)
	kt := h.K.Transpose()
	h.Scores, _ = matrix.MatMul(h.Q, kt)
	scale := 1.0 / math.Sqrt(float64(h.HeadDim))
	h.Scores = h.Scores.Scale(scale)

	// causal mask: impede que tokens vejam o futuro
	if causalMask {
		for i := 0; i < h.Scores.Rows; i++ {
			for j := i + 1; j < h.Scores.Cols; j++ {
				h.Scores.Data[i][j] = -1e9
			}
		}
	}

	// softmax por linha
	h.Scores = h.Scores.SoftmaxRows()

	// output = scores * V
	output, _ := matrix.MatMul(h.Scores, h.V)
	return output
}

// Backward calcula gradientes da attention head
// outputGrad: [seqLen x headDim] → inputGrad: [seqLen x embedDim]
func (h *Head) Backward(outputGrad *matrix.Matrix, lr float64) *matrix.Matrix {
	seqLen := h.Input.Rows
	scale := 1.0 / math.Sqrt(float64(h.HeadDim))

	// grad em relação a V: dV = scores^T * outputGrad
	scoresT := h.Scores.Transpose()
	dV, _ := matrix.MatMul(scoresT, outputGrad)

	// grad em relação a scores: dScores = outputGrad * V^T
	vt := h.V.Transpose()
	dScores, _ := matrix.MatMul(outputGrad, vt)

	// grad através do softmax: dS_ij = S_ij * (dScores_ij - sum_k(S_ik * dScores_ik))
	dScoresSoftmax := matrix.New(seqLen, seqLen)
	for i := 0; i < seqLen; i++ {
		dotSum := 0.0
		for k := 0; k < seqLen; k++ {
			dotSum += h.Scores.Data[i][k] * dScores.Data[i][k]
		}
		for j := 0; j < seqLen; j++ {
			dScoresSoftmax.Data[i][j] = h.Scores.Data[i][j] * (dScores.Data[i][j] - dotSum)
		}
	}

	// scale
	dScoresSoftmax = dScoresSoftmax.Scale(scale)

	// grad em relação a Q: dQ = dScoresScaled * K
	dQ, _ := matrix.MatMul(dScoresSoftmax, h.K)

	// grad em relação a K: dK = dScoresScaled^T * Q
	dScoresT := dScoresSoftmax.Transpose()
	dK, _ := matrix.MatMul(dScoresT, h.Q)

	// gradientes dos pesos: dW = input^T * dProjection
	inputT := h.Input.Transpose()
	h.DWq, _ = matrix.MatMul(inputT, dQ)
	h.DWk, _ = matrix.MatMul(inputT, dK)
	h.DWv, _ = matrix.MatMul(inputT, dV)

	// gradientes dos bias: soma das colunas
	h.DBq = colSum(dQ)
	h.DBk = colSum(dK)
	h.DBv = colSum(dV)

	// atualiza pesos
	h.updateWeights(lr)

	// grad em relação ao input: dInput = dQ*Wq^T + dK*Wk^T + dV*Wv^T
	wqT := h.Wq.Transpose()
	wkT := h.Wk.Transpose()
	wvT := h.Wv.Transpose()

	dInput1, _ := matrix.MatMul(dQ, wqT)
	dInput2, _ := matrix.MatMul(dK, wkT)
	dInput3, _ := matrix.MatMul(dV, wvT)

	dInput, _ := dInput1.Add(dInput2)
	dInput, _ = dInput.Add(dInput3)

	return dInput
}

func (h *Head) updateWeights(lr float64) {
	updateMatrix(h.Wq, h.DWq, lr)
	updateMatrix(h.Wk, h.DWk, lr)
	updateMatrix(h.Wv, h.DWv, lr)
	updateSlice(h.Bq, h.DBq, lr)
	updateSlice(h.Bk, h.DBk, lr)
	updateSlice(h.Bv, h.DBv, lr)
}

// MultiHead combina múltiplas heads de atenção
type MultiHead struct {
	Heads   []*Head
	Wo      *matrix.Matrix // projeção de saída [numHeads*headDim x embedDim]
	Bo      []float64
	NumHeads int
	HeadDim  int
	EmbedDim int

	// cache
	ConcatOutput *matrix.Matrix
	DWo          *matrix.Matrix
	DBo          []float64
}

func NewMultiHead(embedDim, numHeads int) *MultiHead {
	headDim := embedDim / numHeads
	heads := make([]*Head, numHeads)
	for i := range heads {
		heads[i] = NewHead(embedDim, headDim)
	}
	return &MultiHead{
		Heads:    heads,
		Wo:       matrix.Xavier(numHeads*headDim, embedDim),
		Bo:       make([]float64, embedDim),
		NumHeads: numHeads,
		HeadDim:  headDim,
		EmbedDim: embedDim,
	}
}

// Forward: cada head processa o input, concatena, projeta
// input: [seqLen x embedDim] → output: [seqLen x embedDim]
func (mh *MultiHead) Forward(input *matrix.Matrix, causalMask bool) *matrix.Matrix {
	seqLen := input.Rows

	// cada head produz [seqLen x headDim]
	headOutputs := make([]*matrix.Matrix, mh.NumHeads)
	for i, head := range mh.Heads {
		headOutputs[i] = head.Forward(input, causalMask)
	}

	// concatena: [seqLen x (numHeads * headDim)]
	totalDim := mh.NumHeads * mh.HeadDim
	mh.ConcatOutput = matrix.New(seqLen, totalDim)
	for i := 0; i < seqLen; i++ {
		offset := 0
		for _, ho := range headOutputs {
			copy(mh.ConcatOutput.Data[i][offset:], ho.Data[i])
			offset += mh.HeadDim
		}
	}

	// projeção de saída: output = concat * Wo + Bo
	output, _ := matrix.MatMul(mh.ConcatOutput, mh.Wo)
	output = output.AddRow(mh.Bo)
	return output
}

// Backward propaga gradientes por todas as heads
func (mh *MultiHead) Backward(outputGrad *matrix.Matrix, lr float64) *matrix.Matrix {
	// grad da projeção de saída
	concatT := mh.ConcatOutput.Transpose()
	mh.DWo, _ = matrix.MatMul(concatT, outputGrad)
	mh.DBo = colSum(outputGrad)

	// grad em relação ao concat
	woT := mh.Wo.Transpose()
	dConcat, _ := matrix.MatMul(outputGrad, woT)

	// atualiza Wo
	updateMatrix(mh.Wo, mh.DWo, lr)
	updateSlice(mh.Bo, mh.DBo, lr)

	// split grad pra cada head e propaga
	seqLen := dConcat.Rows
	dInput := matrix.New(seqLen, mh.EmbedDim)

	for h := 0; h < mh.NumHeads; h++ {
		// extrai grad desta head
		headGrad := matrix.New(seqLen, mh.HeadDim)
		offset := h * mh.HeadDim
		for i := 0; i < seqLen; i++ {
			copy(headGrad.Data[i], dConcat.Data[i][offset:offset+mh.HeadDim])
		}

		// propaga pela head
		headInputGrad := mh.Heads[h].Backward(headGrad, lr)

		// acumula no dInput
		for i := 0; i < seqLen; i++ {
			for j := 0; j < mh.EmbedDim; j++ {
				dInput.Data[i][j] += headInputGrad.Data[i][j]
			}
		}
	}

	return dInput
}

// --- helpers ---

func colSum(m *matrix.Matrix) []float64 {
	sums := make([]float64, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			sums[j] += m.Data[i][j]
		}
	}
	return sums
}

func updateMatrix(w, dw *matrix.Matrix, lr float64) {
	for i := 0; i < w.Rows; i++ {
		for j := 0; j < w.Cols; j++ {
			w.Data[i][j] -= lr * dw.Data[i][j]
		}
	}
}

func updateSlice(w, dw []float64, lr float64) {
	for i := range w {
		w[i] -= lr * dw[i]
	}
}
