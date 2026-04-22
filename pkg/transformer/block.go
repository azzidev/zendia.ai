package transformer

import (
	"math"

	"github.com/azzidev/zendia.ai/pkg/attention"
	"github.com/azzidev/zendia.ai/pkg/matrix"
)

// LayerNorm normaliza os valores pra estabilizar o treinamento
type LayerNorm struct {
	Gamma []float64 // scale
	Beta  []float64 // shift
	Dim   int
	Eps   float64

	// cache
	Input    *matrix.Matrix
	Normed   *matrix.Matrix
	Mean     []float64
	Variance []float64
}

func NewLayerNorm(dim int) *LayerNorm {
	gamma := make([]float64, dim)
	beta := make([]float64, dim)
	for i := range gamma {
		gamma[i] = 1.0
	}
	return &LayerNorm{Gamma: gamma, Beta: beta, Dim: dim, Eps: 1e-5}
}

// Forward: normaliza cada linha (token) independentemente
// input: [seqLen x dim] → output: [seqLen x dim]
func (ln *LayerNorm) Forward(input *matrix.Matrix) *matrix.Matrix {
	ln.Input = input
	ln.Mean = make([]float64, input.Rows)
	ln.Variance = make([]float64, input.Rows)
	ln.Normed = matrix.New(input.Rows, input.Cols)
	result := matrix.New(input.Rows, input.Cols)

	for i := 0; i < input.Rows; i++ {
		// média
		sum := 0.0
		for j := 0; j < input.Cols; j++ {
			sum += input.Data[i][j]
		}
		ln.Mean[i] = sum / float64(input.Cols)

		// variância
		varSum := 0.0
		for j := 0; j < input.Cols; j++ {
			diff := input.Data[i][j] - ln.Mean[i]
			varSum += diff * diff
		}
		ln.Variance[i] = varSum / float64(input.Cols)

		// normaliza e aplica gamma/beta
		std := math.Sqrt(ln.Variance[i] + ln.Eps)
		for j := 0; j < input.Cols; j++ {
			ln.Normed.Data[i][j] = (input.Data[i][j] - ln.Mean[i]) / std
			result.Data[i][j] = ln.Normed.Data[i][j]*ln.Gamma[j] + ln.Beta[j]
		}
	}
	return result
}

// Backward propaga gradientes pela layer norm
func (ln *LayerNorm) Backward(outputGrad *matrix.Matrix, lr float64) *matrix.Matrix {
	n := float64(ln.Dim)
	inputGrad := matrix.New(outputGrad.Rows, outputGrad.Cols)

	for i := 0; i < outputGrad.Rows; i++ {
		std := math.Sqrt(ln.Variance[i] + ln.Eps)

		// gradientes de gamma e beta
		dGamma := make([]float64, ln.Dim)
		dBeta := make([]float64, ln.Dim)
		for j := 0; j < ln.Dim; j++ {
			dGamma[j] = outputGrad.Data[i][j] * ln.Normed.Data[i][j]
			dBeta[j] = outputGrad.Data[i][j]
		}

		// grad em relação ao normed
		dNormed := make([]float64, ln.Dim)
		for j := 0; j < ln.Dim; j++ {
			dNormed[j] = outputGrad.Data[i][j] * ln.Gamma[j]
		}

		// grad em relação ao input (simplificado)
		dNormedSum := 0.0
		dNormedXSum := 0.0
		for j := 0; j < ln.Dim; j++ {
			dNormedSum += dNormed[j]
			dNormedXSum += dNormed[j] * ln.Normed.Data[i][j]
		}

		for j := 0; j < ln.Dim; j++ {
			inputGrad.Data[i][j] = (dNormed[j] - dNormedSum/n - ln.Normed.Data[i][j]*dNormedXSum/n) / std
		}

		// atualiza gamma e beta
		for j := 0; j < ln.Dim; j++ {
			ln.Gamma[j] -= lr * dGamma[j]
			ln.Beta[j] -= lr * dBeta[j]
		}
	}

	return inputGrad
}

// FeedForward é a rede feed-forward do transformer (2 camadas lineares + GELU)
type FeedForward struct {
	W1, W2 *matrix.Matrix // W1: [embedDim x ffDim], W2: [ffDim x embedDim]
	B1, B2 []float64

	// cache
	Input    *matrix.Matrix
	Hidden   *matrix.Matrix
	Activated *matrix.Matrix

	DW1, DW2 *matrix.Matrix
	DB1, DB2 []float64
}

func NewFeedForward(embedDim, ffDim int) *FeedForward {
	return &FeedForward{
		W1: matrix.Xavier(embedDim, ffDim),
		W2: matrix.Xavier(ffDim, embedDim),
		B1: make([]float64, ffDim),
		B2: make([]float64, embedDim),
	}
}

// GELU activation (usado no GPT)
func gelu(x float64) float64 {
	return 0.5 * x * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(x+0.044715*x*x*x)))
}

func geluDerivative(x float64) float64 {
	c := math.Sqrt(2.0 / math.Pi)
	inner := c * (x + 0.044715*x*x*x)
	tanhVal := math.Tanh(inner)
	sech2 := 1 - tanhVal*tanhVal
	dInner := c * (1 + 3*0.044715*x*x)
	return 0.5*(1+tanhVal) + 0.5*x*sech2*dInner
}

// Forward: input → linear → GELU → linear
func (ff *FeedForward) Forward(input *matrix.Matrix) *matrix.Matrix {
	ff.Input = input

	// hidden = input * W1 + B1
	ff.Hidden, _ = matrix.MatMul(input, ff.W1)
	ff.Hidden = ff.Hidden.AddRow(ff.B1)

	// GELU
	ff.Activated = ff.Hidden.Apply(gelu)

	// output = activated * W2 + B2
	output, _ := matrix.MatMul(ff.Activated, ff.W2)
	output = output.AddRow(ff.B2)
	return output
}

// Backward propaga gradientes pelo feed forward
func (ff *FeedForward) Backward(outputGrad *matrix.Matrix, lr float64) *matrix.Matrix {
	// grad W2: dW2 = activated^T * outputGrad
	activatedT := ff.Activated.Transpose()
	ff.DW2, _ = matrix.MatMul(activatedT, outputGrad)
	ff.DB2 = colSum(outputGrad)

	// grad em relação ao activated
	w2T := ff.W2.Transpose()
	dActivated, _ := matrix.MatMul(outputGrad, w2T)

	// grad através do GELU
	dHidden := matrix.New(dActivated.Rows, dActivated.Cols)
	for i := 0; i < dHidden.Rows; i++ {
		for j := 0; j < dHidden.Cols; j++ {
			dHidden.Data[i][j] = dActivated.Data[i][j] * geluDerivative(ff.Hidden.Data[i][j])
		}
	}

	// grad W1: dW1 = input^T * dHidden
	inputT := ff.Input.Transpose()
	ff.DW1, _ = matrix.MatMul(inputT, dHidden)
	ff.DB1 = colSum(dHidden)

	// atualiza pesos
	updateMatrix(ff.W1, ff.DW1, lr)
	updateMatrix(ff.W2, ff.DW2, lr)
	updateSlice(ff.B1, ff.DB1, lr)
	updateSlice(ff.B2, ff.DB2, lr)

	// grad em relação ao input
	w1T := ff.W1.Transpose()
	inputGrad, _ := matrix.MatMul(dHidden, w1T)
	return inputGrad
}

// Block é um bloco completo do Transformer: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
type Block struct {
	Attention *attention.MultiHead
	FFN       *FeedForward
	Norm1     *LayerNorm
	Norm2     *LayerNorm

	// cache pra residual connections
	AttnInput *matrix.Matrix
	FFNInput  *matrix.Matrix
}

func NewBlock(embedDim, numHeads, ffDim int) *Block {
	return &Block{
		Attention: attention.NewMultiHead(embedDim, numHeads),
		FFN:       NewFeedForward(embedDim, ffDim),
		Norm1:     NewLayerNorm(embedDim),
		Norm2:     NewLayerNorm(embedDim),
	}
}

// Forward: Pre-LN Transformer (mais estável que Post-LN)
// x → LayerNorm → Attention → + residual → LayerNorm → FFN → + residual
func (b *Block) Forward(input *matrix.Matrix) *matrix.Matrix {
	b.AttnInput = input

	// sub-bloco 1: attention com residual
	normed1 := b.Norm1.Forward(input)
	attnOut := b.Attention.Forward(normed1, true) // causal mask = true
	residual1, _ := input.Add(attnOut)

	b.FFNInput = residual1

	// sub-bloco 2: FFN com residual
	normed2 := b.Norm2.Forward(residual1)
	ffnOut := b.FFN.Forward(normed2)
	residual2, _ := residual1.Add(ffnOut)

	return residual2
}

// Backward propaga gradientes pelo bloco inteiro
func (b *Block) Backward(outputGrad *matrix.Matrix, lr float64) *matrix.Matrix {
	// grad do residual 2: vai direto + pelo FFN
	ffnGrad := b.FFN.Backward(b.Norm2.Backward(outputGrad, lr), lr)
	grad1, _ := outputGrad.Add(ffnGrad)

	// grad do residual 1: vai direto + pela attention
	attnGrad := b.Attention.Backward(b.Norm1.Backward(grad1, lr), lr)
	inputGrad, _ := grad1.Add(attnGrad)

	return inputGrad
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
