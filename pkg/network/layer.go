package network

import (
	"github.com/azzidev/zendia.ai/pkg/activation"
	"github.com/azzidev/zendia.ai/pkg/matrix"
)

type Layer struct {
	Weights    *matrix.Matrix
	Biases     *matrix.Matrix
	Activation activation.Func

	// cache do forward (usado no backprop)
	Input      *matrix.Matrix
	RawOutput  *matrix.Matrix
	Output     *matrix.Matrix

	// gradientes
	DWeights *matrix.Matrix
	DBiases  *matrix.Matrix
}

func NewLayer(inputSize, outputSize int, act activation.Func) *Layer {
	return &Layer{
		Weights:    matrix.Randomize(outputSize, inputSize),
		Biases:     matrix.New(outputSize, 1),
		Activation: act,
	}
}

func (l *Layer) Forward(input *matrix.Matrix) *matrix.Matrix {
	l.Input = input

	raw, _ := l.Weights.Mul(input)
	raw, _ = raw.Add(l.Biases)
	l.RawOutput = raw

	l.Output = raw.Apply(l.Activation.Forward)
	return l.Output
}

func (l *Layer) Backward(outputGrad *matrix.Matrix) *matrix.Matrix {
	// derivada da ativação aplicada no raw output
	actDeriv := l.RawOutput.Apply(l.Activation.Derivative)

	// delta = outputGrad ⊙ f'(raw)
	delta, _ := outputGrad.Hadamard(actDeriv)

	// gradiente dos pesos: delta * input^T
	inputT := l.Input.Transpose()
	l.DWeights, _ = delta.Mul(inputT)

	// gradiente dos biases: delta
	l.DBiases = delta

	// propagar erro pra camada anterior: W^T * delta
	weightsT := l.Weights.Transpose()
	inputGrad, _ := weightsT.Mul(delta)

	return inputGrad
}
