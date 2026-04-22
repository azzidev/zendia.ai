package neuron

import (
	"github.com/azzidev/zendia.ai/pkg/activation"
	"github.com/azzidev/zendia.ai/pkg/matrix"
)

type Perceptron struct {
	Weights      *matrix.Matrix
	Bias         float64
	LearningRate float64
	Activation   activation.Func
}

func NewPerceptron(inputSize int, lr float64, act activation.Func) *Perceptron {
	return &Perceptron{
		Weights:      matrix.Randomize(1, inputSize),
		Bias:         0,
		LearningRate: lr,
		Activation:   act,
	}
}

func (p *Perceptron) Predict(inputs []float64) float64 {
	sum := p.Bias
	for i, w := range p.Weights.Data[0] {
		sum += w * inputs[i]
	}
	return p.Activation.Forward(sum)
}

func (p *Perceptron) RawOutput(inputs []float64) float64 {
	sum := p.Bias
	for i, w := range p.Weights.Data[0] {
		sum += w * inputs[i]
	}
	return sum
}

type TrainResult struct {
	Epoch   int       `json:"epoch"`
	Loss    float64   `json:"loss"`
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
}

func (p *Perceptron) Train(inputs [][]float64, targets []float64, epochs int, callback func(TrainResult)) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0

		for i, input := range inputs {
			raw := p.RawOutput(input)
			output := p.Activation.Forward(raw)

			err := targets[i] - output
			totalLoss += err * err

			deriv := p.Activation.Derivative(raw)
			for j := range p.Weights.Data[0] {
				p.Weights.Data[0][j] += p.LearningRate * err * deriv * input[j]
			}
			p.Bias += p.LearningRate * err * deriv
		}

		avgLoss := totalLoss / float64(len(inputs))

		if callback != nil {
			weights := make([]float64, len(p.Weights.Data[0]))
			copy(weights, p.Weights.Data[0])
			callback(TrainResult{
				Epoch:   epoch,
				Loss:    avgLoss,
				Weights: weights,
				Bias:    p.Bias,
			})
		}
	}
}

func (p *Perceptron) GetWeights() []float64 {
	w := make([]float64, len(p.Weights.Data[0]))
	copy(w, p.Weights.Data[0])
	return w
}
