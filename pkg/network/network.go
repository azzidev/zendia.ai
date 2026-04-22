package network

import (
	"github.com/azzidev/zendia.ai/pkg/loss"
	"github.com/azzidev/zendia.ai/pkg/matrix"
)

type Network struct {
	Layers       []*Layer
	LearningRate float64
	LossFunc     loss.Func
}

func New(lr float64, lossFunc loss.Func) *Network {
	return &Network{
		LearningRate: lr,
		LossFunc:     lossFunc,
	}
}

func (n *Network) AddLayer(layer *Layer) {
	n.Layers = append(n.Layers, layer)
}

func (n *Network) Forward(input *matrix.Matrix) *matrix.Matrix {
	current := input
	for _, layer := range n.Layers {
		current = layer.Forward(current)
	}
	return current
}

func (n *Network) Backward(target *matrix.Matrix) {
	lastLayer := n.Layers[len(n.Layers)-1]
	grad := n.LossFunc.Derivative(lastLayer.Output, target)

	for i := len(n.Layers) - 1; i >= 0; i-- {
		grad = n.Layers[i].Backward(grad)
	}
}

func (n *Network) UpdateWeights() {
	for _, layer := range n.Layers {
		scaled := layer.DWeights.Scale(n.LearningRate)
		layer.Weights, _ = layer.Weights.Sub(scaled)

		scaledB := layer.DBiases.Scale(n.LearningRate)
		layer.Biases, _ = layer.Biases.Sub(scaledB)
	}
}

type TrainResult struct {
	Epoch  int              `json:"epoch"`
	Loss   float64          `json:"loss"`
	Layers []LayerSnapshot  `json:"layers"`
}

type LayerSnapshot struct {
	Weights    [][]float64 `json:"weights"`
	Biases     []float64   `json:"biases"`
	Activations []float64  `json:"activations"`
}

func (n *Network) Train(inputs [][]float64, targets [][]float64, epochs int, callback func(TrainResult)) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0

		for i, input := range inputs {
			inputM := matrix.FromSlice(input)
			targetM := matrix.FromSlice(targets[i])

			output := n.Forward(inputM)
			totalLoss += n.LossFunc.Forward(output, targetM)

			n.Backward(targetM)
			n.UpdateWeights()
		}

		avgLoss := totalLoss / float64(len(inputs))

		if callback != nil {
			snapshot := n.Snapshot()
			callback(TrainResult{
				Epoch:  epoch,
				Loss:   avgLoss,
				Layers: snapshot,
			})
		}
	}
}

func (n *Network) Predict(input []float64) []float64 {
	output := n.Forward(matrix.FromSlice(input))
	return output.ToSlice()
}

func (n *Network) Snapshot() []LayerSnapshot {
	snapshots := make([]LayerSnapshot, len(n.Layers))
	for i, layer := range n.Layers {
		weights := make([][]float64, layer.Weights.Rows)
		for r := 0; r < layer.Weights.Rows; r++ {
			weights[r] = make([]float64, layer.Weights.Cols)
			copy(weights[r], layer.Weights.Data[r])
		}

		biases := make([]float64, layer.Biases.Rows)
		for r := 0; r < layer.Biases.Rows; r++ {
			biases[r] = layer.Biases.Data[r][0]
		}

		var activations []float64
		if layer.Output != nil {
			activations = layer.Output.ToSlice()
		}

		snapshots[i] = LayerSnapshot{
			Weights:     weights,
			Biases:      biases,
			Activations: activations,
		}
	}
	return snapshots
}
