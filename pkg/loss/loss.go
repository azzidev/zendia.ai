package loss

import "github.com/azzidev/zendia.ai/pkg/matrix"

type Func struct {
	Name       string
	Forward    func(predicted, target *matrix.Matrix) float64
	Derivative func(predicted, target *matrix.Matrix) *matrix.Matrix
}

var MSE = Func{
	Name: "mse",
	Forward: func(predicted, target *matrix.Matrix) float64 {
		sum := 0.0
		for i := 0; i < predicted.Rows; i++ {
			diff := predicted.Data[i][0] - target.Data[i][0]
			sum += diff * diff
		}
		return sum / float64(predicted.Rows)
	},
	Derivative: func(predicted, target *matrix.Matrix) *matrix.Matrix {
		result := matrix.New(predicted.Rows, 1)
		n := float64(predicted.Rows)
		for i := 0; i < predicted.Rows; i++ {
			result.Data[i][0] = 2 * (predicted.Data[i][0] - target.Data[i][0]) / n
		}
		return result
	},
}
