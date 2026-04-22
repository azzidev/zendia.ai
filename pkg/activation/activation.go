package activation

import "math"

type Func struct {
	Name       string
	Forward    func(float64) float64
	Derivative func(float64) float64
}

var Sigmoid = Func{
	Name: "sigmoid",
	Forward: func(x float64) float64 {
		return 1.0 / (1.0 + math.Exp(-x))
	},
	Derivative: func(x float64) float64 {
		s := 1.0 / (1.0 + math.Exp(-x))
		return s * (1 - s)
	},
}

var ReLU = Func{
	Name: "relu",
	Forward: func(x float64) float64 {
		if x > 0 {
			return x
		}
		return 0
	},
	Derivative: func(x float64) float64 {
		if x > 0 {
			return 1
		}
		return 0
	},
}

var Step = Func{
	Name: "step",
	Forward: func(x float64) float64 {
		if x >= 0.5 {
			return 1
		}
		return 0
	},
	Derivative: func(x float64) float64 {
		return 1
	},
}
