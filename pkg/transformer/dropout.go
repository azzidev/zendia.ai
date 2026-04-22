package transformer

import (
	"math/rand"

	"github.com/azzidev/zendia.ai/pkg/matrix"
)

type Dropout struct {
	Rate    float64
	Enabled bool
	mask    *matrix.Matrix
}

func NewDropout(rate float64) *Dropout {
	return &Dropout{Rate: rate, Enabled: true}
}

// Forward aplica dropout: zera valores aleatórios e escala os restantes
func (d *Dropout) Forward(input *matrix.Matrix) *matrix.Matrix {
	if !d.Enabled || d.Rate <= 0 {
		return input
	}

	d.mask = matrix.New(input.Rows, input.Cols)
	result := matrix.New(input.Rows, input.Cols)
	scale := 1.0 / (1.0 - d.Rate)

	for i := 0; i < input.Rows; i++ {
		for j := 0; j < input.Cols; j++ {
			if rand.Float64() > d.Rate {
				d.mask.Data[i][j] = 1.0
				result.Data[i][j] = input.Data[i][j] * scale
			}
		}
	}
	return result
}

// Backward propaga gradientes respeitando a máscara
func (d *Dropout) Backward(grad *matrix.Matrix) *matrix.Matrix {
	if !d.Enabled || d.Rate <= 0 || d.mask == nil {
		return grad
	}

	result := matrix.New(grad.Rows, grad.Cols)
	scale := 1.0 / (1.0 - d.Rate)

	for i := 0; i < grad.Rows; i++ {
		for j := 0; j < grad.Cols; j++ {
			result.Data[i][j] = grad.Data[i][j] * d.mask.Data[i][j] * scale
		}
	}
	return result
}
