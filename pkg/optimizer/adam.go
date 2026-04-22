package optimizer

import (
	"math"

	"github.com/azzidev/zendia.ai/pkg/matrix"
)

// Adam implementa o otimizador Adam (Adaptive Moment Estimation)
// Usado no GPT, BERT, e praticamente todo modelo moderno
type Adam struct {
	LR      float64
	Beta1   float64 // decay rate do primeiro momento (default 0.9)
	Beta2   float64 // decay rate do segundo momento (default 0.999)
	Epsilon float64 // evita divisão por zero (default 1e-8)
	Step    int     // contador de steps

	// estados por parâmetro (chave = ponteiro do parâmetro)
	mMatrix map[*matrix.Matrix]*matrix.Matrix // primeiro momento (média)
	vMatrix map[*matrix.Matrix]*matrix.Matrix // segundo momento (variância)
	mSlice  map[*[]float64][]float64
	vSlice  map[*[]float64][]float64
}

func NewAdam(lr float64) *Adam {
	return &Adam{
		LR:      lr,
		Beta1:   0.9,
		Beta2:   0.999,
		Epsilon: 1e-8,
		mMatrix: make(map[*matrix.Matrix]*matrix.Matrix),
		vMatrix: make(map[*matrix.Matrix]*matrix.Matrix),
		mSlice:  make(map[*[]float64][]float64),
		vSlice:  make(map[*[]float64][]float64),
	}
}

// UpdateMatrix atualiza uma matrix de pesos com Adam
func (a *Adam) UpdateMatrix(param *matrix.Matrix, grad *matrix.Matrix) {
	if _, ok := a.mMatrix[param]; !ok {
		a.mMatrix[param] = matrix.New(param.Rows, param.Cols)
		a.vMatrix[param] = matrix.New(param.Rows, param.Cols)
	}

	m := a.mMatrix[param]
	v := a.vMatrix[param]

	a.Step++
	t := float64(a.Step)

	for i := 0; i < param.Rows; i++ {
		for j := 0; j < param.Cols; j++ {
			g := grad.Data[i][j]

			// atualiza momentos
			m.Data[i][j] = a.Beta1*m.Data[i][j] + (1-a.Beta1)*g
			v.Data[i][j] = a.Beta2*v.Data[i][j] + (1-a.Beta2)*g*g

			// correção de bias
			mHat := m.Data[i][j] / (1 - math.Pow(a.Beta1, t))
			vHat := v.Data[i][j] / (1 - math.Pow(a.Beta2, t))

			// atualiza peso
			param.Data[i][j] -= a.LR * mHat / (math.Sqrt(vHat) + a.Epsilon)
		}
	}
}

// UpdateSlice atualiza um slice de pesos (bias) com Adam
func (a *Adam) UpdateSlice(param *[]float64, grad []float64) {
	if _, ok := a.mSlice[param]; !ok {
		a.mSlice[param] = make([]float64, len(*param))
		a.vSlice[param] = make([]float64, len(*param))
	}

	m := a.mSlice[param]
	v := a.vSlice[param]

	t := float64(a.Step)
	if t == 0 {
		t = 1
	}

	for i := range *param {
		g := grad[i]

		m[i] = a.Beta1*m[i] + (1-a.Beta1)*g
		v[i] = a.Beta2*v[i] + (1-a.Beta2)*g*g

		mHat := m[i] / (1 - math.Pow(a.Beta1, t))
		vHat := v[i] / (1 - math.Pow(a.Beta2, t))

		(*param)[i] -= a.LR * mHat / (math.Sqrt(vHat) + a.Epsilon)
	}
}

// ClipGradients limita a norma dos gradientes pra evitar explosão
func ClipGradients(grad *matrix.Matrix, maxNorm float64) {
	norm := 0.0
	for i := 0; i < grad.Rows; i++ {
		for j := 0; j < grad.Cols; j++ {
			norm += grad.Data[i][j] * grad.Data[i][j]
		}
	}
	norm = math.Sqrt(norm)

	if norm > maxNorm {
		scale := maxNorm / norm
		for i := 0; i < grad.Rows; i++ {
			for j := 0; j < grad.Cols; j++ {
				grad.Data[i][j] *= scale
			}
		}
	}
}

// ClipGradientsSlice limita gradientes de um slice
func ClipGradientsSlice(grad []float64, maxNorm float64) {
	norm := 0.0
	for _, g := range grad {
		norm += g * g
	}
	norm = math.Sqrt(norm)

	if norm > maxNorm {
		scale := maxNorm / norm
		for i := range grad {
			grad[i] *= scale
		}
	}
}
