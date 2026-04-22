package matrix

import "fmt"

// MulFunc é a função de multiplicação usada globalmente
// Pode ser substituída pelo pacote gpu pra usar OpenCL
var MulFunc func(a, b *Matrix) (*Matrix, error)

func init() {
	// default: CPU
	MulFunc = func(a, b *Matrix) (*Matrix, error) {
		return a.Mul(b)
	}
}

// MatMul usa a função global (GPU se disponível, senão CPU)
func MatMul(a, b *Matrix) (*Matrix, error) {
	if a.Cols != b.Rows {
		return nil, fmt.Errorf("dimensões incompatíveis: (%d,%d) * (%d,%d)", a.Rows, a.Cols, b.Rows, b.Cols)
	}
	return MulFunc(a, b)
}
