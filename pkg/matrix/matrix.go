package matrix

import (
	"fmt"
	"math"
	"math/rand"
)

type Matrix struct {
	Rows int
	Cols int
	Data [][]float64
}

func New(rows, cols int) *Matrix {
	data := make([][]float64, rows)
	for i := range data {
		data[i] = make([]float64, cols)
	}
	return &Matrix{Rows: rows, Cols: cols, Data: data}
}

func FromSlice(data []float64) *Matrix {
	m := New(len(data), 1)
	for i, v := range data {
		m.Data[i][0] = v
	}
	return m
}

func Randomize(rows, cols int) *Matrix {
	m := New(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Data[i][j] = rand.Float64()*2 - 1
		}
	}
	return m
}

func (m *Matrix) Apply(fn func(float64) float64) *Matrix {
	result := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = fn(m.Data[i][j])
		}
	}
	return result
}

func (m *Matrix) Add(other *Matrix) (*Matrix, error) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		return nil, fmt.Errorf("dimensões incompatíveis: (%d,%d) + (%d,%d)", m.Rows, m.Cols, other.Rows, other.Cols)
	}
	result := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] + other.Data[i][j]
		}
	}
	return result, nil
}

func (m *Matrix) Sub(other *Matrix) (*Matrix, error) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		return nil, fmt.Errorf("dimensões incompatíveis: (%d,%d) - (%d,%d)", m.Rows, m.Cols, other.Rows, other.Cols)
	}
	result := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] - other.Data[i][j]
		}
	}
	return result, nil
}

func (m *Matrix) Mul(other *Matrix) (*Matrix, error) {
	if m.Cols != other.Rows {
		return nil, fmt.Errorf("dimensões incompatíveis pra multiplicação: (%d,%d) * (%d,%d)", m.Rows, m.Cols, other.Rows, other.Cols)
	}
	result := New(m.Rows, other.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < other.Cols; j++ {
			sum := 0.0
			for k := 0; k < m.Cols; k++ {
				sum += m.Data[i][k] * other.Data[k][j]
			}
			result.Data[i][j] = sum
		}
	}
	return result, nil
}

func (m *Matrix) Hadamard(other *Matrix) (*Matrix, error) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		return nil, fmt.Errorf("dimensões incompatíveis: (%d,%d) ⊙ (%d,%d)", m.Rows, m.Cols, other.Rows, other.Cols)
	}
	result := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] * other.Data[i][j]
		}
	}
	return result, nil
}

func (m *Matrix) Scale(scalar float64) *Matrix {
	result := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = m.Data[i][j] * scalar
		}
	}
	return result
}

func (m *Matrix) Transpose() *Matrix {
	result := New(m.Cols, m.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			result.Data[j][i] = m.Data[i][j]
		}
	}
	return result
}

func (m *Matrix) ToSlice() []float64 {
	result := make([]float64, 0, m.Rows*m.Cols)
	for i := 0; i < m.Rows; i++ {
		result = append(result, m.Data[i]...)
	}
	return result
}

// Row retorna uma cópia da linha i como slice
func (m *Matrix) Row(i int) []float64 {
	row := make([]float64, m.Cols)
	copy(row, m.Data[i])
	return row
}

// AddRow soma um vetor (1D) a cada linha da matrix (broadcast)
func (m *Matrix) AddRow(row []float64) *Matrix {
	result := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols && j < len(row); j++ {
			result.Data[i][j] = m.Data[i][j] + row[j]
		}
	}
	return result
}

// SoftmaxRows aplica softmax em cada linha independentemente
func (m *Matrix) SoftmaxRows() *Matrix {
	result := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		max := m.Data[i][0]
		for j := 1; j < m.Cols; j++ {
			if m.Data[i][j] > max {
				max = m.Data[i][j]
			}
		}
		sum := 0.0
		for j := 0; j < m.Cols; j++ {
			result.Data[i][j] = math.Exp(m.Data[i][j] - max)
			sum += result.Data[i][j]
		}
		if sum > 0 {
			for j := 0; j < m.Cols; j++ {
				result.Data[i][j] /= sum
			}
		}
	}
	return result
}

// Xavier retorna matrix com inicialização Xavier
func Xavier(rows, cols int) *Matrix {
	scale := math.Sqrt(2.0 / float64(rows+cols))
	m := New(rows, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			m.Data[i][j] = rand.NormFloat64() * scale
		}
	}
	return m
}

// MeanRows retorna a média de cada coluna [1 x cols]
func (m *Matrix) MeanRows() []float64 {
	means := make([]float64, m.Cols)
	for j := 0; j < m.Cols; j++ {
		sum := 0.0
		for i := 0; i < m.Rows; i++ {
			sum += m.Data[i][j]
		}
		means[j] = sum / float64(m.Rows)
	}
	return means
}

// Clone retorna uma cópia profunda da matrix
func (m *Matrix) Clone() *Matrix {
	result := New(m.Rows, m.Cols)
	for i := 0; i < m.Rows; i++ {
		copy(result.Data[i], m.Data[i])
	}
	return result
}
