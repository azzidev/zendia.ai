package matrix

import (
	"runtime"
	"sync"
)

var numWorkers = runtime.NumCPU()

func init() {
	runtime.GOMAXPROCS(numWorkers)
}

// MulParallel multiplica matrizes usando goroutines
// Cada goroutine calcula um bloco de linhas do resultado
func (m *Matrix) MulParallel(other *Matrix) (*Matrix, error) {
	if m.Cols != other.Rows {
		return m.Mul(other)
	}

	// pra matrizes muito pequenas, não vale paralelizar
	if m.Rows*other.Cols < 128 {
		return m.Mul(other)
	}

	result := New(m.Rows, other.Cols)
	var wg sync.WaitGroup

	rowsPerWorker := (m.Rows + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if endRow > m.Rows {
			endRow = m.Rows
		}
		if startRow >= endRow {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				for j := 0; j < other.Cols; j++ {
					sum := 0.0
					for k := 0; k < m.Cols; k++ {
						sum += m.Data[i][k] * other.Data[k][j]
					}
					result.Data[i][j] = sum
				}
			}
		}(startRow, endRow)
	}

	wg.Wait()
	return result, nil
}

// ApplyParallel aplica função em paralelo
func (m *Matrix) ApplyParallel(fn func(float64) float64) *Matrix {
	if m.Rows < 64 {
		return m.Apply(fn)
	}

	result := New(m.Rows, m.Cols)
	var wg sync.WaitGroup

	rowsPerWorker := (m.Rows + numWorkers - 1) / numWorkers

	for w := 0; w < numWorkers; w++ {
		startRow := w * rowsPerWorker
		endRow := startRow + rowsPerWorker
		if endRow > m.Rows {
			endRow = m.Rows
		}
		if startRow >= endRow {
			break
		}

		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				for j := 0; j < m.Cols; j++ {
					result.Data[i][j] = fn(m.Data[i][j])
				}
			}
		}(startRow, endRow)
	}

	wg.Wait()
	return result
}
