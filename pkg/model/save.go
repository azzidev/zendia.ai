package model

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/azzidev/zendia.ai/pkg/matrix"
	"github.com/azzidev/zendia.ai/pkg/tokenizer"
)

type Checkpoint struct {
	Config       Config
	EmbedWeights [][]float64
	Blocks       []BlockCheckpoint
	FinalGamma   []float64
	FinalBeta    []float64
	LMHead       [][]float64
	LMBias       []float64
	TokenizerData TokenizerCheckpoint
}

type BlockCheckpoint struct {
	// attention heads
	Heads []HeadCheckpoint
	// attention output projection
	Wo [][]float64
	Bo []float64
	// FFN
	W1, W2 [][]float64
	B1, B2 []float64
	// layer norms
	Norm1Gamma, Norm1Beta []float64
	Norm2Gamma, Norm2Beta []float64
}

type HeadCheckpoint struct {
	Wq, Wk, Wv [][]float64
	Bq, Bk, Bv []float64
}

type TokenizerCheckpoint struct {
	Vocab    map[string]int
	InvVocab map[int]string
	Merges   []MergeCheckpoint
}

type MergeCheckpoint struct {
	A, B, Result string
}

func (m *LLM) Save(path string) error {
	cp := Checkpoint{
		Config:       m.Config,
		EmbedWeights: matrixToSlice2D(m.EmbedTable.Weights),
		FinalGamma:   copySlice(m.FinalNorm.Gamma),
		FinalBeta:    copySlice(m.FinalNorm.Beta),
		LMHead:       matrixToSlice2D(m.LMHead),
		LMBias:       copySlice(m.LMBias),
	}

	// tokenizer
	cp.TokenizerData.Vocab = m.Tokenizer.Vocab
	cp.TokenizerData.InvVocab = m.Tokenizer.InvVocab
	for _, merge := range m.Tokenizer.Merges {
		cp.TokenizerData.Merges = append(cp.TokenizerData.Merges, MergeCheckpoint{
			A: merge.A, B: merge.B, Result: merge.Result,
		})
	}

	// blocks
	for _, block := range m.Blocks {
		bc := BlockCheckpoint{
			Wo:         matrixToSlice2D(block.Attention.Wo),
			Bo:         copySlice(block.Attention.Bo),
			W1:         matrixToSlice2D(block.FFN.W1),
			W2:         matrixToSlice2D(block.FFN.W2),
			B1:         copySlice(block.FFN.B1),
			B2:         copySlice(block.FFN.B2),
			Norm1Gamma: copySlice(block.Norm1.Gamma),
			Norm1Beta:  copySlice(block.Norm1.Beta),
			Norm2Gamma: copySlice(block.Norm2.Gamma),
			Norm2Beta:  copySlice(block.Norm2.Beta),
		}

		for _, head := range block.Attention.Heads {
			bc.Heads = append(bc.Heads, HeadCheckpoint{
				Wq: matrixToSlice2D(head.Wq),
				Wk: matrixToSlice2D(head.Wk),
				Wv: matrixToSlice2D(head.Wv),
				Bq: copySlice(head.Bq),
				Bk: copySlice(head.Bk),
				Bv: copySlice(head.Bv),
			})
		}

		cp.Blocks = append(cp.Blocks, bc)
	}

	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("erro ao criar arquivo: %w", err)
	}
	defer file.Close()

	if err := gob.NewEncoder(file).Encode(cp); err != nil {
		return fmt.Errorf("erro ao salvar: %w", err)
	}

	info, _ := os.Stat(path)
	fmt.Printf("💾 Modelo salvo: %s (%.1f MB)\n", path, float64(info.Size())/1024/1024)
	return nil
}

func Load(path string) (*LLM, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("erro ao abrir: %w", err)
	}
	defer file.Close()

	var cp Checkpoint
	if err := gob.NewDecoder(file).Decode(&cp); err != nil {
		return nil, fmt.Errorf("erro ao carregar: %w", err)
	}

	// reconstrói tokenizer
	tok := &tokenizer.Tokenizer{
		Vocab:    cp.TokenizerData.Vocab,
		InvVocab: cp.TokenizerData.InvVocab,
	}
	for _, m := range cp.TokenizerData.Merges {
		tok.Merges = append(tok.Merges, tokenizer.Merge{A: m.A, B: m.B, Result: m.Result})
	}

	// reconstrói modelo
	llm := New(cp.Config, tok)

	// carrega pesos
	slice2DToMatrix(cp.EmbedWeights, llm.EmbedTable.Weights)
	slice2DToMatrix(cp.LMHead, llm.LMHead)
	copy(llm.LMBias, cp.LMBias)
	copy(llm.FinalNorm.Gamma, cp.FinalGamma)
	copy(llm.FinalNorm.Beta, cp.FinalBeta)

	for i, bc := range cp.Blocks {
		block := llm.Blocks[i]
		slice2DToMatrix(bc.Wo, block.Attention.Wo)
		copy(block.Attention.Bo, bc.Bo)
		slice2DToMatrix(bc.W1, block.FFN.W1)
		slice2DToMatrix(bc.W2, block.FFN.W2)
		copy(block.FFN.B1, bc.B1)
		copy(block.FFN.B2, bc.B2)
		copy(block.Norm1.Gamma, bc.Norm1Gamma)
		copy(block.Norm1.Beta, bc.Norm1Beta)
		copy(block.Norm2.Gamma, bc.Norm2Gamma)
		copy(block.Norm2.Beta, bc.Norm2Beta)

		for j, hc := range bc.Heads {
			head := block.Attention.Heads[j]
			slice2DToMatrix(hc.Wq, head.Wq)
			slice2DToMatrix(hc.Wk, head.Wk)
			slice2DToMatrix(hc.Wv, head.Wv)
			copy(head.Bq, hc.Bq)
			copy(head.Bk, hc.Bk)
			copy(head.Bv, hc.Bv)
		}
	}

	fmt.Printf("📂 Modelo carregado: %s\n", path)
	fmt.Printf("   %s\n", llm)
	return llm, nil
}

func matrixToSlice2D(m *matrix.Matrix) [][]float64 {
	data := make([][]float64, m.Rows)
	for i := range data {
		data[i] = make([]float64, m.Cols)
		copy(data[i], m.Data[i])
	}
	return data
}

func slice2DToMatrix(src [][]float64, dst *matrix.Matrix) {
	for i := range src {
		if i < dst.Rows {
			copy(dst.Data[i], src[i])
		}
	}
}

func copySlice(src []float64) []float64 {
	dst := make([]float64, len(src))
	copy(dst, src)
	return dst
}
