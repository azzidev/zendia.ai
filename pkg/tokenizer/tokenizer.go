package tokenizer

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

// Tokenizer usando Byte Pair Encoding (BPE)
// Mesmo algoritmo base do GPT — aprende merges a partir do corpus
type Tokenizer struct {
	Vocab    map[string]int // token → id
	InvVocab map[int]string // id → token
	Merges   []Merge
	nextID   int
}

type Merge struct {
	A, B   string
	Result string
}

func New() *Tokenizer {
	t := &Tokenizer{
		Vocab:    make(map[string]int),
		InvVocab: make(map[int]string),
	}

	// tokens especiais
	t.addToken("<pad>")
	t.addToken("<unk>")
	t.addToken("<bos>")
	t.addToken("<eos>")

	// inicializa com todos os caracteres UTF-8 básicos + acentos PT-BR
	chars := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	chars += "áàâãéèêíìîóòôõúùûçÁÀÂÃÉÈÊÍÌÎÓÒÔÕÚÙÛÇ"
	chars += " .,!?;:()-\"'@#$%&*/\\+={[]}<>|~^`\n\t"

	for _, ch := range chars {
		t.addToken(string(ch))
	}

	return t
}

func (t *Tokenizer) addToken(token string) int {
	if id, ok := t.Vocab[token]; ok {
		return id
	}
	id := t.nextID
	t.Vocab[token] = id
	t.InvVocab[id] = token
	t.nextID++
	return id
}

// AddSpecialToken registra um token especial e retorna seu ID
func (t *Tokenizer) AddSpecialToken(token string) int {
	return t.addToken(token)
}

func (t *Tokenizer) VocabSize() int {
	return t.nextID
}

// Train aprende merges BPE a partir de um corpus de texto
func (t *Tokenizer) Train(texts []string, numMerges int) {
	// split cada palavra em caracteres individuais
	wordFreqs := make(map[string]int)
	for _, text := range texts {
		words := strings.Fields(strings.ToLower(text))
		for _, word := range words {
			// cada palavra vira sequência de chars separados por espaço + marcador de fim
			chars := splitChars(word)
			key := strings.Join(chars, " ") + " </w>"
			wordFreqs[key]++
		}
	}

	for i := 0; i < numMerges; i++ {
		// conta pares adjacentes
		pairs := countPairs(wordFreqs)
		if len(pairs) == 0 {
			break
		}

		// acha o par mais frequente
		bestPair := bestPair(pairs)
		if pairs[bestPair] < 2 {
			break
		}

		merged := bestPair.a + bestPair.b
		t.Merges = append(t.Merges, Merge{A: bestPair.a, B: bestPair.b, Result: merged})
		t.addToken(merged)

		// aplica o merge em todas as palavras
		wordFreqs = applyMerge(wordFreqs, bestPair)
	}
}

// Encode transforma texto em sequência de IDs
func (t *Tokenizer) Encode(text string) []int {
	text = strings.ToLower(text)
	words := strings.Fields(text)
	var ids []int

	ids = append(ids, t.Vocab["<bos>"])

	for _, word := range words {
		tokens := splitChars(word)
		tokens = append(tokens, "</w>")

		// aplica merges aprendidos
		for _, merge := range t.Merges {
			tokens = applyMergeToTokens(tokens, merge)
		}

		for _, tok := range tokens {
			if tok == "</w>" {
				continue
			}
			if id, ok := t.Vocab[tok]; ok {
				ids = append(ids, id)
			} else {
				ids = append(ids, t.Vocab["<unk>"])
			}
		}
	}

	ids = append(ids, t.Vocab["<eos>"])
	return ids
}

// Decode transforma IDs de volta em texto
func (t *Tokenizer) Decode(ids []int) string {
	var parts []string
	for _, id := range ids {
		if token, ok := t.InvVocab[id]; ok {
			if token == "<bos>" || token == "<eos>" || token == "<pad>" {
				continue
			}
			parts = append(parts, token)
		}
	}
	result := strings.Join(parts, "")
	result = strings.ReplaceAll(result, "</w>", " ")
	return strings.TrimSpace(result)
}

func (t *Tokenizer) TokenToString(id int) string {
	if s, ok := t.InvVocab[id]; ok {
		return s
	}
	return "<unk>"
}

func (t *Tokenizer) String() string {
	return fmt.Sprintf("Tokenizer(vocab=%d, merges=%d)", t.VocabSize(), len(t.Merges))
}

// --- funções internas ---

func splitChars(word string) []string {
	var chars []string
	for i := 0; i < len(word); {
		r, size := utf8.DecodeRuneInString(word[i:])
		chars = append(chars, string(r))
		i += size
	}
	return chars
}

type pair struct{ a, b string }

func countPairs(wordFreqs map[string]int) map[pair]int {
	counts := make(map[pair]int)
	for word, freq := range wordFreqs {
		tokens := strings.Split(word, " ")
		for i := 0; i < len(tokens)-1; i++ {
			p := pair{tokens[i], tokens[i+1]}
			counts[p] += freq
		}
	}
	return counts
}

func bestPair(pairs map[pair]int) pair {
	type kv struct {
		p pair
		c int
	}
	var sorted []kv
	for p, c := range pairs {
		sorted = append(sorted, kv{p, c})
	}
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].c > sorted[j].c
	})
	return sorted[0].p
}

func applyMerge(wordFreqs map[string]int, p pair) map[string]int {
	result := make(map[string]int)
	merged := p.a + " " + p.b
	replacement := p.a + p.b

	for word, freq := range wordFreqs {
		newWord := strings.ReplaceAll(word, merged, replacement)
		result[newWord] += freq
	}
	return result
}

func applyMergeToTokens(tokens []string, merge Merge) []string {
	var result []string
	i := 0
	for i < len(tokens) {
		if i < len(tokens)-1 && tokens[i] == merge.A && tokens[i+1] == merge.B {
			result = append(result, merge.Result)
			i += 2
		} else {
			result = append(result, tokens[i])
			i++
		}
	}
	return result
}
