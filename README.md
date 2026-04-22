<p align="center">
  <img src="https://img.shields.io/badge/Go-1.21+-00ADD8?style=flat-square&logo=go" />
  <img src="https://img.shields.io/badge/LLM-From%20Scratch-6c5ce7?style=flat-square" />
  <img src="https://img.shields.io/badge/GPU-OpenCL-00cec9?style=flat-square" />
  <img src="https://img.shields.io/badge/Language-PT--BR-fdcb6e?style=flat-square" />
</p>

# 🧠 Zendia.AI

**Uma LLM (Large Language Model) construída inteiramente do zero em Go — sem PyTorch, sem TensorFlow, sem nenhuma lib de ML.** Cada operação matemática, cada neurônio, cada algoritmo foi implementado na mão.

O objetivo não é competir com o ChatGPT. É entender como uma IA funciona por dentro — do neurônio mais simples até um Transformer completo que gera texto em português.

---

## 🏗️ O que foi implementado do zero

```
Perceptron → MLP → Backpropagation → Tokenizer BPE → Embeddings →
Positional Encoding → Self-Attention → Multi-Head Attention →
Layer Normalization → GELU → Transformer Block → LLM → Geração de Texto
```

| Componente | Descrição |
|---|---|
| **Matrix Operations** | Multiplicação, transposição, Hadamard, softmax, Xavier init — tudo na mão |
| **Perceptron** | Neurônio único com treinamento |
| **MLP** | Rede multicamada com backpropagation completo |
| **Tokenizer BPE** | Byte Pair Encoding — mesmo algoritmo base do GPT |
| **Embeddings** | Tabela de embeddings + Positional Encoding (seno/cosseno) |
| **Self-Attention** | Scaled Dot-Product Attention com causal mask |
| **Multi-Head Attention** | Múltiplas heads paralelas com projeção de saída |
| **Transformer Block** | Pre-LN com residual connections (LayerNorm → Attention → FFN) |
| **GELU** | Ativação usada no GPT (implementado no transformer) |
| **Cross-Entropy Loss** | Com softmax estável (implementado no modelo) |
| **Adam Optimizer** | Adaptive Moment Estimation com bias correction (disponível, não integrado) |
| **Gradient Clipping** | Evita explosão de gradientes (disponível, não integrado) |
| **Dropout** | Regularização (disponível, não integrado) |
| **GPU (OpenCL)** | Aceleração via OpenCL — funciona com AMD, NVIDIA e Intel |
| **CPU Parallel** | Multiplicação de matrizes paralela com goroutines |
| **Save/Load** | Serialização completa do modelo (gob encoding) |
| **Tool Calling** | Execução de comandos, calculadora, hora, info do sistema |

---

## 📁 Estrutura do Projeto

```
zendia.ai/
├── cmd/
│   └── main.go                 # entry point — pipeline completo
├── pkg/
│   ├── matrix/                 # operações com matrizes + paralelismo
│   │   ├── matrix.go
│   │   ├── matmul.go           # MatMul global (GPU/CPU)
│   │   └── parallel.go         # multiplicação paralela com goroutines
│   ├── activation/             # sigmoid, relu, step
│   ├── loss/                   # MSE
│   ├── optimizer/              # Adam + gradient clipping
│   ├── neuron/                 # perceptron
│   ├── network/                # MLP + backprop
│   ├── tokenizer/              # BPE tokenizer
│   ├── embedding/              # embeddings + positional encoding
│   ├── attention/              # self-attention + multi-head
│   ├── transformer/            # transformer block + layer norm + dropout + GELU
│   ├── model/                  # LLM completa + save/load
│   ├── gpu/                    # OpenCL via CGo
│   ├── datasource/             # Wikipedia + OpenSubtitles → MongoDB
│   ├── dialogue/               # dataset de diálogos + fine-tuner
│   ├── chat/                   # interface web de chat
│   ├── tools/                  # tool calling (hora, calc, cmd)
│   └── visualizer/             # servidor WebSocket
├── web/                        # front-end separado
│   ├── index.html
│   ├── chat.html
│   ├── css/
│   │   ├── style.css
│   │   └── chat.css
│   └── js/
│       ├── app.js              # inicialização
│       ├── network3d.js        # visualização 3D (Three.js)
│       ├── heatmap.js          # heatmap de ativações
│       ├── chart.js            # gráfico de loss
│       └── websocket.js        # conexão com servidor
└── datasets/                   # dumps e modelos (não versionado)
```

---

## 🚀 Como Rodar

### Pré-requisitos

- **Go 1.21+** — https://go.dev/dl/
- **MongoDB** rodando local na porta 27017
- **GCC** (MinGW no Windows) — necessário pra GPU via OpenCL
  - Windows: https://github.com/niXman/mingw-builds-binaries/releases
  - Linux: `sudo apt install gcc`
  - macOS: `xcode-select --install`

### Setup

```bash
git clone https://github.com/azzidev/zendia.ai.git
cd zendia.ai
go mod tidy
```

### Headers do OpenCL (necessário pra GPU)

```bash
git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers.git pkg/gpu/opencl/headers
```

No Windows, gere a lib do OpenCL:
```bash
cd pkg/gpu
gendef C:\Windows\System32\OpenCL.dll
dlltool -d OpenCL.def -l libOpenCL.a
cd ../..
```

### Rodar tudo (pipeline completo)

```bash
# Windows
set "PATH=C:\mingw64\bin;%PATH%" && set "CGO_ENABLED=1" && go run -buildvcs=false ./cmd/main.go -mode=all

# Linux/macOS
CGO_ENABLED=1 go run ./cmd/main.go -mode=all
```

Isso vai:
1. Baixar Wikipedia PT-BR (~2.5 GB)
2. Baixar OpenSubtitles PT-BR (~2 GB)
3. Processar e inserir no MongoDB
4. Treinar o modelo base
5. Fine-tunar com diálogos
6. Abrir o chat

### Rodar passo a passo

```bash
# Windows: sempre prefixe com:
# set "PATH=C:\mingw64\bin;%PATH%" && set "CGO_ENABLED=1" &&

# 1. Baixar Wikipedia PT-BR → MongoDB
go run -buildvcs=false ./cmd/main.go -mode=download

# 2. Baixar diálogos OpenSubtitles PT-BR → MongoDB
go run -buildvcs=false ./cmd/main.go -mode=download-dialogues

# 3. Treinar modelo base (aprende português)
go run -buildvcs=false ./cmd/main.go -mode=train -epochs=20 -model=datasets/zendia-64x2.gob

# 4. Fine-tuning com diálogos (aprende a conversar)
go run -buildvcs=false ./cmd/main.go -mode=finetune -ft-epochs=10 -model=datasets/zendia-64x2.gob

# 5. Chat
go run -buildvcs=false ./cmd/main.go -mode=chat -model=datasets/zendia-64x2.gob
```

### Flags disponíveis

| Flag | Default | Descrição |
|---|---|---|
| `-mode` | `all` | `download`, `download-dialogues`, `train`, `finetune`, `chat`, `all` |
| `-model` | `datasets/zendia-model.gob` | Caminho do modelo (save/load) |
| `-mongo` | `mongodb://localhost:27017` | URI do MongoDB |
| `-epochs` | `5` | Epochs de treinamento base |
| `-ft-epochs` | `5` | Epochs de fine-tuning |
| `-max-articles` | `10000` | Máximo de artigos da Wikipedia |

### Continuar treinamento de onde parou

O modelo salva checkpoint a cada epoch. Pra continuar:

```bash
go run -buildvcs=false ./cmd/main.go -mode=train -epochs=10 -model=datasets/zendia-64x2.gob
```

Se o arquivo existir, carrega e continua. Se não, cria do zero.

---

## 📊 Datasets

### Wikipedia PT-BR
- **URL:** https://dumps.wikimedia.org/ptwiki/latest/ptwiki-latest-pages-articles.xml.bz2
- **Tamanho:** ~2.5 GB compactado
- **Conteúdo:** Todos os artigos da Wikipedia em português
- **Uso:** Treinamento base — o modelo aprende a estrutura do português

### OpenSubtitles PT-BR
- **URL:** https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/pt_br.txt.gz
- **Tamanho:** ~2 GB compactado
- **Conteúdo:** Legendas de filmes e séries em português brasileiro
- **Uso:** Fine-tuning — o modelo aprende padrões de diálogo

---

## 🧠 Modelos

### zendia-256x6 (grande)
```
EmbedDim: 256 | Layers: 6 | Heads: 8 | FFN: 512
MaxSeqLen: 128 | Params: ~5-8M
Tokenizer: 3000 merges
```
- Precisa de GPU ou CPU potente
- ~65 tok/s em CPU (AMD Radeon iGPU com OpenCL)
- Melhor qualidade de texto

### zendia-64x2 (leve)
```
EmbedDim: 64 | Layers: 2 | Heads: 4 | FFN: 128
MaxSeqLen: 64 | Params: ~214K
Tokenizer: 2000 merges
```
- Roda em qualquer PC
- ~270 tok/s em CPU
- Bom pra aprender e experimentar

---

## 🎮 Visualização

Acesse `http://localhost:8080` durante o treinamento pra ver:

- **Rede neural 3D** — cada camada como plano de partículas (Three.js)
- **Heatmap de ativações** — cores mostram intensidade de cada neurônio
- **Gráfico de loss** — acompanhe o aprendizado em tempo real
- **Métricas** — epoch, step, loss, tokens processados
- **Amostras** — texto gerado pelo modelo durante o treino

Controles:
- **Mouse esquerdo** = rotacionar
- **Mouse direito** = mover câmera
- **Scroll** = zoom

---

## 💬 Chat

Depois de treinar e fine-tunar, rode:

```bash
go run -buildvcs=false ./cmd/main.go -mode=chat -model=datasets/zendia-64x2.gob
```

Abra `http://localhost:8080/chat.html` e converse. O chat também tem **tool calling**:

| Comando | O que faz |
|---|---|
| "que horas são?" | Retorna hora atual |
| "que dia é hoje?" | Retorna data |
| "info do sistema" | Mostra OS, CPU, etc |
| "calcule 2+2*3" | Calcula expressão |
| "execute dir" | Executa comando do sistema |

---

## 📈 Entendendo o Loss

| Loss | Significado |
|---|---|
| ~7.0 | Chutando aleatório (início) |
| ~6.0 | Aprendendo padrões básicos |
| ~5.0 | Formando palavras reais |
| ~4.0 | Frases com estrutura |
| ~3.0 | Texto coerente |
| < 2.0 | Texto bom (difícil com modelo pequeno) |

---

## 🔧 Arquitetura do Transformer

```
Input tokens
    ↓
[Embedding Table] → vetor por token
    ↓
[Positional Encoding] → adiciona info de posição
    ↓
┌─────────────────────────────────┐
│  Transformer Block (x N)        │
│                                 │
│  LayerNorm → Multi-Head Attn    │
│      ↓ + residual               │
│  LayerNorm → FeedForward (GELU) │
│      ↓ + residual               │
└─────────────────────────────────┘
    ↓
[Final LayerNorm]
    ↓
[LM Head] → logits (probabilidade de cada token)
    ↓
[Softmax + Sampling] → próximo token
```

---

## 🤝 Contribuindo

O projeto é educacional. Se quiser contribuir:

1. Fork o repo
2. Crie uma branch (`git checkout -b feature/minha-feature`)
3. Commit (`git commit -m 'adiciona minha feature'`)
4. Push (`git push origin feature/minha-feature`)
5. Abra um PR

Ideias de contribuição:
- Suporte a CUDA (NVIDIA)
- Integrar Adam optimizer no treino
- Integrar Dropout nos transformer blocks
- Batch training
- Learning rate scheduler
- Beam search na geração
- Mais datasets em PT-BR
- Testes unitários
- Benchmark CPU vs GPU

---

## 📝 Licença

MIT — use como quiser.

---

<p align="center">
  <b>Feito do zero, sem atalhos, pra entender como IA funciona de verdade.</b>
</p>
