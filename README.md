# 🏥 Tech Challenge — Fase 1 (Projeto B)
## Sistema Inteligente de Suporte ao Diagnóstico — Hospital Universitário

> Projeto desenvolvido para o **PosTech** como parte do Tech Challenge da Fase 1.  
> Aplicação de Machine Learning para classificação de exames médicos, com separação em treino, validação e teste.

---

## 📋 Índice

- [Sobre o Projeto](#sobre-o-projeto)
- [Dataset](#dataset)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação](#instalação)
- [Como Executar](#como-executar)
- [Divisão dos Dados](#divisão-dos-dados)
- [Modelos Utilizados](#modelos-utilizados)
- [Métricas e Resultados](#métricas-e-resultados)
- [Explicabilidade](#explicabilidade)
- [Discussão Crítica](#discussão-crítica)

---

## 📌 Sobre o Projeto

Um grande hospital universitário precisa de um sistema inteligente capaz de **apoiar médicos e equipes clínicas na análise inicial de exames**, acelerando a triagem e reduzindo erros.

Esta fase implementa a **base do sistema de IA com foco em Machine Learning**, classificando automaticamente resultados de exames para identificar casos de risco.

---

## 📊 Dataset

**Breast Cancer Wisconsin Diagnostic Dataset**

| Atributo | Valor |
|---|---|
| Fonte | UCI Machine Learning Repository / Kaggle |
| Link | [kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) |
| Amostras | 569 pacientes |
| Features | 30 atributos numéricos de biópsias |
| Classes | Maligno (212) / Benigno (357) |
| Valores ausentes | Nenhum |

> O dataset está disponível diretamente via `sklearn.datasets.load_breast_cancer()`.

---

## 🗂️ Estrutura do Projeto

```
tech-challenge-fase1-b/
│
├── tech_challenge_b.ipynb   # Notebook principal
├── dashboard_b.py           # Dashboard visual em localhost
├── README.md                # Este arquivo
│
└── outputs/                 # Gerados ao rodar o notebook
    ├── distribuicao_diagnosticos.png
    ├── distribuicao_features.png
    ├── correlacao.png
    ├── matrizes_confusao.png
    ├── comparacao_modelos.png
    ├── feature_importance.png
    ├── shap_summary.png
    └── shap_beeswarm.png
```

---

## 🚀 Instalação

pip install pandas numpy matplotlib seaborn scikit-learn shap jupyter
```

---

## ▶️ Como Executar

### Notebook completo
```bash
jupyter notebook tech_challenge_b.ipynb
```
Execute as células com **Shift+Enter** ou clique em "Run All".

### No VS Code
Abra o arquivo `.ipynb` e selecione o kernel **Python 3**.

### Dashboard visual
```bash
python dashboard_b.py
```
Abre automaticamente em `http://localhost:8000/dashboard_b.html`.  
Encerre com **Ctrl+C**.

---

## 🔀 Divisão dos Dados

Este projeto utiliza **três conjuntos**, seguindo boas práticas de ML:

| Conjunto | Proporção | Amostras | Finalidade |
|---|---|---|---|
| **Treino** | 70% | ~398 | O modelo aprende com esses dados |
| **Validação** | 15% | ~85 | Comparar e ajustar modelos durante o desenvolvimento |
| **Teste** | 15% | ~86 | Avaliação final — usado apenas uma vez |

A separação usa `stratify=y` para garantir proporção igual de classes em todos os conjuntos.

### Por que 3 conjuntos?
O conjunto de **validação** permite comparar modelos e ajustar hiperparâmetros sem "vazar" informações do teste. O conjunto de **teste** simula dados reais que o modelo nunca viu, garantindo uma avaliação honesta e imparcial.

---

## 🤖 Modelos Utilizados

### Regressão Logística
Modelo linear para classificação binária. Escolhido por interpretabilidade, boa performance em dados médicos correlacionados e compatibilidade com SHAP.

**Configuração:** `max_iter=1000`, `random_state=42`. Requer normalização com StandardScaler.

### Árvore de Decisão
Modelo baseado em regras de decisão em formato de fluxograma. Escolhido pela interpretabilidade visual e Feature Importance nativa.

**Configuração:** `max_depth=5` para controlar overfitting, `random_state=42`.

---

## 📈 Métricas e Resultados

### Métrica prioritária: Recall (classe Maligno)

Em diagnóstico médico, os erros têm custos diferentes:

- **Falso Negativo** (maligno → benigno): paciente não recebe tratamento → **consequência grave**
- **Falso Positivo** (benigno → maligno): paciente realiza exames adicionais → **consequência aceitável**

Por isso **maximizar o Recall da classe Maligno** é a estratégia mais segura.

### Resultados (Conjunto de Teste)

| Modelo | Acurácia | Recall (Maligno) | F1-Score (Maligno) |
|---|---|---|---|
| Regressão Logística | ~97% | ~97% | ~96% |
| Árvore de Decisão | ~93% | ~90% | ~91% |

> Os valores exatos são exibidos no notebook e no dashboard ao executar.

---

## 🔍 Explicabilidade

### Feature Importance (Árvore de Decisão)
Indica globalmente quais features mais influenciam as decisões do modelo.

### SHAP Values (Regressão Logística)
Explica a contribuição de cada feature para cada predição individual:
- Valores positivos → aumentam probabilidade de Benigno
- Valores negativos → aumentam probabilidade de Maligno

Features mais determinantes: `worst concave points`, `worst perimeter`, `mean concave points`.

---

## ⚠️ Discussão Crítica

### O modelo pode ser utilizado na prática?

**Sim, como ferramenta de apoio — com ressalvas:**

✅ Adequado para triagem inicial e priorização de casos urgentes  
✅ Útil como segunda opinião computacional em análises de rotina  
❌ Não substitui o julgamento clínico  
❌ Precisa de validação em dados externos antes de qualquer implantação  
❌ Requer auditoria regular para detecção de viés  

> **O médico sempre deve ter a palavra final no diagnóstico.**

---

## 📦 Dependências

| Biblioteca | Uso |
|---|---|
| pandas | Manipulação de dados |
| numpy | Computação numérica |
| matplotlib + seaborn | Visualizações |
| scikit-learn | Modelos, métricas e pré-processamento |
| shap | Explicabilidade |
| jupyter | Ambiente de execução |

---

*Desenvolvido com Python 🐍 · scikit-learn · SHAP · PosTech 2025*
