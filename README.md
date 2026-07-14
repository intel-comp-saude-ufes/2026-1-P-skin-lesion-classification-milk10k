# 2026-1-P-skin-lesion-classification-milk10k

Este repositório contém código para experimentos de classificação de lesões de pele usando o dataset Milk10k (imagens e metadados integrados em `milk10k_merged.csv`).

**Funcionalidade**
- **Treinamento**: scripts e funções para treinar modelos de classificação (`treino.py`, `main.py`, `loop_treino.py`, `modelo.py`).
- **Pré-processamento**: carregamento e transformações de imagens e rótulos (`carregar_dados.py`, `dataloaders.py`, `transformacoes.py`, `preparar_rotulos.py`).
- **Avaliação**: ferramentas para avaliação do modelo e análise de resultados (`avaliacao_final.py`, `teste_estatistico.py`, `matriz_confusao.py`).
- **Utilitários**: metadados e cálculo de pesos por classe (`metadados.py`, `pesos_classe.py`).

**Dependências**
Instale dependências do requirements.txt:
`pip install -r requirements.txt`

**Estrutura principal**
- [main.py](main.py): ponto de entrada geral/experimentos.
- [treino.py](treino.py): fluxo de treino usado nos experimentos.
- [modelo.py](modelo.py): definições de modelos.
- [loop_treino.py](loop_treino.py): loop de treino/validação.
- [carregar_dados.py](carregar_dados.py): carregamento do dataset e separação entre treino e teste.
- [transformacoes.py](transformacoes.py): transformações de imagem usadas no treino/teste.
- [preparar_rotulos.py](preparar_rotulos.py): scripts para preparar/normalizar rótulos.
- [avaliacao_final.py](avaliacao_final.py), [matriz_confusao.py](matriz_confusao.py), [teste_estatistico.py](teste_estatistico.py): avaliação e métricas.
- [milk10k_merged.csv](milk10k_merged.csv): arquivo CSV com metadados integrados.
- [imagens/](imagens/): diretório com subpastas de imagens do dataset.

**Como executar (exemplos)**
Baixe as imagens do dataset por este link https://isic-archive.s3.amazonaws.com/challenges/milk10k/MILK10k_Training_Input.zip, as imagens devem estar organizadas da seguinte forma, `id_lesao/imagem_dermatológica, imagem_clínica`. Uma pasta da lesão possui os dois tipos de imagem daquela lesão. Renomeie essa pasta para `imagens`, as pastas das lesões devem ficar na raiz dessa pasta imagens.

Rodar testes / avaliação:

```bash
./rodar_testes.sh
./rodar_testes_resnet18.sh
```


**Observações**
- Verifique caminhos relativos a `imagens/` e `milk10k_merged.csv` antes de rodar os scripts.
