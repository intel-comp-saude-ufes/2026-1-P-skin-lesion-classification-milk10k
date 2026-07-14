#!/bin/bash
# rodar_testes.sh
#
# Roda os 3 testes (sem_metadado, metablock, tripleblock) em sequencia -
# um so comeca quando o anterior termina, ja que os 3 disputam a mesma
# GPU (RTX 3060, 12 GiB) - e ao final roda a comparacao estatistica
# (teste_estatistico.py) automaticamente, pra tudo estar pronto quando
# voce acordar.
#
# NAO usa "set -e": se um dos 3 testes falhar (ex.: CUDA out of memory
# no meio da madrugada), o script CONTINUA pros proximos em vez de parar
# tudo - melhor achar 2 de 3 prontos de manha do que 0 de 3.
#
# COMO RODAR (escolha uma opcao, pra sobreviver a queda da conexao SSH):
#
#   Opcao A - tmux (recomendado, da pra "colar" de volta na sessao):
#     tmux new -s treino_noite
#     conda activate gandalf
#     bash rodar_testes.sh
#     (aperte Ctrl+B e depois D pra "desgrudar" sem matar o processo)
#     -> amanha: tmux attach -t treino_noite
#
#   Opcao B - nohup (se nao tiver tmux instalado):
#     conda activate gandalf
#     nohup bash rodar_testes.sh > rodar_testes_full.log 2>&1 &
#     disown
#     -> amanha: cat rodar_testes_full.log (ou tail -f pra acompanhar ao vivo)

K_FOLDS=7   # janela segura: >=6 (Wilcoxon) e <=8 (limite da classe MAL_OTH, que tem so 8 no treino)
LOG_DIR="logs_overnight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

TESTES=("sem_metadado" "metablock" "tripleblock")

echo "===== INICIO: $(date) ====="
echo "K_FOLDS=$K_FOLDS | logs em: $LOG_DIR/"

for teste in "${TESTES[@]}"; do
    echo ""
    echo "----- Rodando: $teste (inicio $(date)) -----"
    python main.py "$teste" "$K_FOLDS" > "$LOG_DIR/${teste}.log" 2>&1
    status=$?
    if [ $status -eq 0 ]; then
        echo "----- OK: $teste terminou (fim $(date)) -----"
    else
        echo "----- ERRO: $teste falhou (codigo $status) - confira $LOG_DIR/${teste}.log -----"
    fi
done

echo ""
echo "----- Rodando comparacao estatistica (Friedman + Wilcoxon) -----"
python teste_estatistico.py > "$LOG_DIR/comparacao_estatistica.log" 2>&1
echo "----- Comparacao estatistica salva em $LOG_DIR/comparacao_estatistica.log -----"
cat "$LOG_DIR/comparacao_estatistica.log"

echo ""
echo "===== FIM: $(date) ====="
echo "Resumo dos logs em: $LOG_DIR/"
