#!/bin/bash
# rodar_testes_resnet18.sh
#
# Mesma logica do rodar_testes.sh, mas pra ResNet18. A DIFERENCA principal:
# a GPU (RTX 3060, 12 GiB) so aguenta 1 treino pesado por vez - rodar as
# duas rodadas (resnet50 e resnet18) ao mesmo tempo arrisca CUDA out of
# memory ou deixar as duas extremamente lentas. Por isso esse script
# PRIMEIRO espera qualquer "python main.py ..." da rodada de resnet50
# (a que voce ja deixou rodando) terminar, e SO DEPOIS comeca a rodada
# de resnet18 - assim voce pode disparar isso agora, sem esperar do lado,
# que ele mesmo aguarda a vez certa.
#
# COMO RODAR (mesma logica de antes, protegendo contra queda de SSH):
#
#   nohup bash rodar_testes_resnet18.sh > rodar_testes_resnet18_full.log 2>&1 &
#   disown
#
# Ou, se ja tiver tmux instalado:
#   tmux new -s treino_resnet18
#   bash rodar_testes_resnet18.sh
#   (Ctrl+B, D pra desgrudar)

BACKBONE="resnet18"
K_FOLDS=7   # mesma janela segura de antes: >=6 (Wilcoxon) e <=8 (limite da classe MAL_OTH)
LOG_DIR="logs_overnight_${BACKBONE}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

TESTES=("sem_metadado" "metablock" "tripleblock")

# ----------------------------- ESPERA A GPU FICAR LIVRE -----------------------------
# Considera "ocupado" qualquer processo "python main.py ..." que NAO seja
# desse proprio script (ou seja, que nao tenha "resnet18" na linha de
# comando) - isso cobre a rodada de resnet50 que ja estava rodando
# (chamadas como "python main.py sem_metadado 7", sem 3o argumento).
echo "===== $(date): checando se a rodada de ResNet50 ainda esta rodando... ====="
while ps aux | grep "python main.py" | grep -v grep | grep -v "resnet18" > /dev/null; do
    echo "  ainda ocupado ($(date)) - esperando 60s..."
    sleep 60
done
echo "===== $(date): GPU livre - comecando rodada ResNet18 ====="

for teste in "${TESTES[@]}"; do
    echo ""
    echo "----- Rodando: $teste ($BACKBONE) (inicio $(date)) -----"
    python main.py "$teste" "$K_FOLDS" "$BACKBONE" > "$LOG_DIR/${teste}.log" 2>&1
    status=$?
    if [ $status -eq 0 ]; then
        echo "----- OK: $teste ($BACKBONE) terminou (fim $(date)) -----"
    else
        echo "----- ERRO: $teste ($BACKBONE) falhou (codigo $status) - confira $LOG_DIR/${teste}.log -----"
    fi
done

echo ""
echo "----- Rodando comparacao estatistica ($BACKBONE) -----"
python teste_estatistico.py "$BACKBONE" > "$LOG_DIR/comparacao_estatistica.log" 2>&1
echo "----- Comparacao estatistica salva em $LOG_DIR/comparacao_estatistica.log -----"
cat "$LOG_DIR/comparacao_estatistica.log"

echo ""
echo "===== FIM ($BACKBONE): $(date) ====="
echo "Resumo dos logs em: $LOG_DIR/"
