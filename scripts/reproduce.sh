#!/bin/bash
# The experiment sequence of "The Schedule Is the Signal", start to finish.
#
# Run from the repository root:   bash scripts/reproduce.sh
#
# Data is generated under data*/ and every metrics file is copied to results/.
# Each step is skipped when its results file already exists, so the script can
# be stopped and restarted. Seeds 42, 7 and 123 are the three seeds of the
# paper; single-seed steps use 42. The Chronos steps need chronos-forecasting
# (and peft for the fine-tuning); the fine-tuning is skipped without a GPU.
set -e -o pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/src"; RES="$ROOT/results"; mkdir -p "$RES"
cd "$ROOT"
export AIRPORT_LAYOUT=mco
export PYTHONPATH="$SRC:$PYTHONPATH"

have()  { [ -f "$RES/$1" ]; }
step()  { echo; echo "=== $(date +%H:%M) $1 ==="; }
save()  { for f in "$@"; do [ -f "$f" ] && cp "$f" "$RES/" || true; done; }
cpn()   { [ -f "$2" ] || cp "$1" "$2"; }
suf()   { [ "$1" = "42" ] && echo "" || echo "_seed$1"; }
GPU=$(python -c "import torch; print(int(torch.cuda.is_available()))")

# ---------------------------------------------------------------- datasets
gen() { local dir=$1; shift; [ -f "$dir/passengers.csv" ] || { step "dataset $dir"; python "$SRC/generate_mco.py" --out "$dir" --days 90 "$@"; }; }
gen data
gen data_dis --delay-frac 0.3 --gate-change-frac 0.05

# tensor NAME DIR [build_features options]
tensor() { local name=$1 dir=$2; shift 2
  [ -f "$dir/$name.npz" ] || python "$SRC/build_features.py" --passengers "$dir/passengers.csv" --flights "$dir/flights.csv" "$@" --out "$dir/$name.npz"; }
B=tensors_mco_1h_h3;         tensor $B  data --freq 1h    --in-seq 12 --tar-seq 3
B15=tensors_mco_15min;       tensor $B15 data --freq 15min --in-seq 12 --tar-seq 12
BE=tensors_mco_1h_h3_exppax; tensor $BE data --freq 1h    --in-seq 12 --tar-seq 3 --pax expected

# ---------------------------------------------------------------- main grid (Table 2)
for s in 42 7 123; do S=$(suf $s)
  have ${B}_results$S.json   || { step "hourly grid, seed $s";    python "$SRC/train_models.py" --data data/$B.npz   --spatials none,gcn,gat --temporals lstm,transformer,tdn --seed $s; save data/${B}_results$S.json data/${B}_preds$S.npz; }
  have ${B15}_results$S.json || { step "15-minute grid, seed $s"; python "$SRC/train_models.py" --data data/$B15.npz --spatials none,gcn,gat --temporals lstm,transformer,tdn --seed $s; save data/${B15}_results$S.json; }
done

# ---------------------------------------------------------------- foundation models (Table 2, Section 5.2)
chronos() { local D=$1 stem=$2 lora=${3:-lora}
  have ${stem}_chronos.json        || { step "Chronos-Bolt zero-shot: $stem";           python "$SRC/baseline_chronos.py"  --data $D && save data/${stem}_chronos.json; }
  have ${stem}_chronos2.json       || { step "Chronos-2 zero-shot + covariates: $stem"; python "$SRC/baseline_chronos2.py" --data $D && save data/${stem}_chronos2.json data/${stem}_chronos2_preds.npz; }
  have ${stem}_chronos2_nocov.json || { step "Chronos-2 zero-shot, no covariates: $stem"; python "$SRC/baseline_chronos2.py" --data $D --no-cov && save data/${stem}_chronos2_nocov.json; }
  if [ "$GPU" = "1" ] && [ "$lora" = "lora" ]; then
    have ${stem}_chronos2_lora.json       || { step "Chronos-2 LoRA + covariates: $stem"; python "$SRC/finetune_chronos2.py" --data $D --out-dir ft_${stem}_cov   && save data/${stem}_chronos2_lora.json data/${stem}_chronos2_lora_preds.npz; }
    have ${stem}_chronos2_lora_nocov.json || { step "Chronos-2 LoRA, no covariates: $stem"; python "$SRC/finetune_chronos2.py" --data $D --out-dir ft_${stem}_nocov --no-covariates && save data/${stem}_chronos2_lora_nocov.json; }
  fi; }
chronos data/$B.npz $B
chronos data/$B15.npz $B15

# ---------------------------------------------------------------- graph diagnostic (Table 3)
cpn data/$B.npz data/${B}_graphdiag.npz; cpn data/$B.npz data/${B}_ops.npz
for s in 42 7 123; do S=$(suf $s)
  have ${B}_graphdiag_results$S.json || { step "isolated nodes and shuffled GCN, seed $s"; python "$SRC/train_models.py" --data data/${B}_graphdiag.npz --spatials isolated,gcn_shuffled --temporals lstm,tdn --seed $s; save data/${B}_graphdiag_results$S.json; }
  have ${B}_ops_results$S.json       || { step "diffusion and Chebyshev operators, seed $s"; python "$SRC/train_models.py" --data data/${B}_ops.npz --spatials diffusion,cheb --temporals lstm,tdn --seed $s; save data/${B}_ops_results$S.json; }
done

# ---------------------------------------------------------------- schedule imperfection (Table 4)
BD=tensors_mcodis_pub_exppax_1h_h3; tensor $BD data_dis --freq 1h --in-seq 12 --tar-seq 3 --schedule published --pax expected
BO=tensors_mcodis_real_1h_h3;       tensor $BO data_dis --freq 1h --in-seq 12 --tar-seq 3 --schedule realized
BM=tensors_mcodis_mix2_1h_h3;       tensor $BM data_dis --freq 1h --in-seq 12 --tar-seq 3 --schedule mixed
for s in 42 7 123; do S=$(suf $s)
  have ${BD}_results$S.json || { step "operator view (published schedule, expected loads), seed $s"; python "$SRC/train_models.py" --data data_dis/$BD.npz --spatials none,gcn --temporals lstm,tdn,schedonly --seed $s; save data_dis/${BD}_results$S.json; }
  have ${BO}_results$S.json || { step "oracle (realized schedule and loads), seed $s";               python "$SRC/train_models.py" --data data_dis/$BO.npz --spatials none,gcn --temporals lstm,tdn --seed $s; save data_dis/${BO}_results$S.json; }
  have ${BM}_results$S.json || { step "mixed schedule, seed $s";                                     python "$SRC/train_models.py" --data data_dis/$BM.npz --spatials none,gcn --temporals lstm,tdn --seed $s; save data_dis/${BM}_results$S.json; }
done

# ---------------------------------------------------------------- leakage controls (Table 5)
for s in 42 7 123; do S=$(suf $s)
  have ${B}_schedonly_results$S.json || { cpn data/$B.npz data/${B}_schedonly.npz; step "schedule-only control, realized loads, seed $s"; python "$SRC/train_models.py" --data data/${B}_schedonly.npz --spatials none --temporals schedonly --seed $s; save data/${B}_schedonly_results$S.json; }
  have ${BE}_results$S.json          || { step "expected loads, seed $s"; python "$SRC/train_models.py" --data data/$BE.npz --spatials none,gcn --temporals lstm,tdn,schedonly --seed $s; save data/${BE}_results$S.json data/${BE}_preds$S.npz; }
done
chronos data/$BE.npz $BE

# ---------------------------------------------------------------- TDN ablation and encoder isolation (Table 6)
cpn data/$B.npz data/${B}_tdnabl.npz; cpn data/$B.npz data/${B}_encabl.npz; cpn data/$BE.npz data/${BE}_tdnabl.npz; cpn data/$BE.npz data/${BE}_encabl.npz
cpn data/$B15.npz data/${B15}_encabl.npz; cpn data/$B.npz data/${B}_encgcn.npz
for s in 42 7 123; do S=$(suf $s)
  have ${B}_tdnabl_results$S.json  || { step "TDN ablation, seed $s";              python "$SRC/train_models.py" --data data/${B}_tdnabl.npz  --spatials none --temporals tdn_noq,tdn_nopsi,tdn_noctx,tdn_noprior,tdn_head,tdn_both,tdn_noq_both --seed $s; save data/${B}_tdnabl_results$S.json data/${B}_tdnabl_preds$S.npz; }
  have ${B}_encabl_results$S.json  || { step "LSTM with the TDN schedule pathway, seed $s"; python "$SRC/train_models.py" --data data/${B}_encabl.npz  --spatials none --temporals lstm_ctx,lstm_ctxboth --seed $s; save data/${B}_encabl_results$S.json data/${B}_encabl_preds$S.npz; }
  have ${B}_encgcn_results$S.json  || { step "same, with the GCN operator, seed $s";  python "$SRC/train_models.py" --data data/${B}_encgcn.npz  --spatials gcn  --temporals lstm_ctx,lstm_ctxboth --seed $s; save data/${B}_encgcn_results$S.json; }
  have ${B15}_encabl_results$S.json|| { step "same, 15-minute bins, seed $s";         python "$SRC/train_models.py" --data data/${B15}_encabl.npz --spatials none --temporals lstm_ctx,lstm_ctxboth --seed $s; save data/${B15}_encabl_results$S.json; }
  have ${BE}_tdnabl_results$S.json || { step "hybrids with expected loads, seed $s";  python "$SRC/train_models.py" --data data/${BE}_tdnabl.npz --spatials none --temporals tdn_both,tdn_noq_both --seed $s; save data/${BE}_tdnabl_results$S.json data/${BE}_tdnabl_preds$S.npz; }
  have ${BE}_encabl_results$S.json || { step "LSTM pathway with expected loads, seed $s"; python "$SRC/train_models.py" --data data/${BE}_encabl.npz --spatials none --temporals lstm_ctx,lstm_ctxboth --seed $s; save data/${BE}_encabl_results$S.json data/${BE}_encabl_preds$S.npz; }
done
# hybrids and the LSTM pathway under the disrupted published schedule and the mixed schedule
cpn data_dis/$BD.npz data_dis/${BD}_tdnabl.npz; cpn data_dis/$BD.npz data_dis/${BD}_encabl.npz; cpn data_dis/$BM.npz data_dis/${BM}_encabl.npz
for s in 42 7 123; do S=$(suf $s)
  have ${BD}_tdnabl_results$S.json || { step "hybrids, disrupted operator view, seed $s"; python "$SRC/train_models.py" --data data_dis/${BD}_tdnabl.npz --spatials none --temporals tdn_both,tdn_noq_both --seed $s; save data_dis/${BD}_tdnabl_results$S.json; }
  have ${BD}_encabl_results$S.json || { step "LSTM pathway, disrupted operator view, seed $s"; python "$SRC/train_models.py" --data data_dis/${BD}_encabl.npz --spatials none --temporals lstm_ctx,lstm_ctxboth --seed $s; save data_dis/${BD}_encabl_results$S.json; }
  have ${BM}_encabl_results$S.json || { step "LSTM pathway, mixed schedule, seed $s";          python "$SRC/train_models.py" --data data_dis/${BM}_encabl.npz --spatials none --temporals lstm_ctx,lstm_ctxboth --seed $s; save data_dis/${BM}_encabl_results$S.json; }
done

# ---------------------------------------------------------------- horizon sweep, 24-hour input (Figure 4)
for H in 1 3 6 12 24; do
  BH=tensors_mco_1h_in24_H$H;           tensor $BH  data --freq 1h --in-seq 24 --tar-seq $H
  have ${BH}_results.json  || { step "horizon $H h, seed 42"; python "$SRC/train_models.py" --data data/$BH.npz --spatials none,gcn,gat --temporals lstm,transformer,tdn --seed 42; save data/${BH}_results.json; }
  chronos data/$BH.npz $BH nolora
  if [ "$H" != "1" ]; then
    BHE=tensors_mco_1h_in24_H${H}_exppax; tensor $BHE data --freq 1h --in-seq 24 --tar-seq $H --pax expected
    cpn data/$BH.npz data/${BH}_tdnabl.npz; cpn data/$BH.npz data/${BH}_encabl.npz; cpn data/$BHE.npz data/${BHE}_encabl.npz
    for s in 42 7 123; do S=$(suf $s)
      have ${BH}_tdnabl_results$S.json || { step "horizon $H h, hybrids, seed $s";        python "$SRC/train_models.py" --data data/${BH}_tdnabl.npz --spatials none --temporals tdn_both,tdn_noq_both --seed $s; save data/${BH}_tdnabl_results$S.json; }
      have ${BHE}_results$S.json       || { step "horizon $H h, expected loads, seed $s"; python "$SRC/train_models.py" --data data/$BHE.npz --spatials none --temporals lstm,tdn,tdn_both,tdn_noq_both --seed $s; save data/${BHE}_results$S.json; }
    done
    have ${BH}_encabl_results.json  || { step "horizon $H h, LSTM pathway";                  python "$SRC/train_models.py" --data data/${BH}_encabl.npz  --spatials none --temporals lstm_ctx,lstm_ctxboth --seed 42; save data/${BH}_encabl_results.json; }
    have ${BHE}_encabl_results.json || { step "horizon $H h, LSTM pathway, expected loads";  python "$SRC/train_models.py" --data data/${BHE}_encabl.npz --spatials none --temporals lstm_ctx,lstm_ctxboth --seed 42; save data/${BHE}_encabl_results.json; }
    have ${BHE}_chronos2.json       || { step "horizon $H h, Chronos-2 zero-shot, expected loads"; python "$SRC/baseline_chronos2.py" --data data/$BHE.npz && save data/${BHE}_chronos2.json; }
  fi
done

# ---------------------------------------------------------------- linear terminal (Table 7)
export AIRPORT_LAYOUT=linear
[ -f data_lin/passengers.csv ] || { step "linear terminal dataset"; python "$SRC/generate_data.py" --out data_lin; }
BL=tensors_lin_1h_h3; tensor $BL data_lin --freq 1h --in-seq 12 --tar-seq 3
cpn data_lin/$BL.npz data_lin/${BL}_encabl.npz
for s in 42 7; do S=$(suf $s)
  have ${BL}_results$S.json        || { step "linear terminal, seed $s";               python "$SRC/train_models.py" --data data_lin/$BL.npz --spatials none,gcn --temporals lstm,tdn --seed $s; save data_lin/${BL}_results$S.json; }
  have ${BL}_encabl_results$S.json || { step "linear terminal, LSTM pathway, seed $s"; python "$SRC/train_models.py" --data data_lin/${BL}_encabl.npz --spatials none,gcn --temporals lstm_ctx,lstm_ctxboth --seed $s; save data_lin/${BL}_encabl_results$S.json; }
done
export AIRPORT_LAYOUT=mco

# ---------------------------------------------------------------- sensitivity to the simulator parameters (Table 8)
sens() { local name=$1; shift; gen data_$name "$@"; tensor tensors_mco_1h_h3_$name data_$name --freq 1h --in-seq 12 --tar-seq 3; }
sens lf75 --lf-mean 0.75; sens lf92 --lf-mean 0.92; sens su125 --showup-scale 1.25; sens su80 --showup-scale 0.80; sens lane75 --lane-scale 0.75; sens seed7 --seed 7
SHIFTS="lf_0.75=data_lf75/tensors_mco_1h_h3_lf75.npz lf_0.92=data_lf92/tensors_mco_1h_h3_lf92.npz showup_x1.25=data_su125/tensors_mco_1h_h3_su125.npz showup_x0.80=data_su80/tensors_mco_1h_h3_su80.npz lanes_x0.75=data_lane75/tensors_mco_1h_h3_lane75.npz delays_30pct=data_dis/$BO.npz new_seed=data_seed7/tensors_mco_1h_h3_seed7.npz"
for s in 42 7 123; do
  have sensitivity_${B}_seed$s.json || { step "sensitivity evaluation, seed $s"; python "$SRC/eval_sensitivity.py" --data data/$B.npz --shift $SHIFTS \
      --models "LSTM+Sched=none/lstm/1,TDN+Sched=none/tdn/1,Hybrid=none/tdn_noq_both/1,LSTM=none/lstm/0" --seed $s --out "$RES/sensitivity_${B}_seed$s.json"; }
done
python "$SRC/sensitivity_summary.py" "$RES" > "$RES/sensitivity_summary.txt"

# ---------------------------------------------------------------- staffing rule, bootstrap intervals, summaries
step "staffing rule and bootstrap intervals"
P=""; for f in data/${B}_preds.npz data/${B}_tdnabl_preds.npz data/${B}_encabl_preds.npz data/${B}_chronos2_preds.npz data/${B}_chronos2_lora_preds.npz; do [ -f $f ] && P="$P $f" || true; done
python "$SRC/staffing_eval.py" $P --out "$RES/staffing_${B}.json" | tee "$RES/staffing_${B}.txt"
python "$SRC/bootstrap_ci.py"  $P --ref "LSTM+Sched" --out "$RES/bootstrap_${B}.json" | tee "$RES/bootstrap_${B}.txt"
PE=""; for f in data/${BE}_preds.npz data/${BE}_tdnabl_preds.npz data/${BE}_encabl_preds.npz data/${BE}_chronos2_preds.npz data/${BE}_chronos2_lora_preds.npz; do [ -f $f ] && PE="$PE $f" || true; done
python "$SRC/staffing_eval.py" $PE --out "$RES/staffing_${BE}.json" | tee "$RES/staffing_${BE}.txt"
python "$SRC/bootstrap_ci.py"  $PE --ref "LSTM+Sched" --out "$RES/bootstrap_${BE}.json" | tee "$RES/bootstrap_${BE}.txt"
python "$SRC/analysis_macro.py" "$RES" > "$RES/macro_all.txt"
python "$SRC/summarize_results.py" "$RES" > "$RES/summary_all.txt"
python "$SRC/model_card.py" --data data/$B.npz --out "$RES/model_card_${B}.json" > "$RES/model_card_${B}.txt"
python "$SRC/simulator_validation.py" --data data --out "$RES/simulator_stats_baseline.json" > /dev/null
echo; echo "done: results in $RES"
