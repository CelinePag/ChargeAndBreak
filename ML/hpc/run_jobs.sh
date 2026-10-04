#!/usr/bin/env bash
# ChargeAndBreak ML jobs on a Linux cluster (moved off the laptop 2026-10-02).
# Run from anywhere; it works from the repository root:
#
#     bash ML/hpc/run_jobs.sh check          # packages + gurobi_cl present?
#     bash ML/hpc/run_jobs.sh pilot          # LA on the rest of the 48 mixed-power training routes
#     bash ML/hpc/run_jobs.sh b1             # direction B: train + validate (validation routes only)
#     bash ML/hpc/run_jobs.sh dagger         # DAgger probe on the stop split
#     bash ML/hpc/run_jobs.sh extract-pmix   # after `pilot`: the pilot's LA runs -> training rows
#     bash ML/hpc/run_jobs.sh pilot-models   # after `extract-pmix`: models trained WITH the pilot routes
#     bash ML/hpc/run_jobs.sh test           # the model chosen on validation: 2 more seeds, test once
#     bash ML/hpc/run_jobs.sh curve-build    # more mixed training routes (seconds)
#     bash ML/hpc/run_jobs.sh curve-la i/n   # one of n parallel shares of their LA runs
#     bash ML/hpc/run_jobs.sh dagger-pmix i/n  # DAgger on the pilot routes, one of n label shares
#     bash ML/hpc/run_jobs.sh curve-train    # after both: extract, train, test once
#     bash ML/hpc/run_jobs.sh la-test i/n    # LA on the 24 mixed test routes without one
#     bash ML/hpc/run_jobs.sh dagger2 i/n    # DAgger round 2, one of n label shares
#     bash ML/hpc/run_jobs.sh final          # after both: 121-route point, round-2 models, test
#     bash ML/hpc/run_jobs.sh mix-build      # power AND spacing mixed: 124 training + 16 validation routes
#     bash ML/hpc/run_jobs.sh mix-la i/n     # one of n shares: LA on those training routes, then the 32 test routes
#     bash ML/hpc/run_jobs.sh mix-oracle     # hindsight oracle on the 16 validation routes
#     bash ML/hpc/run_jobs.sh mix-train      # after all of them: extract, train, validate, test once
#
# The stages are independent and can run at the same time.  On a plain
# machine (no Slurm), start each in the background and give it its share of
# the cores with NCPU, e.g.
#
#     NCPU=16 nohup bash ML/hpc/run_jobs.sh pilot > ML/logs/nohup_pilot.out 2>&1 &
#
# Every step skips work whose output exists, so a killed job is resumed by
# running the same stage again.  Logs: ML/logs/ (summary: ML/logs/hpc_jobs.log).
set -euo pipefail
cd "$(dirname "$0")/../.."

NCPU=${NCPU:-${SLURM_CPUS_PER_TASK:-$(nproc)}}
LOGS=ML/logs
mkdir -p "$LOGS"
PHYS=base,kwh300,kwh700,kwh900,kw150,kw700,kw1000,cs30,cs100
log() { echo "$(date '+%Y-%m-%d %H:%M')  $*" | tee -a "$LOGS/hpc_jobs.log"; }
# The LA runs 8 scenario solves in parallel (as every stored teacher run did);
# give each Gurobi solve its share of the cores instead of all of them.
la_threads() { export CB_GRB_THREADS=$(( NCPU / 8 > 0 ? NCPU / 8 : 1 )); }

case "${1:-}" in
check)
  python - <<'EOF'
import importlib, shutil, sys
for m in ["numpy", "scipy", "pyomo", "lightgbm", "sklearn", "torch", "joblib",
          "matplotlib", "openpyxl", "pptx"]:
    importlib.import_module(m)
print("python", sys.version.split()[0], "- packages ok")
print("gurobi_cl:", shutil.which("gurobi_cl") or "NOT FOUND - load the cluster's Gurobi module")
EOF
  ;;

pilot)
  la_threads
  log "pilot: LA on the remaining mixed-power training routes (CB_GRB_THREADS=$CB_GRB_THREADS)"
  python -u ML/code/run_la_mixed.py --split train >> "$LOGS/la_mixed_train.log" 2>&1
  log "pilot done"
  ;;

b1)
  T=$(( NCPU < 16 ? NCPU : 16 ))
  COMMON="--fset T --physics $PHYS --seed 0 --threads $T --lambda-list 1 --tau 0.25"
  train() {
    local tag=$1; shift
    if [ -f "ML/models/${tag}_torch.pt" ]; then log "b1: $tag exists, skipped"; return; fi
    log "b1: training $tag"
    python -u ML/code/torch_train.py --tag "$tag" $COMMON "$@" > "$LOGS/train_${tag}.log" 2>&1
  }
  train tmlp_T144_phys_split_list_s0 --arch split                       # control: inputs, no structure
  train tmlp_T144_phys_charger_s0    --arch charger --g-exclude power   # ChargerNet
  train tmlp_T144_phys_chargerG_s0   --arch charger --g-exclude none    # ChargerNet, speed also in g
  # validation only: uniform stop split, then the 16 mixed-power validation routes
  for spec in torch:tmlp_T144_phys_split_list_s0 torch:tmlp_T144_phys_charger_s0 \
              torch:tmlp_T144_phys_chargerG_s0 torch:tmlp_F95_phys_split_list_s0 \
              gbt:gbt_F95_phys_s1; do
    kind=${spec%%:*}; tag=${spec#*:}
    out=eval_${tag}_g99sr_stop.json
    if [ -f "ML/results/$out" ]; then continue; fi
    log "b1: validating $tag on the uniform stop split"
    python -u ML/code/evaluate.py --kind "$kind" --tag "$tag" --split stop \
        --guard-q 0.99 --spread-room --out "$out" > "$LOGS/eval_${tag}_g99sr_stop.log" 2>&1
  done
  log "b1: mixed-power validation routes"
  python -u ML/code/mixed_eval.py drive --set val --variants pmix \
      --jobs $(( NCPU < 8 ? NCPU : 8 )) > "$LOGS/mixed_val_drive_b1.log" 2>&1
  log "b1 done"
  ;;

dagger)
  la_threads
  P="--label probe --routes base --split stop --stops 3"
  log "dagger: rollouts"
  python -u ML/code/dagger_rollout.py $P --models torch:tmlp_F95_split_list_s0 --check-stops 1 >> "$LOGS/dagger_probe.log" 2>&1
  python -u ML/code/dagger_rollout.py $P --models gbt:gbt_F95_base_s0 >> "$LOGS/dagger_probe.log" 2>&1
  log "dagger: LA labels (CB_GRB_THREADS=$CB_GRB_THREADS)"
  python -u ML/code/dagger_label.py --label probe >> "$LOGS/dagger_probe.log" 2>&1
  python -u ML/code/dagger_report.py --label probe --reference --check > "$LOGS/dagger_probe_report.txt" 2>&1
  log "dagger done: $LOGS/dagger_probe_report.txt"
  ;;

extract-pmix)
  log "extract: pilot LA runs -> ML/data/dataset_phys_pmix.npz"
  python -u ML/code/extract.py --physics pmix > "$LOGS/extract_pmix.log" 2>&1
  log "extract done"
  ;;

pilot-models)
  # the DATA answer to mixed chargers, next to b1's STRUCTURE answer: the same
  # models with the 47 pilot routes added to every physics value.  Validation
  # routes only, like b1.
  T=$(( NCPU < 16 ? NCPU : 16 ))
  PHYSP=$PHYS,pmix
  fit() {
    local tag=$1; shift
    if [ -f "ML/models/${tag}_meta.json" ]; then log "pilot-models: $tag exists, skipped"; return; fi
    log "pilot-models: training $tag"
    "$@" > "$LOGS/train_${tag}.log" 2>&1
  }
  fit gbt_P102_physpmix_s1    python -u ML/code/gbt_train.py --tag gbt_P102_physpmix_s1 --fset P \
      --physics "$PHYSP" --seed 1
  fit gbt_P102_physpmix_w5_s1 python -u ML/code/gbt_train.py --tag gbt_P102_physpmix_w5_s1 --fset P \
      --physics "$PHYSP" --seed 1 --weight-physics pmix=5
  TCOMMON="--fset T --physics $PHYSP --seed 0 --threads $T --lambda-list 1 --tau 0.25"
  fit tmlp_T144_physpmix_split_list_s0 python -u ML/code/torch_train.py \
      --tag tmlp_T144_physpmix_split_list_s0 $TCOMMON --arch split
  fit tmlp_T144_physpmix_charger_s0 python -u ML/code/torch_train.py \
      --tag tmlp_T144_physpmix_charger_s0 $TCOMMON --arch charger --g-exclude power
  for spec in gbt:gbt_P102_physpmix_s1 gbt:gbt_P102_physpmix_w5_s1 \
              torch:tmlp_T144_physpmix_split_list_s0 torch:tmlp_T144_physpmix_charger_s0; do
    kind=${spec%%:*}; tag=${spec#*:}
    out=eval_${tag}_g99sr_stop.json
    if [ -f "ML/results/$out" ]; then continue; fi
    log "pilot-models: validating $tag on the uniform stop split"
    python -u ML/code/evaluate.py --kind "$kind" --tag "$tag" --split stop \
        --guard-q 0.99 --spread-room --out "$out" > "$LOGS/eval_${tag}_g99sr_stop.log" 2>&1
  done
  log "pilot-models: mixed-power validation routes"
  python -u ML/code/mixed_eval.py drive --set val --variants pmix \
      --jobs $(( NCPU < 8 ? NCPU : 8 )) > "$LOGS/mixed_val_drive_pilot.log" 2>&1
  log "pilot-models done"
  ;;

test)
  # ChargerNet + pilot was CHOSEN on the validation routes (2026-10-03).  Two
  # more seeds, then every direction-B / pilot model on the TEST routes, once:
  # base case (125 routes), mixed routes (pmix / dmix / mix, 32 each), and the
  # LA comparison on its 8 pmix routes.  The other models are ablation rows.
  T=$(( NCPU < 16 ? NCPU : 16 ))
  for s in 1 2; do
    tag=tmlp_T144_physpmix_charger_s$s
    if [ -f "ML/models/${tag}_torch.pt" ]; then log "test: $tag exists, skipped"; continue; fi
    log "test: training $tag"
    python -u ML/code/torch_train.py --tag "$tag" --fset T --physics "$PHYS,pmix" --seed "$s" \
        --threads "$T" --lambda-list 1 --tau 0.25 --arch charger --g-exclude power \
        > "$LOGS/train_${tag}.log" 2>&1
  done
  for spec in torch:tmlp_T144_physpmix_charger_s0 torch:tmlp_T144_physpmix_charger_s1 \
              torch:tmlp_T144_physpmix_charger_s2 torch:tmlp_T144_physpmix_split_list_s0 \
              torch:tmlp_T144_phys_charger_s0 torch:tmlp_T144_phys_split_list_s0 \
              torch:tmlp_F95_phys_split_list_s0 gbt:gbt_P102_physpmix_s1 \
              gbt:gbt_P102_physpmix_w5_s1 gbt:gbt_F95_phys_s1; do
    kind=${spec%%:*}; tag=${spec#*:}
    out=eval_${tag}_g99sr_test.json
    if [ -f "ML/results/$out" ]; then continue; fi
    log "test: $tag on the 125 base-case test routes"
    python -u ML/code/evaluate.py --kind "$kind" --tag "$tag" --split test \
        --guard-q 0.99 --spread-room --out "$out" > "$LOGS/eval_${tag}_g99sr_test.log" 2>&1
  done
  log "test: mixed test routes (pmix, dmix, mix)"
  python -u ML/code/mixed_eval.py drive --jobs $(( NCPU < 8 ? NCPU : 8 )) \
      > "$LOGS/mixed_test_drive_b.log" 2>&1
  python -u ML/code/mixed_eval.py la --variants pmix > "$LOGS/mixed_test_la_b.log" 2>&1
  log "test done"
  ;;

# ── more mixed data vs DAgger at equal LA time (2026-10-03) ─────────────────
# Learning curve: no mixed data -> the 47 short pilot routes (frozen as the
# physics tag pmix47) -> every mixed training route (short seeds 1-19 +
# medium seeds 1-12, tag pmix).  DAgger: the PyTorch model trained WITHOUT
# mixed data drives the pilot's 48 short routes and the LA labels every stop
# where it has left the LA's trajectory (~2,000 calls, the pilot made 2,279).
curve-build)
  log "curve-build: more mixed-power training routes"
  python -u ML/code/mixed_instances.py --split train --variants pmix --lengths short --seeds 13-19
  python -u ML/code/mixed_instances.py --split train --variants pmix --lengths medium --seeds 1-12
  log "curve-build done"
  ;;

curve-la)
  # one share of the new LA runs: bash ML/hpc/run_jobs.sh curve-la 0/3  (and 1/3, 2/3)
  la_threads
  SL=${2:-0/1}
  log "curve-la $SL: LA on the new mixed training routes (CB_GRB_THREADS=$CB_GRB_THREADS)"
  python -u ML/code/run_la_mixed.py --split train --slice "$SL" \
      >> "$LOGS/la_mixed_train_curve_${SL/\//of}.log" 2>&1
  log "curve-la $SL done"
  ;;

dagger-pmix)
  # rollout once, then one share of the labels: bash ML/hpc/run_jobs.sh dagger-pmix 0/2  (and 1/2)
  la_threads
  SL=${2:-0/1}
  if [ "$SL" = "0/1" ] || [ "${SL%%/*}" = "0" ]; then
    log "dagger-pmix: rollout of tmlp_T144_phys_split_list_s0 on the pilot's 48 short routes"
    python -u ML/code/dagger_rollout.py --label dgpmix --routes pmix --split fit \
        --route-seeds 1-12 --lengths short --per-family 12 --stops 999 \
        --models torch:tmlp_T144_phys_split_list_s0 >> "$LOGS/dagger_pmix.log" 2>&1
  else
    while [ ! -f ML/data/dagger/dgpmix/queries/.done ]; do sleep 30; done
  fi
  [ "${SL%%/*}" = "0" ] && touch ML/data/dagger/dgpmix/queries/.done
  log "dagger-pmix $SL: LA labels (CB_GRB_THREADS=$CB_GRB_THREADS)"
  python -u ML/code/dagger_label.py --label dgpmix --slice "$SL" \
      >> "$LOGS/dagger_pmix_label_${SL/\//of}.log" 2>&1
  log "dagger-pmix $SL done"
  ;;

curve-train)
  # after every curve-la and dagger-pmix share has finished
  T=$(( NCPU < 16 ? NCPU : 16 ))
  log "curve-train: extract every mixed training route"
  python -u ML/code/extract.py --physics pmix > "$LOGS/extract_pmix_all.log" 2>&1
  TC="--fset T --threads $T --lambda-list 1 --tau 0.25 --arch split"
  fitm() {
    local tag=$1; shift
    if [ -f "ML/models/${tag}_meta.json" ]; then log "curve-train: $tag exists, skipped"; return; fi
    log "curve-train: training $tag"
    "$@" > "$LOGS/train_${tag}.log" 2>&1
  }
  for s in 0 1 2; do
    fitm tmlp_T144_physpmix_split_list_s$s python -u ML/code/torch_train.py \
        --tag tmlp_T144_physpmix_split_list_s$s --physics "$PHYS,pmix47" --seed $s $TC
    fitm tmlp_T144_physpmixall_split_list_s$s python -u ML/code/torch_train.py \
        --tag tmlp_T144_physpmixall_split_list_s$s --physics "$PHYS,pmix" --seed $s $TC
    fitm tmlp_T144_physdg_split_list_s$s python -u ML/code/torch_train.py \
        --tag tmlp_T144_physdg_split_list_s$s --physics "$PHYS" --dagger dgpmix --seed $s $TC
  done
  fitm gbt_P102_physpmixall_s1 python -u ML/code/gbt_train.py --tag gbt_P102_physpmixall_s1 \
      --fset P --physics "$PHYS,pmix" --seed 1
  fitm gbt_P102_physdg_s1 python -u ML/code/gbt_train.py --tag gbt_P102_physdg_s1 \
      --fset P --physics "$PHYS" --dagger dgpmix --seed 1
  for spec in torch:tmlp_T144_physpmix_split_list_s1 torch:tmlp_T144_physpmix_split_list_s2 \
              torch:tmlp_T144_physpmixall_split_list_s0 torch:tmlp_T144_physpmixall_split_list_s1 \
              torch:tmlp_T144_physpmixall_split_list_s2 torch:tmlp_T144_physdg_split_list_s0 \
              torch:tmlp_T144_physdg_split_list_s1 torch:tmlp_T144_physdg_split_list_s2 \
              gbt:gbt_P102_physpmixall_s1 gbt:gbt_P102_physdg_s1; do
    kind=${spec%%:*}; tag=${spec#*:}
    out=eval_${tag}_g99sr_test.json
    if [ -f "ML/results/$out" ]; then continue; fi
    log "curve-train: $tag on the 125 base-case test routes"
    python -u ML/code/evaluate.py --kind "$kind" --tag "$tag" --split test \
        --guard-q 0.99 --spread-room --out "$out" > "$LOGS/eval_${tag}_g99sr_test.log" 2>&1
  done
  log "curve-train: mixed test routes"
  python -u ML/code/mixed_eval.py drive --jobs $(( NCPU < 8 ? NCPU : 8 )) \
      > "$LOGS/mixed_test_drive_curve.log" 2>&1
  python -u ML/code/mixed_eval.py la --variants pmix > "$LOGS/mixed_test_la_curve.log" 2>&1
  log "curve-train done"
  ;;

# ── round 3 (2026-10-03): more LA reference, DAgger round 2, the 121 point ──
# Phase 1 (LA, parallel shares): la-test i/n and dagger2 i/n.
# Phase 2 (only when every phase-1 share is done): final.
la-test)
  # the LA on the 24 mixed-power TEST routes it has not run (8 already have it)
  la_threads
  SL=${2:-0/1}
  log "la-test $SL: LA on the remaining mixed test routes (CB_GRB_THREADS=$CB_GRB_THREADS)"
  python -u ML/code/run_la_mixed.py --split test --routes all --slice "$SL" \
      >> "$LOGS/la_mixed_test_all_${SL/\//of}.log" 2>&1
  log "la-test $SL done"
  ;;

dagger2)
  # DAgger round 2: the student trained with round-1 labels drives the same 48
  # short routes; the LA corrects it where it has left the LA's trajectory.
  la_threads
  SL=${2:-0/1}
  if [ "${SL%%/*}" = "0" ]; then
    log "dagger2: rollout of tmlp_T144_physdg_split_list_s0 on the pilot's 48 short routes"
    python -u ML/code/dagger_rollout.py --label dgpmix2 --routes pmix --split fit \
        --route-seeds 1-12 --lengths short --per-family 12 --stops 999 \
        --models torch:tmlp_T144_physdg_split_list_s0 >> "$LOGS/dagger_pmix2.log" 2>&1
    touch ML/data/dagger/dgpmix2/queries/.done
  else
    while [ ! -f ML/data/dagger/dgpmix2/queries/.done ]; do sleep 30; done
  fi
  log "dagger2 $SL: LA labels (CB_GRB_THREADS=$CB_GRB_THREADS)"
  python -u ML/code/dagger_label.py --label dgpmix2 --slice "$SL" \
      >> "$LOGS/dagger_pmix2_label_${SL/\//of}.log" 2>&1
  log "dagger2 $SL done"
  ;;

final)
  # refuse to start while any LA work is still running (round 2 trained on 89
  # of 121 routes because training started before the LA shares had finished)
  if pgrep -f "run_la_mixed.py|dagger_label.py|dagger_rollout.py" > /dev/null; then
    echo "LA work is still running -- wait until every la-test / dagger2 share says done"
    exit 1
  fi
  T=$(( NCPU < 16 ? NCPU : 16 ))
  log "final: extract every mixed training route (121 usable expected)"
  python -u ML/code/extract.py --physics pmix > "$LOGS/extract_pmix_121.log" 2>&1
  TC="--fset T --threads $T --lambda-list 1 --tau 0.25 --arch split"
  fitm() {
    local tag=$1; shift
    if [ -f "ML/models/${tag}_meta.json" ]; then log "final: $tag exists, skipped"; return; fi
    log "final: training $tag"
    "$@" > "$LOGS/train_${tag}.log" 2>&1
  }
  for s in 0 1 2; do
    fitm tmlp_T144_physpmix121_split_list_s$s python -u ML/code/torch_train.py \
        --tag tmlp_T144_physpmix121_split_list_s$s --physics "$PHYS,pmix" --seed $s $TC
    fitm tmlp_T144_physdg2_split_list_s$s python -u ML/code/torch_train.py \
        --tag tmlp_T144_physdg2_split_list_s$s --physics "$PHYS" --dagger dgpmix,dgpmix2 --seed $s $TC
  done
  fitm gbt_P102_physpmix121_s1 python -u ML/code/gbt_train.py --tag gbt_P102_physpmix121_s1 \
      --fset P --physics "$PHYS,pmix" --seed 1
  for spec in torch:tmlp_T144_physpmix121_split_list_s0 torch:tmlp_T144_physpmix121_split_list_s1 \
              torch:tmlp_T144_physpmix121_split_list_s2 torch:tmlp_T144_physdg2_split_list_s0 \
              torch:tmlp_T144_physdg2_split_list_s1 torch:tmlp_T144_physdg2_split_list_s2 \
              gbt:gbt_P102_physpmix121_s1; do
    kind=${spec%%:*}; tag=${spec#*:}
    out=eval_${tag}_g99sr_test.json
    if [ -f "ML/results/$out" ]; then continue; fi
    log "final: $tag on the 125 base-case test routes"
    python -u ML/code/evaluate.py --kind "$kind" --tag "$tag" --split test \
        --guard-q 0.99 --spread-room --out "$out" > "$LOGS/eval_${tag}_g99sr_test.log" 2>&1
  done
  log "final: mixed test routes, then the LA comparison on every route it has run"
  python -u ML/code/mixed_eval.py drive --jobs $(( NCPU < 8 ? NCPU : 8 )) \
      > "$LOGS/mixed_test_drive_final.log" 2>&1
  python -u ML/code/mixed_eval.py la --variants pmix > "$LOGS/mixed_test_la_final.log" 2>&1
  log "final done"
  ;;

mix-build)
  # Power AND spacing mixed along the route (variant "mix"), same families and
  # seeds as the 124 power-mixed training routes, plus validation seeds 20-21
  log "mix-build: power+spacing mixed routes (training and validation)"
  python -u ML/code/mixed_instances.py --split train --variants mix --lengths short --seeds 1-19
  python -u ML/code/mixed_instances.py --split train --variants mix --lengths medium --seeds 1-12
  python -u ML/code/mixed_instances.py --split val --variants mix
  log "mix-build done"
  ;;

mix-la)
  # one share: bash ML/hpc/run_jobs.sh mix-la 0/4  (and 1/4, 2/4, 3/4).  Routes
  # go seed by seed, so a share stopped early still covers every family.
  la_threads
  SL=${2:-0/1}
  log "mix-la $SL: LA on the mix training routes (CB_GRB_THREADS=$CB_GRB_THREADS)"
  python -u ML/code/run_la_mixed.py --variant mix --split train --slice "$SL" \
      >> "$LOGS/la_mix_train_${SL/\//of}.log" 2>&1
  log "mix-la $SL: LA on the 32 mix test routes"
  python -u ML/code/run_la_mixed.py --variant mix --split test --routes all --slice "$SL" \
      >> "$LOGS/la_mix_test_${SL/\//of}.log" 2>&1
  log "mix-la $SL done"
  ;;

mix-oracle)
  log "mix-oracle: hindsight oracle on the mix validation routes"
  python -u ML/code/mixed_eval.py oracle --set val --variants mix \
      > "$LOGS/mixed_val_oracle_mix.log" 2>&1
  log "mix-oracle done"
  ;;

mix-train)
  if pgrep -f "run_la_mixed.py|mixed_eval.py oracle" > /dev/null; then
    echo "LA or oracle work is still running -- wait until every mix-la share and mix-oracle say done"
    exit 1
  fi
  T=$(( NCPU < 16 ? NCPU : 16 ))
  log "mix-train: extract the mix training routes"
  python -u ML/code/extract.py --physics mix > "$LOGS/extract_mix.log" 2>&1
  TC="--fset T --threads $T --lambda-list 1 --tau 0.25 --arch split"
  fitm() {
    local tag=$1; shift
    if [ -f "ML/models/${tag}_meta.json" ]; then log "mix-train: $tag exists, skipped"; return; fi
    log "mix-train: training $tag"
    "$@" > "$LOGS/train_${tag}.log" 2>&1
  }
  for s in 0 1 2; do
    # the pool + both kinds of mixed routes / mixed routes only / pool, mixed x5
    fitm tmlp_T144_physmix_split_list_s$s python -u ML/code/torch_train.py \
        --tag tmlp_T144_physmix_split_list_s$s --physics "$PHYS,pmix,mix" --seed $s $TC
    fitm tmlp_T144_mixonly_split_list_s$s python -u ML/code/torch_train.py \
        --tag tmlp_T144_mixonly_split_list_s$s --physics "pmix,mix" --stop-seeds 11,12 \
        --seed $s $TC
    fitm tmlp_T144_physmixw5_split_list_s$s python -u ML/code/torch_train.py \
        --tag tmlp_T144_physmixw5_split_list_s$s --physics "$PHYS,pmix,mix" \
        --weight-physics pmix=5,mix=5 --seed $s $TC
  done
  for cfg in physmix mixonly physmixw5; do
    for s in 0 1 2; do
      tag=tmlp_T144_${cfg}_split_list_s$s
      out=eval_${tag}_g99sr_test.json
      if [ -f "ML/results/$out" ]; then continue; fi
      log "mix-train: $tag on the 125 base-case test routes"
      python -u ML/code/evaluate.py --kind torch --tag "$tag" --split test \
          --guard-q 0.99 --spread-room --out "$out" > "$LOGS/eval_${tag}_g99sr_test.log" 2>&1
    done
  done
  J=$(( NCPU < 8 ? NCPU : 8 ))
  log "mix-train: validation routes (pmix + mix), then the test routes"
  python -u ML/code/mixed_eval.py drive --set val --variants pmix,mix --jobs $J \
      > "$LOGS/mixed_val_drive_mix.log" 2>&1
  python -u ML/code/mixed_eval.py drive --jobs $J > "$LOGS/mixed_test_drive_mix.log" 2>&1
  python -u ML/code/mixed_eval.py la --variants pmix > "$LOGS/mixed_test_la_pmix_mix.log" 2>&1
  python -u ML/code/mixed_eval.py la --variants mix > "$LOGS/mixed_test_la_mix.log" 2>&1
  log "mix-train done"
  ;;

*)
  echo "usage: bash ML/hpc/run_jobs.sh check|pilot|b1|dagger|extract-pmix|pilot-models|test"
  echo "       |curve-build|curve-la i/n|dagger-pmix i/n|curve-train"
  echo "       |la-test i/n|dagger2 i/n|final"
  echo "       |mix-build|mix-la i/n|mix-oracle|mix-train"
  exit 1
  ;;
esac
