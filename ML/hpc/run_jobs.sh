#!/usr/bin/env bash
# ChargeAndBreak ML jobs on a Linux cluster (moved off the laptop 2026-10-02).
# Run from anywhere; it works from the repository root:
#
#     bash ML/hpc/run_jobs.sh check          # packages + gurobi_cl present?
#     bash ML/hpc/run_jobs.sh pilot          # LA on the rest of the 48 mixed-power training routes
#     bash ML/hpc/run_jobs.sh b1             # direction B: train + validate (validation routes only)
#     bash ML/hpc/run_jobs.sh dagger         # DAgger probe on the stop split
#     bash ML/hpc/run_jobs.sh extract-pmix   # after `pilot`: the pilot's LA runs -> training rows
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

*)
  echo "usage: bash ML/hpc/run_jobs.sh check|pilot|b1|dagger|extract-pmix"
  exit 1
  ;;
esac
