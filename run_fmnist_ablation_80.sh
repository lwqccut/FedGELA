#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
common=(--dataset=fmnist --partition=dirichlet --n_parties=2 --party_per_round=1
  --comm_round=50 --epochs=1 --train_batchsize=128 --test_batchsize=256
  --model=resnet20 --logdir=./log --modeldir=./models --proxy=etf
  --temperature=0.2 --beta=0.5)

run_one() {
  name="$1"; shift
  echo "starting $name"
  .venv/bin/python -u FedGELA_improved.py "${common[@]}" --log_file_name="$name" "$@"
  echo "finished $name"
}

run_one fmnist80_etf --use_etf_improved
run_one fmnist80_adapter --use_adapter
run_one fmnist80_curation --use_curation
run_one fmnist80_all --use_etf_improved --use_adapter --use_curation
