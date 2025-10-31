set -e

python cli/train_aux_imca.py -m seed=7,8,9,10,11
python cli/train_nce_imca.py -m seed=7,8,9,10,11
python cli/train_ivae_imca.py -m seed=7,8,9,10,11
