set -e

python cli/train_nce.py -m seed=2,3,4,5,6,7,8,9,10,11
python cli/train_encoder.py -m seed=2,3,4,5,6,7,8,9,10,11

