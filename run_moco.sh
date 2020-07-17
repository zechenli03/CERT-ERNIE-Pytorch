python MOCO.py \
  --lr 0.0001 \
  --b 32 \
  --data ./augmented_data/RTE/rte_augment.csv \
  --epochs 50 \
  --dist-url 'tcp://localhost:10001' \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 