#!/bin/bash
seeds=(1 7 21 42 54 84 105 111 123 999)
for cls in 1 2; do
    for i in ${seeds[@]}; do
    python create_tfrecords.py \
        --train_path ../../data/parquets/mhc${cls}/seed_splits/train_seed_${i}.parquet \
        --embed_npz ../../data/ESM/esm3-open/PMGen_whole_seq_/mhc${cls}_encodings.npz \
        --seq_csv ../../data/alleles/aligned_PMGen_class_${cls}.csv \
        --embed_key ../../data/ESM/esm3-open/PMGen_whole_seq_/mhc${cls}_encodings.csv \
        --output_dir ../../data/tfrecords/normal/mhc${cls}/train_seed_${i} \
        --mhc_class ${cls}
    done
done