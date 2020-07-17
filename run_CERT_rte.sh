export GLUE_DIR=./glue_data
export STATE_DICT=./moco_model/moco.p
export TASK_NAME=RTE
export OUTPUT_DIR=./output

python run_glue.py \
    --model_name_or_path nghuyong/ernie-2.0-large-en \
    --state_dict $STATE_DICT \
    --task_name $TASK_NAME \
    --data_dir $GLUE_DIR/$TASK_NAME \
    --max_seq_length 128 \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluate_during_training \
    --per_device_train_batch_size 16 \
    --weight_decay 0 \
    --learning_rate 2e-5 \
    --num_train_epochs 5.0 \
    --save_steps 156 \
    --warmup_steps 78 \
    --logging_steps 39 \
    --eval_steps 39 \
    --seed 199733 \
    --output_dir $OUTPUT_DIR/$TASK_NAME/4
