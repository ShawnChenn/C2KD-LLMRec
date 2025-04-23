export CUDA_VISIBLE_DEVICES=0

python main_dpo.py \
--mode train \
--stage distillation_stage2 \
--neg_num 1 \
--batch_size 4 \
--accumulate_grad_batches 16 \
--dataset steam_data \
--data_dir ./hf_data/steam \
--cans_num 20 \
--prompt_path ./prompt/game.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./hf_model \
--rec_model_path ./rec_model/steam.pt \
--ckpt_path ./checkpoints/steam_llama2/epoch=03-metric=0.459.ckpt \
--ckpt_head_path ./checkpoints/steam_llama2_distillation_stage1/epoch=01-metric=0.439.ckpt \
--student_pretrained_path ./tinyllama \
--ckpt_dir ./checkpoints/steam_llama2_distillation_stage2_test/ \
--output_dir ./output/steam/ \
--log_dir steam_logs \
--lr_warmup_start_lr 5e-6 \
--lr 5e-4 \
--lr_decay_min_lr 5e-6 \
--max_epochs 10