export CUDA_VISIBLE_DEVICES=0,1

python main_test.py \
--mode test \
--stage distillation_stage2 \
--batch_size 1 \
--accumulate_grad_batches 1 \
--dataset steam_data \
--data_dir ./hf_data/steam \
--cans_num 60 \
--prompt_path ./prompt/game.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./llama3_8b \
--rec_model_path ./rec_model/steam.pt \
--student_pretrained_path ./llama3_1b \
--student_ckpt_path /data1/chenxiao/LLaRA/checkpoints/steam_llama3_distillation_stage2_top1moe_klcosloss/epoch=03-metric=0.434.ckpt \
--output_dir ./output/steam/ \
--log_dir steam_logs \
--lr_warmup_start_lr 5e-6 \
--lr 5e-4 \
--lr_decay_min_lr 5e-6 \
--max_epochs 1
