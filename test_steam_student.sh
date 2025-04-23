export CUDA_VISIBLE_DEVICES=2,3

python main_test.py \
--mode test \
--stage teacher_sft \
--batch_size 1 \
--accumulate_grad_batches 1 \
--dataset steam_data \
--data_dir ./hf_data/steam \
--cans_num 60 \
--prompt_path ./prompt/game.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./llama3_1b \
--rec_model_path ./rec_model/steam.pt \
--ckpt_path /data/cx/checkpoints/steam_llama3_student_0117/epoch=04-metric=0.395.ckpt \
--output_dir ./output/steam/ \
--log_dir steam_logs 