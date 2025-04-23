export CUDA_VISIBLE_DEVICES=0,1

python main_test.py \
--stage teacher_sft \
--mode test \
--batch_size 1 \
--accumulate_grad_batches 1 \
--dataset steam_data \
--data_dir ./hf_data/steam \
--cans_num 40 \
--prompt_path ./prompt/game.txt \
--rec_embed SASRec \
--llm_tuning lora \
--llm_path ./llama3_8b \
--rec_model_path ./rec_model/steam.pt \
--ckpt_path /data/cx/checkpoints/steam_llama3/epoch=03-metric=0.473.ckpt \
--output_dir ./output/steam/ 