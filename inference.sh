# The number of processes can only be one for inference
export CUDA_VISIBLE_DEVICES=7

torchrun --nproc_per_node 1 --master_port=25648 inference.py \
        --dataset lastfm \
        --rec_model_path /data1/chenxiao/LLaRA/rec_model/lastfm.pt \
        --prompt_path ./prompt/artist.txt \
        --batch_size 1 \
        --base_model /data1/chenxiao/LLaRA/hf_model \
        --resume_from_checkpoint /data1/chenxiao/LLaRA/checkpoints/lastfm_llama2_distill_stage0/final_checkpoint 
	>  eval_lastfm_sft.log