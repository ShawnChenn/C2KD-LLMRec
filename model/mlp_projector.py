from torch import nn

class MlpProjector(nn.Module):
    def __init__(self, rec_size=64, llm_size=4096):
        super().__init__()
        self.mlp_proj = nn.Sequential(
            nn.Linear(rec_size, llm_size),
            nn.GELU(),
            nn.Linear(llm_size, llm_size)
        )

    def forward(self, x):
        x = self.mlp_proj(x)
        return x

if __name__ == "__main__":
    import torch
    ckpt_head_path = '/data/mcy/LLaRA/checkpoints/lastfm_peft_stage1/epoch=01-metric=0.639.ckpt'
    ckpt = torch.load(ckpt_head_path, map_location='cpu')
    for k1 in ckpt['state_dict'].keys():
        print('k1', k1)