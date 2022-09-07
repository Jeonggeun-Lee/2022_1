import torch

common_config = {
  'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
  'sequence_len': 8 * 16, # 8마디, 16 quantize
}