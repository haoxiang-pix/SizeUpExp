from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

import multiprocessing
import tqdm, json

# DEMO: https://huggingface.co/spaces/Qwen/Qwen-VL-Max

def proc(params):
  gpu, fpaths = params

  tokenizer = AutoTokenizer.from_pretrained("pretrained_models/Qwen-VL-Chat", device_map=f'cuda:{gpu}', trust_remote_code=True)
  model = AutoModelForCausalLM.from_pretrained("pretrained_models/Qwen-VL-Chat", device_map=f"cuda:{gpu}", trust_remote_code=True).eval()
  model.generation_config = GenerationConfig.from_pretrained("pretrained_models/Qwen-VL-Chat", trust_remote_code=True)

  res = {}
  failed = []
  for fpath in tqdm.tqdm(fpaths):
    try:
      query = tokenizer.from_list_format([
          {'image': fpath},
          {'text': 'Answer YES, if this is a picture of a model in frontal view and standing pose, with both arms and legs visible, otherwise answer NO'}
      ])
      response, history = model.chat(tokenizer, query=query, history=None)
      res[fpath] = response
    except:
      failed.append(fpath)
      print ('failed',len(failed),gpu)
  return res

import argparse
if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('fpaths_list')
  parser.add_argument('--gpus', default=1, type=int)
  parser.add_argument('output_path')
  args = parser.parse_args()

  fpaths = [l.rstrip() for l in open(args.fpaths_list).readlines()]
  num_gpus = args.gpus
  output_path = args.output_path

  jobs_vec = [[g,[]] for g in range(num_gpus)]
  for i,fpath in enumerate(fpaths):
    jobs_vec[i%num_gpus][1].append(fpath)
  res_all = {}
  for res in multiprocessing.Pool(num_gpus).imap_unordered(proc, jobs_vec):
    res_all.update(res)
  json.dump(res_all, open(output_path, 'w'))
