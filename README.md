# LLM practice
Collect resource, do research on small LLM, 

## Learn transformer by coding reading

### [nanochat](https://github.com/karpathy/nanochat)

Step-by-step to learn how to train chatgpt (set small model size and small data set for quick start) locally. 

Follow `speedrun.sh`. But need reduce model size and token size to make it runnable in low resources (cpu/gpu memory)

### [ktransformers](https://github.com/kvcache-ai/ktransformers)


```
python data/shakespeare_char/prepare.py

python train.py config/train_shakespeare_char.py

python sample.py --out_dir=out-shakespeare-char
```
