Here the model to be downloaded from [Hugging Face](https://huggingface.co/philschmid/flan-t5-xxl-sharded-fp16/tree/main). 
No `handler.py` required since we will not use TorchServe. Note also the size (around 20 GiB):
```sh
README.md
config.json
pytorch_model-00001-of-00012.bin
pytorch_model-00002-of-00012.bin
pytorch_model-00003-of-00012.bin
pytorch_model-00004-of-00012.bin
pytorch_model-00005-of-00012.bin
pytorch_model-00006-of-00012.bin
pytorch_model-00007-of-00012.bin
pytorch_model-00008-of-00012.bin
pytorch_model-00009-of-00012.bin
pytorch_model-00010-of-00012.bin
pytorch_model-00011-of-00012.bin
pytorch_model-00012-of-00012.bin
pytorch_model.bin.index.json
special_tokens_map.json
spiece.model
tokenizer.json
tokenizer_config.json
```