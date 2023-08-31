# TransCoder Package
This is a library implementation of the infamous Transcoder model [Unsupervised Translation of Programming Languages](https://arxiv.org/pdf/2006.03511.pdf)
![Model](https://dl.fbaipublicfiles.com/transcoder/TransCoder_Schema.jpg)

## Usage
### Installation
Create a virtual environment and activate it:
```bash
virtualenv venv --python=python3.8
source venv/bin/activate
```
Install library
```bash
pip install transcode
```

If your `libclang.so` is not in `/usr/lib/llvm-10/lib/`, replace the path to `libclang.so` to the correct path in `clang.cindex.Config.set_library_path('path_to_libclang')` in `code_tokenizer.py` 


## Translate
Just provide the target and source langauges and the code, and we handle the rest :wink:
```bash
translate --src_lang python --tgt_lang java < input_code.cpp
```

## Train or Evaluate
To train or evaluate the model, please refer to the [original repository](https://github.com/facebookresearch/TransCoder).

