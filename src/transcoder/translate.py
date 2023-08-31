"""
Copyright (c) 2023-present, Ahmed Heakl
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Translate sentences from the input stream.
The model will be faster is sentences are sorted by length.
Input sentences must have the same tokenization and BPE codes than the ones used in the model.

Usage:
    python translate.py
    --src_lang cpp --tgt_lang java \
    --model_path trained_model.pth < input_code.cpp
"""

from typing import Tuple
import argparse
import os
import sys

import fastBPE
import torch
from google_drive_downloader import GoogleDriveDownloader as gdd

import transcoder.preprocessing.src.code_tokenizer as code_tokenizer
from transcoder.XLM.src.data.dictionary import (
    Dictionary,
    BOS_WORD,
    EOS_WORD,
    PAD_WORD,
    UNK_WORD,
    MASK_WORD,
)
from transcoder.XLM.src.model import build_model
from transcoder.XLM.src.utils import AttrDict

SUPPORTED_LANGUAGES = ["cpp", "java", "python"]
FIRST_MODEL_LINK = "https://dl.fbaipublicfiles.com/transcoder/model_1.pth"
SECOND_MODEL_LINK = "https://dl.fbaipublicfiles.com/transcoder/model_2.pth"
BPE_CODE_LINK = (
    "https://drive.google.com/file/d/1yVgA8opMghe1_IDrF8hNU-DL7i1xsATp/view?usp=sharing"
)
FIRST_MODEL_PAIRS = ["cpp -> java", "java -> cpp", "java -> python"]

SECOND_MODEL_PAIRS = ["cpp -> python", "python -> cpp", "python -> java"]


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Translate sentences")

    # model
    parser.add_argument("--model_path", type=str, default="", help="Model path")
    parser.add_argument(
        "--src_lang",
        type=str,
        default="cpp",
        help=f"Source language, should be either {', '.join(SUPPORTED_LANGUAGES[:-1])} or {SUPPORTED_LANGUAGES[-1]}",
    )
    parser.add_argument(
        "--tgt_lang",
        type=str,
        default="python",
        help=f"Target language, should be either {', '.join(SUPPORTED_LANGUAGES[:-1])} or {SUPPORTED_LANGUAGES[-1]}",
    )
    parser.add_argument(
        "--BPE_path",
        type=str,
        default="data/BPE_with_comments_codes",
        help="Path to BPE codes.",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=1,
        help="Beam size. The beams will be printed in order of decreasing likelihood.",
    )

    return parser


def download_model(first_model: bool = True) -> Tuple[str, str]:
    """Download model weights in .cache/transcoder folder

    Args:
        first_model (bool, optional): Which model to download.
        1st model: C++ -> Java, Java -> C++ and Java -> Python
        2nd Model: C++ -> Python, Python -> C++ and Python -> Java
        Defaults to True.

    Returns:
        Tuple[str, str]: path to the downloaded model
    """
    if first_model:
        model_link = FIRST_MODEL_LINK
    else:
        model_link = SECOND_MODEL_LINK

    common_path = os.path.join(os.path.expanduser("~"), ".cache", "transcoder")
    if not os.path.exists(common_path):
        os.makedirs(common_path)
    model_path = os.path.join(common_path, f"model_{first_model}.pth")
    if not os.path.exists(model_path):
        print("Downloading model weights...")
        os.system(f"wget {model_link} -O {model_path}")
    bpe_path = os.path.join(common_path, "BPE_with_comments_codes")
    if not os.path.exists(bpe_path):
        print("Downloading BPE codes...")
        url_id = BPE_CODE_LINK.split("/")[-2]
        gdd.download_file_from_google_drive(file_id=url_id, dest_path=bpe_path)
    return model_path, bpe_path


class Translator:
    """Translator class implementation of transcoder"""

    def __init__(self, params):
        reloaded = torch.load(params.model_path, map_location="cpu")
        reloaded["encoder"] = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in reloaded["encoder"].items()
        }
        assert "decoder" in reloaded or (
            "decoder_0" in reloaded and "decoder_1" in reloaded
        )
        if "decoder" in reloaded:
            decoders_names = ["decoder"]
        else:
            decoders_names = ["decoder_0", "decoder_1"]
        for decoder_name in decoders_names:
            reloaded[decoder_name] = {
                (k[len("module.") :] if k.startswith("module.") else k): v
                for k, v in reloaded[decoder_name].items()
            }

        self.reloaded_params = AttrDict(reloaded["params"])

        # build dictionary / update parameters
        self.dico = Dictionary(
            reloaded["dico_id2word"], reloaded["dico_word2id"], reloaded["dico_counts"]
        )
        assert self.reloaded_params.n_words == len(self.dico)
        assert self.reloaded_params.bos_index == self.dico.index(BOS_WORD)
        assert self.reloaded_params.eos_index == self.dico.index(EOS_WORD)
        assert self.reloaded_params.pad_index == self.dico.index(PAD_WORD)
        assert self.reloaded_params.unk_index == self.dico.index(UNK_WORD)
        assert self.reloaded_params.mask_index == self.dico.index(MASK_WORD)

        # build model / reload weights
        self.reloaded_params["reload_model"] = ",".join([params.model_path] * 2)
        encoder, decoder = build_model(self.reloaded_params, self.dico)

        self.encoder = encoder[0]
        self.encoder.load_state_dict(reloaded["encoder"])
        assert len(reloaded["encoder"].keys()) == len(
            list(p for p, _ in self.encoder.state_dict().items())
        )

        self.decoder = decoder[0]
        self.decoder.load_state_dict(reloaded["decoder"])
        assert len(reloaded["decoder"].keys()) == len(
            list(p for p, _ in self.decoder.state_dict().items())
        )

        self.encoder.cuda()
        self.decoder.cuda()

        self.encoder.eval()
        self.decoder.eval()
        self.bpe_model = fastBPE.fastBPE(os.path.abspath(params.BPE_path))

    def translate(
        self,
        input_code,
        lang1,
        lang2,
        n: int = 1,
        beam_size: int = 1,
        sample_temperature=None,
        device: str = "cuda:0",
    ):
        """Translate input code from lang1 to lang2"""
        with torch.no_grad():
            assert lang1 in {"python", "java", "cpp"}, lang1
            assert lang2 in {"python", "java", "cpp"}, lang2

            tokenizer = getattr(code_tokenizer, f"tokenize_{lang1}")
            detokenizer = getattr(code_tokenizer, f"detokenize_{lang2}")
            lang1 += "_sa"
            lang2 += "_sa"

            lang1_id = self.reloaded_params.lang2id[lang1]
            lang2_id = self.reloaded_params.lang2id[lang2]

            tokens = list(tokenizer(input_code))
            tokens = self.bpe_model.apply(tokens)
            tokens = ["</s>"] + tokens + ["</s>"]
            input_code = " ".join(tokens)
            # create batch
            len1 = len(input_code.split())
            len1_tensor = torch.LongTensor(1).fill_(len1).to(device)

            x1 = torch.LongTensor([self.dico.index(w) for w in input_code.split()]).to(
                device
            )[:, None]
            langs1 = x1.clone().fill_(lang1_id)

            enc1 = self.encoder(
                "fwd", x=x1, lengths=len1_tensor, langs=langs1, causal=False
            )
            enc1 = enc1.transpose(0, 1)
            if n > 1:
                enc1 = enc1.repeat(n, 1, 1)
                len1_tensor = len1_tensor.expand(n)

            if beam_size == 1:
                out, _ = self.decoder.generate(
                    enc1,
                    len1_tensor,
                    lang2_id,
                    max_len=int(
                        min(
                            self.reloaded_params.max_len,
                            3 * len1_tensor.max().item() + 10,
                        )
                    ),
                    sample_temperature=sample_temperature,
                )
            else:
                out, _ = self.decoder.generate_beam(
                    enc1,
                    len1_tensor,
                    lang2_id,
                    max_len=int(
                        min(
                            self.reloaded_params.max_len,
                            3 * len1_tensor.max().item() + 10,
                        )
                    ),
                    early_stopping=False,
                    length_penalty=1.0,
                    beam_size=beam_size,
                )
            tok = []
            for i in range(out.shape[1]):
                wid = [self.dico[out[j, i].item()] for j in range(len(out))][1:]
                wid = wid[: wid.index(EOS_WORD)] if EOS_WORD in wid else wid
                tok.append(" ".join(wid).replace("@@ ", ""))

            results = []
            for token in tok:
                results.append(detokenizer(token))
            return results


def main():
    """Main runner for translating programming languages"""
    parser = get_parser()
    params = parser.parse_args()

    assert (
        params.src_lang in SUPPORTED_LANGUAGES
    ), f"The source language should be in {SUPPORTED_LANGUAGES}."
    assert (
        params.tgt_lang in SUPPORTED_LANGUAGES
    ), f"The target language should be in {SUPPORTED_LANGUAGES}."

    src_target_pair = f"{params.src_lang} -> {params.tgt_lang}"
    model_weight, bpe_codes = download_model(
        first_model=src_target_pair in FIRST_MODEL_PAIRS
    )
    params.model_path = model_weight
    params.BPE_path = bpe_codes

    translator = Translator(params)

    input_lines = sys.stdin.read().strip()

    with torch.no_grad():
        output = translator.translate(
            input_lines,
            lang1=params.src_lang,
            lang2=params.tgt_lang,
            beam_size=params.beam_size,
        )

    for out in output:
        print("=" * 20)
        print(out)


if __name__ == "__main__":
    main()
