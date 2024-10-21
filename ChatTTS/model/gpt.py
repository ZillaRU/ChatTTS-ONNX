import os, platform
from dataclasses import dataclass
import logging
from typing import Union, List, Optional, Tuple, Callable
import gc
from pathlib import Path
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from einops import rearrange
# from torch.nn.utils.parametrizations import weight_norm
from tqdm import tqdm

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import is_flash_attn_2_available

from ..utils import del_all

module_path = "./ChatTTS/model"

if module_path not in sys.path:
    sys.path.append(module_path)

class GPT(nn.Module):
    def __init__(
        self,
        model_path: str,
        gpt_config: dict,
        tpu_id: int = 0,
        logger=logging.getLogger(__name__),
    ):
        super().__init__()

        self.logger = logger

        self.generator = torch.Generator(device=torch.device("cpu"))

        self.config = gpt_config
        self.num_vq = int(gpt_config["num_vq"])
        self.num_audio_tokens = int(gpt_config["num_audio_tokens"])
        self.num_text_tokens = int(gpt_config["num_text_tokens"])

        import llama
        self.gpt = llama.TTSLlama()
        self.gpt.init([tpu_id], model_path)

        # 添加新的属性
        self.gpt.max_new_tokens = 512
        self.gpt.SEQLEN = 512
        self.gpt.top_p = 0.7
        self.gpt.DEBUGGING = False
        self.gpt.temperature = 0.7
        self.gpt.repeat_penalty = 1.0


    class Context:
        def __init__(self):
            self._interrupt = False

        def set(self, v: bool):
            self._interrupt = v

        def get(self) -> bool:
            return self._interrupt


    def prepare(self, compile=False):
        if self.use_flash_attn and is_flash_attn_2_available():
            self.gpt = self.gpt.to(dtype=torch.float16)
        if compile and not self.is_te_llama and not self.is_vllm:
            try:
                self.compile(backend="inductor", dynamic=True)
                self.gpt.compile(backend="inductor", dynamic=True)
            except RuntimeError as e:
                self.logger.warning(f"compile failed: {e}. fallback to normal mode.")

    def __call__(
        self, input_ids: torch.Tensor, text_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        get_emb
        """
        return super().__call__(input_ids, text_mask)

    @dataclass(repr=False, eq=False)
    class GenerationOutputs:
        ids: List[torch.Tensor]
        attentions: List[Optional[Tuple[torch.FloatTensor, ...]]]
        hiddens: List[torch.Tensor]

        def destroy(self):
            del_all(self.ids)
            del_all(self.attentions)
            del_all(self.hiddens)

    @torch.no_grad()
    def _prepare_generation_outputs(
        self,
        inputs_ids: torch.Tensor,
        start_idx: int,
        end_idx: torch.Tensor,
        attentions: List[Optional[Tuple[torch.FloatTensor, ...]]],
        hiddens: List[torch.Tensor],
        infer_text: bool,
    ) -> GenerationOutputs:
        inputs_ids = [
            inputs_ids[idx].narrow(0, start_idx, i) for idx, i in enumerate(end_idx)
        ]
        if infer_text:
            inputs_ids = [i.narrow(1, 0, 1).squeeze_(1) for i in inputs_ids]

        if len(hiddens) > 0:
            hiddens = torch.stack(hiddens, 1)
            hiddens = [
                hiddens[idx].narrow(0, 0, i) for idx, i in enumerate(end_idx.int())
            ]

        return self.GenerationOutputs(
            ids=inputs_ids,
            attentions=attentions,
            hiddens=hiddens,
        )

    def generate_text(
        self, 
        inputs_ids,
        temperature, 
        eos_token, 
        attention_mask = None,
        max_new_token = 500, 
        min_new_token = 0,
        LogitsWarpers = [],
        LogitsProcessors = [],
        return_hidden=False,
    ):
        inputs_ids_list = inputs_ids[0].tolist()
        if isinstance(eos_token, torch.Tensor):
            eos_token = eos_token.item()
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.item()
        self.gpt.temperature = temperature
        self.gpt.repeat_penalty = 1.0
        inputs_ids = self.gpt.generate_text(inputs_ids_list, eos_token, temperature)
        inputs_ids = torch.tensor(inputs_ids, dtype=torch.int64).unsqueeze(0).unsqueeze(0)
        return inputs_ids

    @torch.no_grad()
    def generate(
        self,
        inputs_ids: torch.Tensor,
        temperature: torch.Tensor,
        eos_token: Union[int, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        max_new_token=2048,
        min_new_token=0,
        logits_processors: Tuple[
            Callable[[torch.LongTensor, torch.FloatTensor], torch.FloatTensor]
        ] = (),
        infer_text=False,
        spk_emb=None,
        return_attn=False,
        return_hidden=False,
        stream=False,
        show_tqdm=True,
        ensure_non_empty=True,
        stream_batch=24,
        manual_seed: Optional[int] = None,
        context=Context(),
    ):
        if infer_text:
            # 文本生成不支持stream输出
            inputs_ids = self.generate_text(
                inputs_ids,
                temperature,
                eos_token,
                attention_mask,
                max_new_token,
                min_new_token,
                logits_processors,
                return_hidden,
            )
            yield self.GenerationOutputs(
                ids=[inputs_ids],
                attentions=[],
                hiddens=[],
            )
        else: # audio code generation 支持stream输出
            temperature = temperature.unsqueeze(1)
            attentions = []
            hiddens = []
            stream_iter = 0

            start_idx, end_idx = inputs_ids.shape[1], torch.zeros(inputs_ids.shape[0], device=inputs_ids.device, dtype=torch.long)
            finish = torch.zeros(inputs_ids.shape[0], device=inputs_ids.device).bool()

            pbar = tqdm(total=max_new_token, desc="code", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]") if show_tqdm else None

            for i in range(max_new_token):
                if i == 0:        
                    logits, hidden = self.gpt.forward_first_code_core(inputs_ids[0].tolist(), *self.get_speaker(inputs_ids, spk_emb))
                    inputs_ids = inputs_ids.unsqueeze(2).expand(-1, -1, self.num_vq)
                else:
                    logits, hidden = self.gpt.forward_next_code_core(curr_input_id)
                
                hiddens.append(torch.tensor(hidden, dtype=torch.float32).unsqueeze(0))
                logits = torch.tensor(logits).reshape(self.num_audio_tokens, self.num_vq).transpose(0, 1)
                
                # 应用logits处理器
                # logits_token = inputs_ids[:, -1:]
                # logits_token = rearrange(logits_token, "b c n -> (b n) c") #####
                inputs_ids_sliced = inputs_ids.narrow(
                    1,
                    start_idx,
                    inputs_ids.size(1) - start_idx,
                ).permute(0, 2, 1)
                logits_token = inputs_ids_sliced.reshape(
                    inputs_ids_sliced.size(0) * inputs_ids_sliced.size(1),
                    -1,
                )
                del inputs_ids_sliced

                for processor in logits_processors:
                    logits = processor(logits_token, logits)
                
                if i < min_new_token:
                    logits[:, eos_token] = -float('inf')
                
                scores = F.softmax(logits / temperature, dim=-1)
                idx_next = torch.multinomial(scores, num_samples=1)
                
                idx_next = idx_next.view(-1, self.num_vq)
                finish |= (idx_next == eos_token).any(1)
                inputs_ids = torch.cat([inputs_ids, idx_next.unsqueeze(1)], 1)
                curr_input_id = inputs_ids[0, -1].int().tolist()
                
                not_finished = ~finish
                end_idx += not_finished.int()
                stream_iter += not_finished.any().int()

                if stream and stream_iter > 0 and stream_iter % stream_batch == 0:
                    yield self._prepare_generation_outputs(
                        inputs_ids,
                        start_idx,
                        end_idx,
                        attentions,
                        hiddens,
                        infer_text,
                    )

                if finish.all() or context.get():
                    break

                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()

            if not finish.all():
                if context.get():
                    self.logger.warning("generation is interrupted")
                else:
                    self.logger.warning(f"incomplete result. hit max_new_token: {max_new_token}")

            yield self._prepare_generation_outputs(
                inputs_ids,
                start_idx,
                end_idx,
                attentions,
                hiddens,
                infer_text,
            )

    def get_speaker(self, inputs_ids, spk_emb):
        temp = torch.where(inputs_ids[0] == 21143)
        if temp[0].shape[0] == 0:
            spk_idx = -1
            spk_emb = list(range(768))
            self.logger.info("Not set speaker")
        else:
            spk_idx = temp[0].item()
            spk_emb = spk_emb.tolist()
        return spk_idx, spk_emb