"""
Custom dataset for DAPO recipe.

Adapted from recipe/papo/rl_dataset_spec.py.  Key differences:
- Uses the qwen3_thinking prompt template (hardcoded, no config field needed).
- No prompt_template_type dependency — avoids adding papo-specific fields to
  the verl/DAPO config.
- pandas-based loading handles heterogeneous val-dataset schemas gracefully.

Usage via config overrides::

    data.custom_cls.path=/path/to/recipe/dapo/src/dapo_dataset.py
    data.custom_cls.name=DAPOMathDataset
"""

import copy
import logging
import os
import re
from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{input}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
    "<|im_end|>\n<|im_start|>assistant\n"
)


def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)
    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)
    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)
    return {**tensors, **non_tensors}


class DAPOMathDataset(Dataset):

    def __init__(
        self,
        data_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.serialize_dataset = False

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dfs = [pd.read_parquet(f) for f in self.data_files]
        self.dataframe = pd.concat(dfs, ignore_index=True)
        print(f"dataset len: {len(self.dataframe)}")

        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe[
                self.dataframe.apply(
                    lambda doc: len(
                        tokenizer.tokenize(
                            _PROMPT_TEMPLATE.replace("{input}", doc[prompt_key][0]["content"])
                        )
                    ) <= self.max_prompt_length,
                    axis=1,
                )
            ].reset_index(drop=True)
            print(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)
            self._read_files_and_tokenize()
        else:
            print("old dataloader ckpt file is used, please train from scratch")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})
                message["content"] = content_list
        return messages

    def __getitem__(self, item):
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row_dict)

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}
            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images
            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]
            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)
        else:
            raw_prompt = _PROMPT_TEMPLATE.replace("{input}", messages[0]["content"])
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} > {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids

        if self.return_raw_chat:
            row_dict["raw_prompt"] = raw_prompt

        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if "dataframe" in state:
                del state["dataframe"]
            return state
        return self.__dict__.copy()
