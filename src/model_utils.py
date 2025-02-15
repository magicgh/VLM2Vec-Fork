import logging
import torch

from src.utils import print_master

from src.vlm_backbone.llava_next import LlavaNextForConditionalGeneration
from src.vlm_backbone.phi3_v.modeling_phi3_v import Phi3VForCausalLM
from src.vlm_backbone.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from src.vlm_backbone.qwen2_vl import Qwen2VLForConditionalGeneration
from src.vlm_backbone.qwen2_vl.modeling_qwen2_vl import Qwen2MLP

logger = logging.getLogger(__name__)


PHI_IMAGE_TOKEN_MAX_INPUT_ID = int(1e9)
LLAVA_IMAGE_TOKEN_ID = 32000

PHI3V = 'phi3_v'
LLAVA_NEXT = 'llava_next'
QWEN2_VL = 'qwen2_vl'
QWEN2_5_VL = 'qwen2_5_vl'
MODEL2BACKBONE = {  # keys are from hf_config.model_type
    'phi3_v': PHI3V,
    'llava_next': LLAVA_NEXT,
    'qwen2_vl': QWEN2_VL,
    'qwen2_5_vl': QWEN2_5_VL,
}
SUPPORTED_MODELS = set(MODEL2BACKBONE.keys())

vlm_image_tokens = {
    PHI3V: "<|image_1|>",
    LLAVA_NEXT: "<image>",
    QWEN2_VL: "<|image_pad|>",
    QWEN2_5_VL: "<|image_pad|>",
}

backbone2model = {
    PHI3V: Phi3VForCausalLM,
    LLAVA_NEXT: LlavaNextForConditionalGeneration,
    QWEN2_VL: Qwen2VLForConditionalGeneration,
    QWEN2_5_VL: Qwen2_5_VLForConditionalGeneration,
}

def load_processor(model_args):
    """
    Load processor based on VLM backbone.
    """
    print_master('Loading processor')
    model_name = model_args.processor_name if model_args.processor_name else model_args.model_name
    if model_args.model_backbone == PHI3V:
        from src.vlm_backbone.phi3_v.processing_phi3_v import Phi3VProcessor
        processor = Phi3VProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
            num_crops=model_args.num_crops
        )
        processor.tokenizer.padding_side = "right"
    elif model_args.model_backbone == LLAVA_NEXT:
        from transformers import LlavaNextProcessor
        processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",
            trust_remote_code=True
        )
    elif model_args.model_backbone == QWEN2_VL:
        from src.vlm_backbone.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor
        from src.vlm_backbone.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from src.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_name)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
        processor = Qwen2VLProcessor.from_pretrained(
            model_name,
            image_processor=image_processor, tokenizer=tokenizer,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
            uigraph_train=model_args.uigraph_train, uigraph_test=model_args.uigraph_test,
            uigraph_diff=model_args.uigraph_diff,  uigraph_rand=model_args.uigraph_rand,
            uimask_pre=model_args.uimask_pre, uimask_ratio=model_args.uimask_ratio, uimask_rand=model_args.uimask_rand
        )
    elif model_args.model_backbone == QWEN2_5_VL:
        from src.vlm_backbone.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from src.vlm_backbone.qwen2_5_vl.image_processing_qwen2_5_vl import Qwen2_5_VLImageProcessor
        from src.vlm_backbone.qwen2_vl.tokenization_qwen2_fast import Qwen2TokenizerFast
        image_processor = Qwen2_5_VLImageProcessor.from_pretrained(model_name)
        tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_name, image_processor=image_processor, tokenizer=tokenizer)
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(
            model_args.processor_name if model_args.processor_name else model_args.model_name,
            trust_remote_code=True,
        )
    return processor


def get_backbone_name(hf_config):
    assert hf_config.model_type in SUPPORTED_MODELS, f"Unknown backbone name {hf_config.model_type}.Supported models are {SUPPORTED_MODELS}"
    return MODEL2BACKBONE[hf_config.model_type]


def Llava_NEXT_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(images=None, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(images=image, text=text, return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        # dummy image inputs based on the first valid data point
        pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
        image_size_for_padding = torch.from_numpy(list(v for v in image_sizes if v is not None)[0])
        # make the batch full tensors
        pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
        pixel_values = torch.cat(pixel_values, dim=0)
        image_sizes = [torch.from_numpy(v) if v is not None else image_size_for_padding for v in image_sizes]
        image_sizes = torch.cat(image_sizes, dim=0)
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Phi3V_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(text, None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
        else:
            image_exists = True
            inputs = processor(text=text, images=[image], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            if 'image_sizes' in inputs:
                image_sizes.append(inputs['image_sizes'])
            if 'image_grid_thw' in inputs:
                image_grid_thw.append(inputs['image_grid_thw'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_sizes'] = image_sizes
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_sizes'] = torch.ones(input_ids.shape[0], 1)

    return inputs


def Qwen2_VL_process_fn(model_inputs: dict, processor, max_length=None):
    input_ids, pixel_values, image_sizes, image_grid_thw = [], [], [], []
    patch_assign, patch_assign_sep, patch_assign_len, patch_pos, select_mask = [], [], [], [], []
    texts, images = model_inputs['text'], model_inputs['image']
    image_exists = False
    # 1. iterate each pair and process (since processors do not support batch processing)
    for text, image in zip(texts, images):
        if image is None:
            inputs = processor(text=[text], images=None, return_tensors="np", max_length=max_length, truncation=True)
            input_id = inputs["input_ids"].squeeze().tolist()
            if isinstance(input_id, int):
                # in case of empty string, only BOS is included
                input_id = [input_id]
            input_ids.append(input_id)
            pixel_values.append(None)
            image_sizes.append(None)
            image_grid_thw.append(None)
            patch_assign.append(None)
            patch_assign_sep.append(None)
            patch_assign_len.append(None)
            patch_pos.append(None)
            select_mask.append(None)
        else:
            image_exists = True
            inputs = processor(images=[image], text=[text], return_tensors="np", max_length=max_length, truncation=True)
            input_ids.append(inputs["input_ids"].squeeze().tolist())
            pixel_values.append(inputs['pixel_values'])
            image_grid_thw.append(inputs['image_grid_thw'])
            if 'patch_assign' in inputs:
                patch_assign.append(inputs['patch_assign'])
            if 'patch_assign_sep' in inputs:
                patch_assign_sep.append(inputs['patch_assign_sep'])
            if 'patch_assign_len' in inputs:
                patch_assign_len.append(inputs['patch_assign_len'])
            if 'patch_pos' in inputs:
                patch_pos.append(inputs['patch_pos'])
            if 'select_mask' in inputs:
                select_mask.append(inputs['select_mask'])

    # 2. padding inputs
    batch_encoding = processor.tokenizer.pad({'input_ids': input_ids}, return_tensors="pt")
    input_ids, attention_mask = batch_encoding['input_ids'], batch_encoding['attention_mask']
    # manually enforce long type due to:
    # (1) [rank7]: RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
    # (2) [rank7]:   File "/fsx/home/ruimeng/project/VLM2Vec/src/model.py", line 45, in _pooling
    #     [rank7]:     reps = last_hidden_state[
    #     [rank7]: IndexError: tensors used as indices must be long, int, byte or bool tensors
    inputs = {
        'input_ids': input_ids.long(),
        'attention_mask': attention_mask.long(),
        'texts': texts,
        'images': images,
    }
    # 3. special postcare for mixed batch (examples w/ and w/o images in the same batch)
    if image_exists:
        pixel_value_shape_for_padding = list(v.shape for v in pixel_values if v is not None)[0]
        pixel_values = [torch.from_numpy(v) if v is not None else torch.zeros(pixel_value_shape_for_padding) for v in pixel_values]
        pixel_values = torch.stack(pixel_values, dim=0)

        if patch_assign:
            patch_assign_shape_for_padding = list(v.shape for v in patch_assign if v is not None)[0]
            patch_assign = [torch.from_numpy(v) if v is not None else torch.zeros(patch_assign_shape_for_padding) for v in patch_assign]
            patch_assign = torch.cat(patch_assign, dim=0)
            inputs['patch_assign'] = patch_assign
        if patch_assign_sep:
            patch_assign_sep_shape_for_padding = list(v.shape for v in patch_assign_sep if v is not None)[0]
            patch_assign_sep = [torch.from_numpy(v) if v is not None else torch.zeros(patch_assign_sep_shape_for_padding) for v in patch_assign_sep]
            patch_assign_sep = torch.cat(patch_assign_sep, dim=0)
            inputs['patch_assign_sep'] = patch_assign_sep
        if patch_assign_len:
            patch_assign_len_shape_for_padding = list(v.shape for v in patch_assign_len if v is not None)[0]
            patch_assign_len = [torch.from_numpy(v) if v is not None else torch.zeros(patch_assign_len_shape_for_padding) for v in patch_assign_len]
            patch_assign_len = torch.cat(patch_assign_len, dim=0)
            inputs['patch_assign_len'] = patch_assign_len
        if patch_pos:
            patch_pos_shape_for_padding = list(v.shape for v in patch_pos if v is not None)[0]
            key_tmp = [torch.from_numpy(v) if v is not None else torch.zeros(patch_pos_shape_for_padding) for v in patch_pos]
            max_length = key_tmp[0].size(0)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(0)), value=0) for pos in key_tmp]
            patch_pos = torch.cat(padded_key, dim=0)
            inputs['patch_pos'] = patch_pos
        if select_mask:
            select_mask_shape_for_padding = list(v.shape for v in select_mask if v is not None)[0]
            key_tmp = [torch.from_numpy(v) if v is not None else torch.zeros(select_mask_shape_for_padding) for v in select_mask]
            max_length = key_tmp[0].size(0)
            padded_key = [torch.nn.functional.pad(pos, (0, max_length - pos.size(0)), value=0) for pos in key_tmp]
            select_mask = torch.cat(padded_key, dim=0)
            inputs['select_mask'] = select_mask

        # image_grid_thw = np.concatenate(image_grid_thw, axis=0)
        # add them to inputs
        inputs['pixel_values'] = pixel_values
        inputs['image_grid_thw'] = image_grid_thw
    else:
        inputs['pixel_values'] = torch.zeros(input_ids.shape[0], 1)
        inputs['image_grid_thw'] = [None] * input_ids.shape[0]

    return inputs


process_vlm_inputs_fns = {
    PHI3V: Phi3V_process_fn,
    LLAVA_NEXT: Llava_NEXT_process_fn,
    QWEN2_VL: Qwen2_VL_process_fn,
    QWEN2_5_VL: Qwen2_VL_process_fn,
}
