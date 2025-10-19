# Zero-Shot Fine-Grained Image Classification Using Large Vision-Language Models (EMNLP 2025)
# Abstract: 
Large Vision-Language Models (LVLMs) have demonstrated impressive performance on vision-language reasoning tasks. However, their potential for zero-shot fine-grained image classification, a challenging task requiring precise differentiation between visually similar categories, remains underexplored. We present a novel method that transforms zero-shot fine-grained image classification into a visual question-answering framework, leveraging LVLMs' comprehensive understanding capabilities rather than relying on direct class name generation. We enhance model performance through a novel attention intervention technique. We also address a key limitation in existing datasets by developing more comprehensive and precise class description benchmarks. We validate the effectiveness of our method through extensive experimentation across multiple fine-grained image classification benchmarks. Our proposed method consistently outperforms the current state-of-the-art (SOTA) approach, demonstrating both the effectiveness of our method and the broader potential of LVLMs for zero-shot fine-grained classification tasks.

---
# Code
**Attention Intervention with LLaVA-v1.5-7B Model:**

The `modeling_llama.py` file contains the code for our introduced attention intervention method. To run the code, replace the existing `modeling_llama.py` file (typically located in your transformers installation under `transformers/models/llama/`) with this one. Then load your LLaVA-v1.5-7B model with `attn_implementation="eager"`, not with `"flash_attention_2"`.

**Example:**
```python
model_path = "liuhaotian/llava-v1.5-7b" 
disable_torch_init() 
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name, device_map=device, device=device, attn_implementation='eager',
)
```
