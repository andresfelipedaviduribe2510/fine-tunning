# Fine-Tuning Workshop: Unsloth & LoRA

## Slide 1: Introduction
- **Code Preview:**
  ```python
  import unsloth
  model = load_llm()
  adapter = LoRA(r=16)
  trainer.train()
  model.save() ✓
  ```
- **Title:** TALLER IA - Fine-Tuning con Unsloth & LoRA
- **Topic:** Pre-entrenando tu primer LLM en Google Colab
- **Key Areas:** Colab & GPU, Librerías, LoRA Adapters, Código paso a paso.

## Slide 2: 01 Google Colab — La computadora en la nube
- **Hardware:** Tesla T4 GPU — GRATIS (`$ nvidia-smi` → Tesla T4 | 16 GB VRAM)
- **What is it?** Cloud computer with professional GPU.
- **Why?** Fine-tuning needs GPU; laptops usually lack this power.
- **Tool:** Notebook (cell-based code, instant results).
- **Free Access:** Tesla T4 is enough for Llama-3-8B.
- **Link:** colab.research.google.com

## Slide 3: 02 El Stack de Instalación — ¿Qué instalamos y por qué?
- **Commands:**
  ```bash
  !pip install "unsloth[colab-new] @ git+..."
  !pip install --no-deps trl peft accelerate bitsandbytes
  ```
- **Unsloth:** Racing engine. 2x faster, 70% less memory. Prevents OOM in Colab.
- **PEFT:** Parameter-Efficient Fine-Tuning. Trains ONLY adapters, not the whole model.
- **TRL:** Provides `SFTTrainer`, injecting data into the model.
- **BitsAndBytes:** Compressor. Loads 4-bit quantization (like a lightweight PDF).
- **Accelerate:** Optimizes Python ↔ GPU communication.

## Slide 4: 03 Python · PyTorch · Caché — El motor detrás de todo
- **Python:** The universal AI language for Llama, GPT, Claude. Handles massive tensors.
- **PyTorch (torch):** Physics engine of AI. Mathematical calculations (tensors) on GPU.
- **The Cache: `use_cache`**
  - **`use_cache = False` (Training):** Prevents the model from getting confused with old responses while learning. No OOM errors.
  - **`use_cache = True` (Inference):** Fast response time when testing the trained model.

## Slide 5: 04 El Modelo y los Adaptadores LoRA — El corazón del taller
- **Llama-3-8B:** Meta AI, 8B parameters, 4-bit quantized (via BitsAndBytes).
- **Chef Analogy:** The model is a chef (already knows how to cook). Fine-tuning is giving it a specific recipe (e.g., Goku). The chef still knows how to cook but follows the new recipe.
- **LoRA Adapters:**
  - Base Model (Read only, ~5GB) + LoRA (Adapters, ~50MB).
  - **`r=16`:** Adapter rank. Higher = more learning (and weight).
  - **`lora_alpha=16`:** Learning intensity.
  - **`target_modules`:** Specific model layers (attention) to attach adapters.

## Slide 6: 05 Código en 5 Sesiones — Del setup al entrenamiento
1. **Session 1:** Install libraries (`unsloth`, `trl`, `peft`, `bitsandbytes`, `accelerate`).
2. **Session 2:** Load Model + LoRA (`FastLanguageModel.from_pretrained()`, `get_peft_model()`).
3. **Session 3:** Test BASE model (`model.generate()`).
4. **Session 4:** Load Dataset (`load_dataset("json")` → `format_goku()`).
5. **Session 5:** Training (`trainer.train()`).

## Slide 7: 06 El Código Limpio — Ejecútalo celda por celda
- **Cell 1: Installation**
  ```python
  !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  !pip install --no-deps trl peft accelerate bitsandbytes
  ```
- **Cell 2: Model + LoRA**
  ```python
  from unsloth import FastLanguageModel; import torch
  model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/llama-3-8b-bnb-4bit", max_seq_length=2048, load_in_4bit=True)
  model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth")
  ```
- **Cell 3: Base Test** (Setting `use_cache=False` for training-like config check)
- **Cell 4: Dataset Loading & Formatting (Goku style)**
- **Cell 5: Training Execution**

## Slide 8: Summary — What you learned today
- **Colab:** Cloud GPU (Tesla T4).
- **Libraries:** Unsloth, PEFT, TRL, BitsAndBytes, Accelerate.
- **Models:** Llama-3-8B (4-bit).
- **LoRA:** Small learning layers.
- **PyTorch:** Mathematical engine (tensors).
- **Caching:** `False` for learning, `True` for talking.
- **Result:** You can now teach Fine-tuning with Unsloth!
