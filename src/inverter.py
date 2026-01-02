from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import dispatch_model
from peft import PeftModel
from datasets import load_from_disk 
import re
from utils.PanoHandler import PanoPlus
from utils.pii_handler import match_pii


def top_k_top_p_filtering(
    logits: torch.Tensor,
    top_k: int = 10,
    top_p: float = 1.0,
    filter_value: float = float("-inf"),
) -> torch.Tensor:
    """
    Filters a distribution of logits using top-k and/or nucleus (top-p) filtering.
    Works with shapes [..., V]. Returns a new tensor (doesn't modify in-place).
    """
    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        # Get threshold for each row
        values, _ = torch.topk(logits, top_k, dim=-1)
        min_values = values[..., -1, None]
        logits = logits.masked_fill(logits < min_values, filter_value)

    if top_p is not None and top_p < 1.0:
        # sort by logit descending
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # mask tokens with cumulative prob above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # keep at least the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter back to original indices
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        logits = logits.masked_fill(indices_to_remove, filter_value)

    return logits

@dataclass
class LargeLangModel:
    model_name: str
    lora_model_name: Optional[str] = None
    torch_dtype: torch.dtype = torch.float16

    # Post init structs
    base_model: Optional[AutoModelForCausalLM] = field(init=False, default=None)
    tokenizer: Optional[AutoTokenizer] = field(init=False, default=None)
    n_model_layers: int = field(init=False, default=0)
    model_layers: List[str] = field(init=False, default_factory=list)
    device_map: Dict[str, int] = field(init=False, default_factory=dict)
    model: Optional[AutoModelForCausalLM] = field(init=False, default=None)

    def __post_init__(self) -> None:
        use_cuda = torch.cuda.is_available()
        dtype = self.torch_dtype if use_cuda else torch.float32

        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            trust_remote_code=True,
            **{"low_cpu_mem_usage": True, "use_cache": False}
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=False
        )

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'

        self.n_model_layers = len(self.base_model.model.layers)

        # If no lora_model is provided
        if not self.lora_model_name:
            if use_cuda:
                num_gpus = torch.cuda.device_count()
                for n in range(self.n_model_layers):
                    self.model_layers.append("model.layers." + str(n))
                    self.device_map[self.model_layers[n]] = int(n / self.n_model_layers * num_gpus)

                self.device_map["model.embed_tokens"] = 0
                self.device_map["model.norm"] = num_gpus - 1
                self.device_map["lm_head"] = num_gpus - 1
                try:
                    self.model = dispatch_model(self.base_model, device_map=self.device_map)
                except RuntimeError as e:
                    # Fallback for driver mismatch: stay on CPU
                    print(f"[WARN] dispatch_model failed ({e}); falling back to CPU.")
                    self.device_map = {"": "cpu"}
                    self.model = self.base_model.to("cpu")
                    use_cuda = False

            if not use_cuda:
                self.device_map = {"": "cpu"}
                self.model = self.base_model.to("cpu")

            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False
        
        #If LoRA model is provided
        else:
            if use_cuda:
                num_gpus = torch.cuda.device_count()

                for n in range(self.n_model_layers):
                    layer_name = f"base_model.model.layers.{n}"   # was base_model.model.model.layers
                    self.model_layers.append(layer_name)
                    self.device_map[layer_name] = int(n / self.n_model_layers * num_gpus)

                self.device_map["base_model.model.embed_tokens"] = 0   # was ...model.model.embed_tokens
                self.device_map["base_model.model.norm"] = num_gpus - 1
                self.device_map["base_model.lm_head"] = num_gpus - 1   # was base_model.model.lm_head

                lora_model = PeftModel.from_pretrained(
                    self.base_model,
                    self.lora_model_name,
                    dtype=dtype
                )

                try:
                    self.model = dispatch_model(lora_model, device_map=self.device_map)
                except RuntimeError as e:
                    print(f"[WARN] dispatch_model failed ({e}); falling back to CPU.")
                    self.device_map = {"": "cpu"}
                    self.model = lora_model.to("cpu")
                    use_cuda = False
            if not use_cuda:
                self.device_map = {"": "cpu"}
                lora_model = PeftModel.from_pretrained(
                    self.base_model,
                    self.lora_model_name,
                    dtype=dtype
                )
                self.model = lora_model.to("cpu")

            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "config"):
                self.model.config.use_cache = False


@dataclass
class Inference:
    model: torch.nn.Module
    tokenizer: AutoTokenizer
    hidden_states: List[torch.Tensor] = field(init=False, default_factory=list)
    hook_handles: List = field(init=False, default_factory=list)

    def _infer_device(self) -> torch.device:
        # Works for AutoModelForCausalLM and PEFT-wrapped models
        core = getattr(self.model, "model", self.model)
        emb = core.get_input_embeddings()
        return emb.weight.device

    def _register_layer_output(self, layer_id: int):
        """Capture output of a single transformer block matching layer_id."""
        def hook(module, inputs, output):
            self.hidden_states.append(output[0] if isinstance(output, tuple) else output)

        target_suffixes = (f"layers.{layer_id}", f"model.layers.{layer_id}")
        for name, module in self.model.named_modules():
            if any(name.endswith(suf) for suf in target_suffixes):
                h = module.register_forward_hook(hook)
                self.hook_handles.append(h)
                break

    def _remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    @torch.no_grad()
    def capture(
        self,
        layer_id: int,
        prompt: Optional[str] = None,
        input_embed: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        assert (prompt is not None) or (input_embed is not None), \
            "Provide either `prompt` or `input_embed`."

        self.hidden_states.clear()
        device = self._infer_device()

        try:
            if prompt is not None:
                toks = self.tokenizer(
                    prompt,
                    padding=True,
                    truncation=False,
                    return_tensors="pt"
                )
                target_input_ids = toks["input_ids"].to(device)
                target_attention_mask = toks["attention_mask"].to(device)
                inputs = {
                    "input_ids": target_input_ids,
                    "attention_mask": target_attention_mask
                }

                self._register_layer_output(layer_id)
                _ = self.model(**inputs)

                # recover the exact input embeddings used
                emb = getattr(self.model, "model", self.model).get_input_embeddings()
                ori_input_embed = emb(target_input_ids)

            else:
                # inputs_embeds path
                target_input_ids = None
                ori_input_embed = input_embed.to(device)
                inputs = {
                    "inputs_embeds": ori_input_embed,
                    "attention_mask": attention_mask.to(device) if attention_mask is not None else None
                }

                self._register_layer_output(layer_id)
                _ = self.model(**inputs)

                target_attention_mask = inputs["attention_mask"]

        finally:
            self._remove_hooks()

        return target_input_ids, target_attention_mask, ori_input_embed, self.hidden_states

@dataclass
class Inversion:
    model: torch.nn.Module
    tokenizer: "AutoTokenizer"
    forward_last_hidden: Callable = field(default=None)  # defaults to internal method
    perplexity_fn: Callable = field(default=None)        # defaults to internal method

    # cached at init
    _core: torch.nn.Module = field(init=False, repr=False)
    _emb: torch.nn.Module = field(init=False, repr=False)
    _device: torch.device = field(init=False, repr=False)

    def __post_init__(self):
        # unwrap PEFT-wrapped models if needed; cache embeddings + device
        self._core = getattr(self.model, "model", self.model)
        self._emb = self._core.get_input_embeddings()
        self._device = self._emb.weight.device
        with torch.no_grad():
            self._emb_min = self._emb.weight.min(dim=0).values
            self._emb_max = self._emb.weight.max(dim=0).values

        if self.forward_last_hidden is None:
            self.forward_last_hidden = self._forward_last_hidden_state
        if self.perplexity_fn is None:
            self.perplexity_fn = self._get_per_token_probs 

    def _forward_last_hidden_state(self, model, input_ids, attention_mask, layer_id, inputs_embeds=None):
        device = self._device
        emb_dtype = self._emb.weight.dtype

        # normalize inputs
        ids = None
        if inputs_embeds is None:
            if isinstance(input_ids, torch.Tensor):
                ids = input_ids
            else:
                ids = torch.tensor(input_ids, dtype=torch.long)

            if ids.dim() < 2:  # [seq] -> [1, seq]
                ids = ids.unsqueeze(0)
            ids = ids.to(device)

        mask = attention_mask.to(device) if attention_mask is not None else None
        new_inputs = {'attention_mask': mask}
        if inputs_embeds is not None:
            new_inputs['inputs_embeds'] = inputs_embeds.to(device=device, dtype=emb_dtype)
        else:
            new_inputs['input_ids'] = ids

        hidden_state_list: List[torch.Tensor] = []
        hook_handles = []

        def forward_hook(module, inputs, output):
            if isinstance(output, tuple):
                for item in output:
                    hidden_state_list.append(item)
            else:
                hidden_state_list.append(output)

        target_suffixes = (f"layers.{layer_id}", f"model.layers.{layer_id}")
        for name, module in model.named_modules():
            if any(name.endswith(suf) for suf in target_suffixes):
                h = module.register_forward_hook(forward_hook)
                hook_handles.append(h)
                break

        _ = model(**new_inputs)

        for h in hook_handles:
            h.remove()

        last_hidden_state = hidden_state_list[0]
        return last_hidden_state
    
    # Evaluates probability that our guessed tokens are right!
    @torch.no_grad()
    def _get_per_token_probs(
        self,
        input_ids,
        model,
        *,
        attention_mask=None,
        filter_top_k=10,
        filter_top_p=1.0,
        next_ids=None,
        top_k=10,
        **kwargs, 
    ):
        device = self._device

        # Normalize inputs
        ids = input_ids if isinstance(input_ids, torch.Tensor) else torch.tensor(input_ids, dtype=torch.long)
        if ids.dim() < 2:
            ids = ids.unsqueeze(0)                           # [1, seq]
        ids = ids.to(device)

        inputs = {'input_ids': ids}
        if attention_mask is not None:
            inputs['attention_mask'] = attention_mask.to(device)

        # Forward → next-token distribution
        logits = model(**inputs).logits[:, -1, :]            # [B, V]
        filtered = top_k_top_p_filtering(logits, top_k=filter_top_k, top_p=filter_top_p)
        probs = F.softmax(filtered, dim=-1)                  # [B, V]

        # Select scores
        if next_ids is not None:
            next_ids = next_ids if isinstance(next_ids, torch.Tensor) else torch.tensor(next_ids, dtype=torch.long, device=probs.device)
            scores = probs[0][next_ids]
            return scores, next_ids
        if top_k is not None:
            top_ids = torch.topk(probs[0], top_k).indices
            scores = probs[0][top_ids]
            return scores, top_ids

        raise NotImplementedError("Provide next_ids or top_k")

    @staticmethod
    def _as_seq(hidden_state: torch.Tensor) -> torch.Tensor:
        """Ensure shape [seq, dim] (squeezes batch if present)."""
        return hidden_state.squeeze(0) if hidden_state.dim() >= 3 else hidden_state

    def _nn_topk(self, vec: torch.Tensor, method: str, top_k: int) -> torch.Tensor:
        """Return top-k token IDs by similarity between vec and embedding matrix."""
        # Ensure both vec and embedding matrix are on the same device (dispatch_model may shard)
        W = self._emb.weight
        if W.device != vec.device:
            W = W.to(vec.device)
        if method == "L2":
            dist = -torch.norm(W - vec, p=2, dim=1)                 # bigger=better
            return torch.topk(dist, top_k).indices
        elif method == "cosine":
            dist = F.cosine_similarity(vec.float(), W.float(), dim=-1).detach().cpu()
            return torch.topk(dist, top_k).indices                  # CPU indices
        else:
            raise NotImplementedError(f"Unknown invert_method: {method}")

    def _pick_first_ascii(
        self,
        cand_ids: torch.Tensor,
        filter_nonascii: bool,
        special_block: Tuple[int, int] = (0, 2)
    ) -> int:
        """Choose first candidate passing simple ASCII and special-token filters."""
        cand_ids = cand_ids.detach().cpu()
        if not filter_nonascii:
            return int(cand_ids[0].item())
        for idx in range(len(cand_ids)):
            tid = int(cand_ids[idx].item())
            s = self.tokenizer.decode([tid])
            if s.isascii() and tid not in special_block:
                return tid
        return int(cand_ids[0].item())

    # ---------- optimization + discretization (paper-inspired) ----------

    def _optimize_embeddings(
        self,
        target_act: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_id: int,
        seq_len: int,
        steps: int = 60,
        lr: float = 5e-2,
        lambda_reg: float = 0.1,
        topk_reg: int = 32,
        clip_bounds: bool = True,
    ) -> torch.Tensor:
        """Constrained optimization toward target activation with manifold regularizer."""
        device = self._device
        emb_weight = self._emb.weight.to(device)
        emb_dtype = emb_weight.dtype

        # init around embedding mean with small noise (no ground-truth leakage)
        emb_mean = emb_weight.mean(dim=0, keepdim=True)
        v_hat = (emb_mean + 0.01 * torch.randn(seq_len, emb_mean.size(1), device=device, dtype=emb_dtype)).clone().detach()
        v_hat.requires_grad_(True)

        opt = torch.optim.Adam([v_hat], lr=lr)
        tgt = target_act.detach()

        for _ in range(steps):
            opt.zero_grad()
            act = self.forward_last_hidden(
                self.model,
                input_ids=None,
                attention_mask=attention_mask,
                layer_id=layer_id,
                inputs_embeds=v_hat.unsqueeze(0)
            )
            # align devices/dtypes
            act = act.to(device)
            tgt = tgt.to(device)
            loss_main = F.mse_loss(act, tgt)

            # manifold regularizer: distance to nearest embeddings (approx via top-k)
            dists = torch.cdist(v_hat, emb_weight)  # [seq, vocab]
            vals, _ = torch.topk(dists, k=min(topk_reg, dists.size(1)), dim=1, largest=False)
            loss_reg = vals.mean()

            loss = loss_main + lambda_reg * loss_reg
            loss.backward()
            opt.step()

            if clip_bounds:
                v_hat.data = torch.max(torch.min(v_hat.data, self._emb_max.to(device)), self._emb_min.to(device))

        return v_hat.detach()

    def _activation_calibrated_decode(
        self,
        v_hat: torch.Tensor,
        target_act: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_id: int,
        add_semantic: bool = True,
        top_k_cos: int = 10,
        top_k_ppl: int = 10,
    ) -> List[int]:
        """Discretize optimized embeddings via activation calibration + semantic speculation."""
        seq_len = v_hat.size(0)
        device = self._device
        base_nearest = []
        for pos in range(seq_len):
            base_nearest.append(int(self._nn_topk(v_hat[pos], "cosine", top_k_cos)[0]))

        ret_list = base_nearest.copy()

        for i in range(seq_len):
            emb_cands = [int(x) for x in self._nn_topk(v_hat[i], "cosine", top_k_cos).tolist()]
            candidate_ids = list(dict.fromkeys(emb_cands))

            if add_semantic and i > 0:
                ppl_vals, topk_ids = self.perplexity_fn(
                    ret_list[:i],
                    self.model,
                    layer_id=layer_id,
                    top_k=top_k_ppl
                )
                candidate_ids.extend([int(x) for x in topk_ids.detach().cpu().tolist()])
                candidate_ids = list(dict.fromkeys(candidate_ids))

            best_tok = candidate_ids[0]
            best_dist = None

            for tok in candidate_ids:
                seq = ret_list.copy()
                seq[i] = tok
                # ensure full length sequence
                if len(seq) < seq_len:
                    seq = seq + base_nearest[len(seq):]
                seq_tensor = torch.tensor(seq, device=device).unsqueeze(0)
                embeds = self._emb(seq_tensor).squeeze(0)
                act = self.forward_last_hidden(
                    self.model,
                    input_ids=None,
                    attention_mask=attention_mask,
                    layer_id=layer_id,
                    inputs_embeds=embeds.unsqueeze(0)
                )
                dist = F.mse_loss(
                    act[:, i, :].to(self._device),
                    target_act[:, i, :].to(self._device)
                )
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_tok = tok

            ret_list[i] = best_tok

        return ret_list

    # ---------- public API ----------

    def invert(
        self,
        hidden_state: torch.Tensor,
        total_input_ids: torch.Tensor,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        invert_method: str = "cosine",   # kept for compatibility
        filter_nonascii: bool = True,
        top_k_cos: int = 10,             # candidates for NN + calibration
        refine: bool = True,             # run optimization + discretization
        layer_id: Optional[int] = None,  # required for refine
        add_perplexity: bool = True,
        top_k_ppl: int = 10,
        logger: Optional[Callable[[str], None]] = None
    ) -> Tuple[float, str, List[int]]:
        """
        Returns:
          acc: float          # per-position exact match vs total_input_ids
          ret_tokens: str     # decoded tokens (skipping the first token, like original)
          ret_list: List[int] # predicted token ids per position
        """
        if layer_id is None:
            raise ValueError("invert requires layer_id for boundary activation.")

        target_act = self._as_seq(hidden_state).unsqueeze(0) if hidden_state.dim() == 2 else hidden_state
        seq_len = target_act.size(1)
        attn = attention_mask if attention_mask is not None else torch.ones(1, seq_len, device=self._device, dtype=torch.long)

        if refine:
            v_hat = self._optimize_embeddings(
                target_act=target_act.to(self._device),
                attention_mask=attn,
                layer_id=layer_id,
                seq_len=seq_len
            )
            ret_list = self._activation_calibrated_decode(
                v_hat=v_hat,
                target_act=target_act.to(self._device),
                attention_mask=attn,
                layer_id=layer_id,
                add_semantic=add_perplexity,
                top_k_cos=top_k_cos,
                top_k_ppl=top_k_ppl
            )
        else:
            # fallback: simple NN decode on target activation
            seq_vecs = self._as_seq(target_act.squeeze(0))
            ret_list = []
            for pos in range(seq_vecs.size(0)):
                cand = self._nn_topk(seq_vecs[pos], invert_method, top_k_cos)
                tid = self._pick_first_ascii(cand, filter_nonascii)
                ret_list.append(tid)

        prompt_len = len(total_input_ids[0])
        acc_cnt = sum(int(total_input_ids[0][j]) == ret_list[j] for j in range(prompt_len))
        acc = acc_cnt / len(ret_list)
        ret_tokens = self.tokenizer.decode(ret_list[1:])

        return acc, ret_tokens, ret_list
    
def pii_to_set(pii_obj) -> set:
    """
    Convert a PromptPII-style object into a set of (field, value) pairs
    for easy comparison.

    Handles:
      - dicts
      - values that might be lists
    """
    if pii_obj is None:
        return set()

    s = set()
    if isinstance(pii_obj, dict):
        items = pii_obj.items()
    else:
        # if you stored it as a string or something else, bail gracefully
        return set()

    for key, val in items:
        if val is None:
            continue
        if isinstance(val, list):
            for v in val:
                if v is not None:
                    s.add((str(key), str(v).strip().lower()))
        else:
            s.add((str(key), str(val).strip().lower()))

    return s


def compute_pii_metrics(gt_pii, guess_pii):
    """
    Compute precision/recall/F1 for PII reconstruction.

    gt_pii, guess_pii: PromptPII-style dicts (ground truth and guessed).
    """
    gt_set    = pii_to_set(gt_pii)
    guess_set = pii_to_set(guess_pii)

    if not gt_set and not guess_set:
        return 0.0, 0.0, 0.0  # nothing to measure

    tp = len(gt_set & guess_set)
    fp = len(guess_set - gt_set)
    fn = len(gt_set - guess_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return precision, recall, f1


def prompt_has_pii(row) -> bool:
    """Return True if the PromptPII dict contains any non-empty value."""
    pii = row.get("PromptPII", {})
    if not isinstance(pii, dict):
        return False
    for v in pii.values():
        if v not in (None, "", {}, [], "{}", "N/A"):
            return True
    return False


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def extract_pii_tokens(pii: dict) -> set:
    """Flatten PII values into a token set for token-level recall/precision."""
    if not isinstance(pii, dict):
        return set()
    tokens = set()
    for val in pii.values():
        if val in (None, "", {}, [], "{}", "N/A"):
            continue
        if isinstance(val, dict):
            vals = val.values()
        elif isinstance(val, list):
            vals = val
        else:
            vals = [val]
        for v in vals:
            if v in (None, "", "N/A"):
                continue
            tokens.update(tokenize_words(str(v)))
    return tokens

if __name__=='__main__':
    PANOPTICON_PATH = "../PANOPTICON_v1.4"
    MAX_SAMPLES = 1000  
    pan_ds = load_from_disk(PANOPTICON_PATH)
    pan_ds = pan_ds.filter(lambda ex: prompt_has_pii(ex))
    if len(pan_ds) > MAX_SAMPLES:
        pan_ds = pan_ds.shuffle(seed=42).select(range(MAX_SAMPLES))

    pano_p = PanoPlus()
    panop_id = pano_p.get_ids()  # full PII per profile ID

    lm = LargeLangModel(model_name="meta-llama/Llama-3.1-8B-Instruct")
    inf = Inference(model=lm.model, tokenizer=lm.tokenizer)
    inv = Inversion(model=lm.model, tokenizer=lm.tokenizer)

    layer_id = lm.n_model_layers - 1

    # Running totals
    total_prec = 0.0
    total_rec  = 0.0
    total_f1   = 0.0
    total_acc  = 0.0   # token-level accuracy from inv.invert (if that's what `acc` is)
    total_prompt_token_recall = 0.0
    total_prompt_token_precision = 0.0
    total_pii_token_recall = 0.0
    total_pii_token_precision = 0.0

    count = 0          # total samples we included in the averages
    pii_count = 0      # how many of those actually had GT PII (optional, but useful)
   
    for i, row in enumerate(pan_ds):
        if count >= MAX_SAMPLES:
            break

        gt_prompt     = row['Prompt']
        gt_prompt_pii = row['PromptPII']   # ground truth PII that was actually in the prompt

        full_pii = pano_p.dataset.filter(lambda x:x['Unique ID'] == row['ID'])[0]

        # capture hidden states
        ids, mask, embeds, hiddens = inf.capture(
            layer_id=layer_id,
            prompt=gt_prompt
        )

        # pick last valid [1, seq, dim] hidden
        hidden_state = None
        for t in reversed(hiddens):
            if isinstance(t, torch.Tensor) and t.dim() >= 3 and t.size(0) == 1:
                hidden_state = t
                break
        if hidden_state is None:
            continue

        # inversion
        acc, ret_tokens, ret_list = inv.invert(
            hidden_state=hidden_state,
            total_input_ids=ids,
            attention_mask=mask,
            invert_method="cosine",
            filter_nonascii=True,
            top_k_cos=20,
            refine=True,
            layer_id=layer_id,
            add_perplexity=True,
            top_k_ppl=10
        )

        guess_prompt = ret_tokens

        # PII matched in the reconstructed prompt
        # panop_id[row['ID']] should be the full PII record for this profile
        guess_prompt_pii = match_pii(guess_prompt, full_pii)

        # --- compute PII metrics for this example ---
        prec, rec, f1 = compute_pii_metrics(gt_prompt_pii, guess_prompt_pii)
        gt_set = pii_to_set(gt_prompt_pii)

        # Only count toward PII metrics if there actually *was* PII in the GT prompt
        if gt_set:
            total_prec += prec
            total_rec  += rec
            total_f1   += f1
            pii_count  += 1

            gt_pii_tokens = extract_pii_tokens(gt_prompt_pii)
            guess_tokens = set(tokenize_words(guess_prompt))
            guess_pii_tokens = extract_pii_tokens(guess_prompt_pii)
            if gt_pii_tokens:
                overlap = len(gt_pii_tokens & guess_tokens)
                total_pii_token_recall += overlap / len(gt_pii_tokens)
            if guess_pii_tokens:
                total_pii_token_precision += len(gt_pii_tokens & guess_pii_tokens) / len(guess_pii_tokens)

        # token-level accuracy from inversion (if you care about overall textual accuracy)
        total_acc += acc

        # prompt-level token precision/recall for the whole sentence
        gt_prompt_tokens = set(tokenize_words(gt_prompt))
        guess_prompt_tokens = set(tokenize_words(guess_prompt))
        if gt_prompt_tokens:
            total_prompt_token_recall += len(gt_prompt_tokens & guess_prompt_tokens) / len(gt_prompt_tokens)
        if guess_prompt_tokens:
            total_prompt_token_precision += len(gt_prompt_tokens & guess_prompt_tokens) / len(guess_prompt_tokens)

        count += 1

    # compute averages
    # avg over samples where GT PII existed
    avg_precision = total_prec / pii_count if pii_count > 0 else 0.0
    avg_recall    = total_rec  / pii_count if pii_count > 0 else 0.0
    avg_f1        = total_f1   / pii_count if pii_count > 0 else 0.0

    # token-level accuracy: averaged over all processed samples
    avg_acc       = total_acc  / count     if count > 0 else 0.0
    avg_prompt_token_recall = total_prompt_token_recall / count if count > 0 else 0.0
    avg_prompt_token_precision = total_prompt_token_precision / count if count > 0 else 0.0
    avg_pii_token_recall = total_pii_token_recall / pii_count if pii_count > 0 else 0.0
    avg_pii_token_precision = total_pii_token_precision / pii_count if pii_count > 0 else 0.0

    summary = {
        "samples_processed": count,
        "samples_with_pii": pii_count,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "avg_token_accuracy": avg_acc,
        "avg_prompt_token_recall": avg_prompt_token_recall,
        "avg_prompt_token_precision": avg_prompt_token_precision,
        "avg_pii_token_recall": avg_pii_token_recall,
        "avg_pii_token_precision": avg_pii_token_precision
    }

    print("\n=== SUMMARY OVER", count, "EXAMPLES ===")
    print(summary)
