"""Causal language model subject: log-probability scoring and controlled generation."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch

from stereotype_audit.stats import log_sum_exp
from stereotype_audit.subjects.base import SubjectInfo, gpu_snapshot, library_versions, pick_device

YES_VARIANTS = ["Yes", " Yes", "yes", " yes", "YES", " YES"]
NO_VARIANTS = ["No", " No", "no", " no", "NO", " NO"]
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class Generation:
    text: str
    n_new_tokens: int
    n_think_tokens: int
    finished: bool


def ids_enum(ids):
    return list(enumerate(ids))


def _find_subsequence(seq: list[int], sub: list[int]) -> int | None:
    if not sub or len(sub) > len(seq):
        return None
    for k in range(len(seq) - len(sub) + 1):
        if seq[k : k + len(sub)] == sub:
            return k
    return None


def has_open_think(rendered: str) -> bool:
    """True when the generation prompt ends inside a <think> block that was never closed."""
    tail = rendered[-400:]
    last_open = tail.rfind("<think>")
    if last_open == -1:
        return False
    return "</think>" not in tail[last_open:]


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class CausalLMSubject:
    family = "lm"

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        dtype: str = "auto",
        quant: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 2048,
        trust_remote_code: bool = False,
    ):
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self.device = pick_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.quant = quant
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        torch_dtype = self._resolve_dtype(dtype)
        self.dtype_name = str(torch_dtype).replace("torch.", "")
        self.model = self._load_model(model_id, revision, torch_dtype, quant, trust_remote_code)
        self.model.eval()
        self.loader_class = type(self.model).__name__
        self.supports_thinking = self._template_mentions("enable_thinking")
        self._single_token_cache: dict[str, int | None] = {}
        self.forced_think_close = False
        self.thinking_control = self._probe_thinking_control()

    def _resolve_dtype(self, dtype: str):
        if dtype == "auto":
            return torch.bfloat16 if self.device == "cuda" else torch.float32
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]

    def _load_model(self, model_id, revision, torch_dtype, quant, trust_remote_code):
        import transformers

        kwargs = {"revision": revision, "trust_remote_code": trust_remote_code}
        if quant == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
            )
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["dtype"] = torch_dtype
        classes = ["AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM"]
        last_err: Exception | None = None
        for name in classes:
            cls = getattr(transformers, name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(model_id, **kwargs)
            except (ValueError, KeyError, OSError) as err:
                last_err = err
                continue
            if quant != "4bit":
                model.to(self.device)
            return model
        raise RuntimeError(f"could not load {model_id} with any causal/multimodal class: {last_err}")

    def _template_mentions(self, needle: str) -> bool:
        tpl = getattr(self.tokenizer, "chat_template", None)
        return bool(tpl) and needle in tpl

    def _probe_thinking_control(self) -> str:
        """How reasoning is switched off for scoring: 'template-switch', 'not-applicable', or 'forced-close'."""
        if not getattr(self.tokenizer, "chat_template", None):
            return "not-applicable"
        rendered = self._render_messages([{"role": "user", "content": "probe"}], thinking=False)
        if has_open_think(rendered):
            return "forced-close"
        return "template-switch" if self.supports_thinking else "not-applicable"

    def info(self) -> SubjectInfo:
        return SubjectInfo(
            model_id=self.model_id,
            family=self.family,
            revision=self.revision,
            dtype=self.dtype_name,
            quantization=self.quant,
            device=self.device,
            library_versions=library_versions(),
            extra={
                "loader_class": self.loader_class,
                "supports_thinking": self.supports_thinking,
                "thinking_control": self.thinking_control,
                "forced_think_close": self.forced_think_close,
                "has_chat_template": bool(getattr(self.tokenizer, "chat_template", None)),
                **gpu_snapshot(),
            },
        )

    # ----- prompt rendering -------------------------------------------------

    def _render_messages(self, messages: list[dict], thinking: bool) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self.supports_thinking:
            kwargs["enable_thinking"] = thinking
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def render_chat(self, user: str, system: str | None = None, thinking: bool = False) -> str:
        """Render one user turn in the model's chat template.

        With thinking=False the rendered prompt must not leave a reasoning block open,
        otherwise the scored token would be the first token of the model's reasoning
        rather than its answer. Templates without a switch that still open a block get
        the block closed explicitly, and the subject records that it had to.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        if not getattr(self.tokenizer, "chat_template", None):
            prefix = f"{system}\n\n" if system else ""
            return f"{prefix}User: {user}\nAssistant:"
        rendered = self._render_messages(messages, thinking)
        if not thinking and has_open_think(rendered):
            rendered = rendered + "</think>\n"
            self.forced_think_close = True
        return rendered

    # ----- tokenisation helpers --------------------------------------------

    def _encode(self, text: str) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=True)["input_ids"]
        if len(ids) > self.max_length:
            # silent truncation could cut the two channels of a pair differently; fail loudly instead
            raise ValueError(f"prompt of {len(ids)} tokens exceeds max_length={self.max_length}")
        return ids

    def _eos_ids(self) -> set[int]:
        ids = set()
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            ids.add(int(eos))
        gen_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if isinstance(gen_eos, int):
            ids.add(gen_eos)
        elif isinstance(gen_eos, (list, tuple)):
            ids.update(int(x) for x in gen_eos)
        return ids

    def single_token_id(self, text: str) -> int | None:
        """Token id if `text` is exactly one token without special tokens, else None."""
        if text not in self._single_token_cache:
            ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
            self._single_token_cache[text] = ids[0] if len(ids) == 1 else None
        return self._single_token_cache[text]

    def _pad_batch_left(self, id_lists: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Left-pad so every sequence ends at the same column; positions restart at 0 on the first real token."""
        pad = self.tokenizer.pad_token_id
        max_len = max(len(ids) for ids in id_lists)
        input_ids = torch.full((len(id_lists), max_len), pad, dtype=torch.long)
        attn = torch.zeros((len(id_lists), max_len), dtype=torch.long)
        for i, ids in enumerate(id_lists):
            input_ids[i, max_len - len(ids) :] = torch.tensor(ids)
            attn[i, max_len - len(ids) :] = 1
        position_ids = (attn.cumsum(dim=-1) - 1).clamp(min=0)
        dev = self.model.device
        return input_ids.to(dev), attn.to(dev), position_ids.to(dev)

    def _tail_logits(self, input_ids, attn, position_ids, keep: int) -> torch.Tensor:
        """Logits for the last `keep` columns only; falls back to a full forward on models without logits_to_keep."""
        try:
            out = self.model(
                input_ids=input_ids, attention_mask=attn, position_ids=position_ids, logits_to_keep=keep
            )
            logits = out.logits
            if logits.shape[1] != keep:
                logits = logits[:, -keep:]
            return logits
        except TypeError:
            out = self.model(input_ids=input_ids, attention_mask=attn, position_ids=position_ids)
            return out.logits[:, -keep:]

    def _batches(self, n: int, lengths: list[int]):
        order = sorted(range(n), key=lambda i: lengths[i])
        for b in range(0, n, self.batch_size):
            yield order[b : b + self.batch_size]

    # ----- scoring ----------------------------------------------------------

    @torch.inference_mode()
    def _score_batch(self, id_lists: list[list[int]], start_positions: list[int]) -> list[float]:
        """Sum of log P(token_t | tokens_<t) for t >= start for each sequence.

        Sequences are left-padded so the scored spans all sit at the end, and only
        the logits for those trailing columns are materialised.
        """
        input_ids, attn, position_ids = self._pad_batch_left(id_lists)
        max_len = input_ids.shape[1]
        spans = [max(len(ids) - start, 0) for ids, start in zip(id_lists, start_positions, strict=True)]
        keep = min(max_len, max(spans) + 1)
        logits = self._tail_logits(input_ids, attn, position_ids, keep)
        out = []
        for i, ids in enumerate(id_lists):
            span = spans[i]
            if span == 0:
                # an empty continuation has probability one
                out.append(0.0)
                continue
            # kept columns cover absolute positions [max_len-keep, max_len); the continuation targets
            # are the last `span` tokens, predicted from the columns just before each of them
            rows = torch.log_softmax(logits[i, keep - span - 1 : keep - 1].float(), dim=-1)
            targets = torch.tensor(ids[-span:], device=rows.device)
            out.append(float(rows.gather(1, targets[:, None]).sum().item()))
        return out

    def continuation_logprobs(self, prompts: list[str], continuations: list[str]) -> np.ndarray:
        """log P(continuation | prompt) for each pair, tokenised jointly so boundary merges are consistent."""
        if len(prompts) != len(continuations):
            raise ValueError("prompts and continuations must align")
        id_lists, starts = [], []
        for p, c in zip(prompts, continuations, strict=True):
            ids_p = self._encode(p)
            ids_pc = self._encode(p + c)
            k = min(_common_prefix_len(ids_p, ids_pc), len(ids_p))
            id_lists.append(ids_pc)
            starts.append(max(k, 1))
        scored: dict[int, float] = {}
        for idx in self._batches(len(id_lists), [len(x) for x in id_lists]):
            vals = self._score_batch([id_lists[i] for i in idx], [starts[i] for i in idx])
            for i, v in zip(idx, vals, strict=True):
                scored[i] = v
        return np.asarray([scored[i] for i in range(len(id_lists))], dtype=np.float64)

    @torch.inference_mode()
    def next_token_logprobs(self, prompts: list[str], token_ids: list[int]) -> np.ndarray:
        """Matrix (n_prompts, len(token_ids)) of log P(token | prompt) from one forward pass per prompt."""
        id_lists = [self._encode(p) for p in prompts]
        ids_t = torch.tensor(token_ids)
        out = np.empty((len(prompts), len(token_ids)), dtype=np.float64)
        for idx in self._batches(len(id_lists), [len(x) for x in id_lists]):
            chunk = [id_lists[i] for i in idx]
            input_ids, attn, position_ids = self._pad_batch_left(chunk)
            logits = self._tail_logits(input_ids, attn, position_ids, 1)
            last = torch.log_softmax(logits[:, -1].float(), dim=-1)
            picked = last[:, ids_t.to(last.device)].cpu().numpy()
            for row, i in enumerate(idx):
                out[i] = picked[row]
        return out

    def option_logprobs(self, prompts: list[str], options: list[str]) -> np.ndarray:
        """Matrix (n_prompts, n_options) of log P(option | prompt).

        Uses the one-pass next-token path when every option is a single token,
        otherwise scores each option as a continuation.
        """
        ids = [self.single_token_id(o) for o in options]
        out = np.empty((len(prompts), len(options)), dtype=np.float64)
        single = [j for j, t in ids_enum(ids) if t is not None]
        multi = [j for j, t in ids_enum(ids) if t is None]
        if single:
            out[:, single] = self.next_token_logprobs(prompts, [int(ids[j]) for j in single])
        if multi:
            flat_p = [p for p in prompts for _ in multi]
            flat_c = [options[j] for _ in prompts for j in multi]
            out[:, multi] = self.continuation_logprobs(flat_p, flat_c).reshape(len(prompts), len(multi))
        return out

    def sequence_logprobs(self, texts: list[str]) -> np.ndarray:
        """log P(text | start token): every token of the text is scored after one start token id.

        The start token is prepended as an id, not as text, so tokenizers that add
        their own special tokens cannot double it.
        """
        start = self.tokenizer.bos_token_id
        if start is None:
            start = self.tokenizer.eos_token_id
        id_lists, starts = [], []
        for t in texts:
            body = self.tokenizer(t, add_special_tokens=False)["input_ids"]
            if len(body) + 1 > self.max_length:
                raise ValueError(f"text of {len(body)} tokens exceeds max_length={self.max_length}")
            id_lists.append([int(start)] + body)
            starts.append(1)
        scored: dict[int, float] = {}
        for idx in self._batches(len(id_lists), [len(x) for x in id_lists]):
            vals = self._score_batch([id_lists[i] for i in idx], [starts[i] for i in idx])
            for i, v in zip(idx, vals, strict=True):
                scored[i] = v
        return np.asarray([scored[i] for i in range(len(id_lists))], dtype=np.float64)

    def token_counts(self, texts: list[str]) -> list[int]:
        return [len(self.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]

    def yes_probability(self, prompts: list[str]) -> np.ndarray:
        """P(yes) normalised over yes/no surface variants, as in Discrim-Eval."""
        return self.yes_probability_with_mass(prompts)[0]

    def yes_probability_with_mass(self, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(P(yes | yes-or-no), P(yes-or-no)) — the second value is the next-token mass the two answers
        captured before renormalisation; the rest went to refusals, hedges, or other continuations."""
        mat = self.option_logprobs(prompts, YES_VARIANTS + NO_VARIANTS)
        n_yes = len(YES_VARIANTS)
        p_yes = np.empty(len(prompts), dtype=np.float64)
        mass = np.empty(len(prompts), dtype=np.float64)
        for i in range(len(prompts)):
            ly = log_sum_exp(mat[i, :n_yes])
            ln = log_sum_exp(mat[i, n_yes:])
            total = log_sum_exp([ly, ln])
            p_yes[i] = float(np.exp(ly - total))
            mass[i] = float(np.exp(total))
        return p_yes, mass

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def choice_logprobs(self, prompts: list[str], choices: list[list[str]]) -> np.ndarray:
        """For each prompt, log-sum-exp over the surface variants of each choice. Returns (n, n_choices)."""
        flat = [v for group in choices for v in group]
        mat = self.option_logprobs(prompts, flat)
        out = np.empty((len(prompts), len(choices)), dtype=np.float64)
        pos = 0
        for j, group in enumerate(choices):
            out[:, j] = [log_sum_exp(mat[i, pos : pos + len(group)]) for i in range(len(prompts))]
            pos += len(group)
        return out

    # ----- generation -------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self, prompts: list[str], max_new_tokens: int = 256, thinking: bool = False
    ) -> list[Generation]:
        """Greedy generation from already-rendered prompts. Thinking-token counts use the <think> tags."""
        self.tokenizer.padding_side = "left"
        outputs: list[Generation] = []
        try:
            for b in range(0, len(prompts), self.batch_size):
                chunk = prompts[b : b + self.batch_size]
                enc = self.tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=True).to(
                    self.model.device
                )
                gen = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                new_tokens = gen[:, enc["input_ids"].shape[1] :]
                eos_ids = self._eos_ids()
                close_ids = self.tokenizer("</think>", add_special_tokens=False)["input_ids"]
                for row in new_tokens:
                    raw = [int(t) for t in row.tolist()]
                    # cut at the first end-of-sequence token; everything after it is padding
                    stop = next((k for k, t in enumerate(raw) if t in eos_ids), None)
                    finished = stop is not None
                    ids = raw[:stop] if finished else raw
                    text = self.tokenizer.decode(ids, skip_special_tokens=False)
                    n_think = 0
                    if thinking:
                        # the opening tag usually sits in the prompt; count generated ids up to the closing tag,
                        # or every generated id when the model never closed its reasoning (truncated run)
                        pos = _find_subsequence(ids, close_ids)
                        n_think = pos if pos is not None else len(ids)
                    outputs.append(
                        Generation(
                            text=text, n_new_tokens=len(ids), n_think_tokens=n_think, finished=finished
                        )
                    )
        finally:
            self.tokenizer.padding_side = "right"
        return outputs
