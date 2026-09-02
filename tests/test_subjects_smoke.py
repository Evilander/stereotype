"""CPU smoke tests on tiny subjects. They need the models in the local Hugging Face cache."""

import numpy as np
import pytest
import torch

SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="module")
def smol():
    from stereotype_audit.subjects.causal_lm import CausalLMSubject

    try:
        return CausalLMSubject(SMOL, device="cpu", batch_size=4)
    except Exception as err:  # noqa: BLE001
        pytest.skip(f"fixture model unavailable: {err}")


def test_continuation_logprob_matches_manual_forward(smol):
    prompt = "The capital of France is"
    cont = " Paris."
    got = smol.continuation_logprobs([prompt], [cont])[0]
    ids_p = smol.tokenizer(prompt)["input_ids"]
    ids_pc = smol.tokenizer(prompt + cont)["input_ids"]
    k = len(ids_p)
    with torch.inference_mode():
        logits = smol.model(input_ids=torch.tensor([ids_pc])).logits[0].float()
    lp = torch.log_softmax(logits, dim=-1)
    manual = sum(float(lp[t - 1, ids_pc[t]]) for t in range(k, len(ids_pc)))
    assert got == pytest.approx(manual, abs=1e-3)
    assert got < 0


def test_batching_and_order_do_not_change_scores(smol):
    prompts = ["A short one", "A somewhat longer prompt that has more tokens in it", "Mid length prompt here"]
    conts = [" x", " and then some more", " ok"]
    single = np.array([smol.continuation_logprobs([p], [c])[0] for p, c in zip(prompts, conts, strict=True)])
    batched = smol.continuation_logprobs(prompts, conts)
    assert np.allclose(single, batched, atol=1e-2)


def test_yes_probability_is_a_probability(smol):
    prompt = smol.render_chat("Is water wet? Answer Yes or No.")
    p = smol.yes_probability([prompt])
    assert 0.0 <= p[0] <= 1.0
    p2, mass = smol.yes_probability_with_mass([prompt])
    assert p2[0] == p[0] and 0.0 < mass[0] <= 1.0
    assert smol.token_count(" Lakisha") >= 1


def test_sequence_logprobs_prefers_grammatical_sentence(smol):
    good = "The cat sat on the mat."
    bad = "Mat the on sat cat the."
    lp = smol.sequence_logprobs([good, bad])
    assert lp[0] > lp[1]


def test_open_think_detection_and_control(smol):
    from stereotype_audit.subjects.causal_lm import has_open_think

    assert has_open_think("<|im_start|>assistant\n<think>\n")
    assert not has_open_think("<|im_start|>assistant\n<think></think>")
    assert not has_open_think("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert not has_open_think("<|im_start|>assistant\n")
    assert smol.thinking_control in {"not-applicable", "template-switch", "forced-close"}
    assert not has_open_think(smol.render_chat("hi", thinking=False))


def test_generate_returns_token_counts(smol):
    out = smol.generate([smol.render_chat("Say hello.")], max_new_tokens=8)
    assert len(out) == 1 and 1 <= out[0].n_new_tokens <= 8
    assert out[0].n_think_tokens == 0


def test_generate_detects_end_of_sequence(smol):
    out = smol.generate([smol.render_chat("Reply with the single word OK.")], max_new_tokens=64)
    assert out[0].finished, "a short answer must end with an end-of-sequence token before the limit"
    assert out[0].n_new_tokens < 64
    assert "<|im_end|>" not in out[0].text or out[0].text.count("<|im_end|>") == 0


def test_sequence_logprobs_does_not_double_start_token(smol):
    text = "Hello there."
    lp = smol.sequence_logprobs([text])[0]
    ids = [smol.tokenizer.bos_token_id or smol.tokenizer.eos_token_id] + smol.tokenizer(
        text, add_special_tokens=False
    )["input_ids"]
    with torch.inference_mode():
        logits = smol.model(input_ids=torch.tensor([ids])).logits[0].float()
    lps = torch.log_softmax(logits, dim=-1)
    manual = sum(float(lps[t - 1, ids[t]]) for t in range(1, len(ids)))
    assert lp == pytest.approx(manual, abs=1e-3)
