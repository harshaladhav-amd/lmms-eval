import copy
import re
from typing import List, Optional, Tuple, Union

from tqdm import tqdm
from vllm import LLM, SamplingParams

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

# Early-EOT retry when the first pass emitted no timestamps.
# English LibriSpeech ~2–3 w/s. Not in HF generate_with_fallback.
# Loops: WhisperLLM.generate (HF WhisperGenerationMixin analog), not LLM.generate.
MIN_RETRY_DURATION_S = 2.0
EXPECTED_WPS = 2.0
SHORT_FRACTION = 0.5
KEEP_RECOVERED_WPS = 0.5
TOKENS_PER_SEC = 1.5

# Timestamp + unread-tail re-decode. LibriSpeech only.
# Client stand-in for HF WhisperGenerationMixin seek / vLLM STT #53145.
# LLM.generate has no mel-frame seek cursor.
TS_RE = re.compile(r"<\|(\d+\.\d+)\|>")
MIN_GAP_S = 2.0
PREROLL_S = 0.5
MIN_DENSITY_TAIL_S = 4.0
MAX_TAIL_PASSES = 3
LIBRI_TS_PROMPT = "<|startoftranscript|><|en|><|transcribe|>"


def _duration_s(audio) -> float:
    arr, sr = audio["array"], float(audio["sampling_rate"])
    return len(arr) / max(sr, 1.0)


def _too_short(text: str, duration_s: float) -> bool:
    n = len(text.split())
    if duration_s < MIN_RETRY_DURATION_S:
        return n == 0
    if n == 0:
        return True
    expected = duration_s * EXPECTED_WPS
    return n < SHORT_FRACTION * expected


def _keep_retry(old: str, new: str, duration_s: float) -> bool:
    nw, ow = new.split(), old.split()
    if len(nw) <= len(ow):
        return False
    if duration_s > 0 and (len(nw) / duration_s) < KEEP_RECOVERED_WPS:
        return False
    return True


def _strip_ts(text: str) -> str:
    return " ".join(TS_RE.sub(" ", text or "").split())


def _last_timestamp_s(text: str, token_ids=None, tokenizer=None) -> float | None:
    if token_ids is not None and tokenizer is not None:
        last = None
        for tid in token_ids:
            tok = tokenizer.convert_ids_to_tokens(int(tid))
            m = TS_RE.fullmatch(tok) if tok else None
            if m:
                last = float(m.group(1))
        if last is not None:
            return last
    hits = [float(x) for x in TS_RE.findall(text or "")]
    return hits[-1] if hits else None


def _slice_audio(audio, start_s: float):
    arr, sr = audio["array"], float(audio["sampling_rate"])
    i = max(int(start_s * sr), 0)
    return {"array": arr[i:], "sampling_rate": audio["sampling_rate"]}


def _seam_dedup(head: str, tail: str, max_overlap: int = 12) -> str:
    hw, tw = head.split(), tail.split()
    for n in range(min(max_overlap, len(hw), len(tw)), 0, -1):
        if hw[-n:] == tw[:n]:
            tw = tw[n:]
            break
    return " ".join(hw + tw)


def _keep_tail(tail_text: str, tail_dur: float) -> bool:
    n = len(tail_text.split())
    if n == 0:
        return False
    if tail_dur >= MIN_DENSITY_TAIL_S and (n / tail_dur) < KEEP_RECOVERED_WPS:
        return False
    return True


def _audio_prompt(audio, prompt_text: str) -> dict:
    return {
        "prompt": prompt_text,
        "multi_modal_data": {"audio": (audio["array"], audio["sampling_rate"])},
    }


def _llm_class():
    """HF WhisperForConditionalGeneration analog; stock LLM if mixin is absent."""
    try:
        from vllm.whisper_generation import WhisperLLM

        return WhisperLLM
    except ImportError:
        return LLM


def _generate_with_fallback(llm, prompts, sampling_params, use_tqdm=False):
    """Call Mixin.generate with HF long-form kwargs when WhisperLLM accepts them."""
    prompts_list = prompts if isinstance(prompts, (list, tuple)) else [prompts]
    gen_kwargs = dict(use_tqdm=use_tqdm)
    try:
        from vllm.whisper_generation import (
            COMPRESSION_RATIO_THRESHOLD,
            LOGPROB_THRESHOLD,
            TEMPERATURES,
        )

        gen_kwargs.update(
            temperature=TEMPERATURES,
            compression_ratio_threshold=COMPRESSION_RATIO_THRESHOLD,
            logprob_threshold=LOGPROB_THRESHOLD,
        )
    except ImportError:
        pass
    try:
        return llm.generate(prompts_list, sampling_params, **gen_kwargs)
    except TypeError:
        gen_kwargs = dict(use_tqdm=use_tqdm)
        return llm.generate(prompts_list, sampling_params, **gen_kwargs)


@register_model("whisper_vllm")
class WhisperVllm(lmms):
    """
    Whisper Audio Model VLLM
    """

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        batch_size: Optional[Union[int, str]] = 1,
        **model_kwargs,
    ) -> None:
        super().__init__()
        self._batch_size = int(batch_size)

        self._model = _llm_class()(
            model=pretrained,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **model_kwargs,
        )

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        raise NotImplementedError()

    @property
    def tokenizer(self):
        return self._model.get_tokenizer()

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Whisper")

    def flatten(self, input):  # noqa: A002 - Preserve the existing keyword argument name.
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        batched_requests = [requests[i : i + self.batch_size] for i in range(0, len(requests), self.batch_size)]
        for batch_requests in batched_requests:
            batched_prompts = []
            durations = []
            audios = []
            libri_ts = []
            sampling_params = None
            for idx in range(len(batch_requests)):
                contexts, gen_kwargs, doc_to_visual, doc_id, task, split = batch_requests[idx].arguments

                # generation parameters
                sampling_params = SamplingParams(
                    temperature=gen_kwargs.get("temperature", 0),
                    top_p=gen_kwargs.get("top_p", 0),
                    max_tokens=gen_kwargs.get("max_new_tokens", 256),
                )

                # prepare multimodal inputs
                audio = doc_to_visual(self.task_dict[task][split][doc_id])
                assert len(audio) == 1
                audio = audio[0]

                pre_prompt = gen_kwargs.get("pre_prompt", "")
                post_prompt = gen_kwargs.get("post_prompt", "")

                # Fleurs stays on notimestamps (HF short-form default).
                # LibriSpeech uses timestamps so unread-tail can recover prefix cuts.
                task_name = str(task).strip()
                use_libri_ts = not task_name.startswith("fleurs")
                if use_libri_ts:
                    prompt_text = LIBRI_TS_PROMPT
                else:
                    language = self.task_dict[task][split][doc_id]["language"]
                    language_to_token = {"Mandarin Chinese": "zh", "Cantonese Chinese": "zh", "English": "en"}

                    if language in language_to_token:
                        token = language_to_token[language]
                        prompt_text = f"{pre_prompt}<|startoftranscript|><|{token}|><|transcribe|><|notimestamps|>{post_prompt}"
                    else:
                        prompt_text = f"{pre_prompt}Please recognize the speech and only output the recognized content:{post_prompt}"

                batched_prompts.append(_audio_prompt(audio, prompt_text))
                durations.append(_duration_s(audio))
                audios.append(audio)
                libri_ts.append(use_libri_ts)

            outputs = _generate_with_fallback(
                self.model, batched_prompts, sampling_params, use_tqdm=False
            )
            transcriptions = [output.outputs[0].text for output in outputs]
            tok = self.model.get_tokenizer()
            covered = []
            for i, text in enumerate(transcriptions):
                ids = list(outputs[i].outputs[0].token_ids)
                covered.append(_last_timestamp_s(text, ids, tok))

            for _ in range(MAX_TAIL_PASSES):
                retry_idx, retry_prompts, retry_starts = [], [], []
                for i, (dur, cov) in enumerate(zip(durations, covered)):
                    if not libri_ts[i] or cov is None or (dur - cov) < MIN_GAP_S:
                        continue
                    start = max(cov - PREROLL_S, 0.0)
                    tail_audio = _slice_audio(audios[i], start)
                    if len(tail_audio["array"]) == 0:
                        covered[i] = dur
                        continue
                    retry_idx.append(i)
                    retry_starts.append(start)
                    retry_prompts.append(_audio_prompt(tail_audio, LIBRI_TS_PROMPT))
                if not retry_idx:
                    break
                tail_out = _generate_with_fallback(
                    self.model, retry_prompts, sampling_params, use_tqdm=False
                )
                for j, i in enumerate(retry_idx):
                    tail_raw = tail_out[j].outputs[0].text
                    tail_txt = _strip_ts(tail_raw)
                    tail_dur = durations[i] - retry_starts[j]
                    if not _keep_tail(tail_txt, tail_dur):
                        covered[i] = durations[i]
                        continue
                    transcriptions[i] = _seam_dedup(_strip_ts(transcriptions[i]), tail_txt)
                    t2 = _last_timestamp_s(
                        tail_raw, list(tail_out[j].outputs[0].token_ids), tok
                    )
                    if t2 is None:
                        covered[i] = durations[i]
                    else:
                        covered[i] = retry_starts[j] + t2

            retry_idx, retry_prompts, retry_sp = [], [], []
            for i, (text, dur, cov) in enumerate(zip(transcriptions, durations, covered)):
                # English w/s floor is LibriSpeech-only. Fleurs stays one-shot notimestamps.
                if not libri_ts[i] or cov is not None:
                    continue
                if not _too_short(_strip_ts(text), dur):
                    continue
                sp = sampling_params.clone() if hasattr(sampling_params, "clone") else copy.copy(sampling_params)
                floor = int(dur * TOKENS_PER_SEC)
                sp.min_tokens = min(max(floor, 1), sp.max_tokens or 256)
                retry_idx.append(i)
                retry_prompts.append(batched_prompts[i])
                retry_sp.append(sp)

            if retry_idx:
                groups: dict = {}
                for j, sp in enumerate(retry_sp):
                    groups.setdefault(sp.min_tokens, []).append(j)
                retry_out = [None] * len(retry_idx)
                for js in groups.values():
                    outs = _generate_with_fallback(
                        self.model,
                        [retry_prompts[j] for j in js],
                        retry_sp[js[0]],
                        use_tqdm=False,
                    )
                    for k, j in enumerate(js):
                        retry_out[j] = outs[k]
                for j, i in enumerate(retry_idx):
                    new = retry_out[j].outputs[0].text
                    if _keep_retry(_strip_ts(transcriptions[i]), _strip_ts(new), durations[i]):
                        transcriptions[i] = new

            answers = [tok.normalize(_strip_ts(t)) for t in transcriptions]

            assert len(answers) == len(batch_requests)
            res.extend(answers)
            pbar.update(len(batch_requests))

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
