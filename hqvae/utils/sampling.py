# ------------------------------------------------------------------------------------
# Modified from Minimal DALL-E
# https://github.com/kakaobrain/minDALL-E/blob/main/dalle/utils/sampling.py
# ------------------------------------------------------------------------------------

import torch
from typing import Optional, List
from tqdm import tqdm
from torch.nn import functional as F


def cutoff_topk_logits(logits: torch.FloatTensor, k: int) -> torch.FloatTensor:
    if k is None:
        return logits
    else:
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[:, [-1]]] = -float('Inf')
        return out


def cutoff_topp_probs(probs: torch.FloatTensor, p: float) -> torch.FloatTensor:
    if p is None:
        return probs
    else:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_idx_remove_cond = cum_probs >= p

        sorted_idx_remove_cond[..., 1:] = sorted_idx_remove_cond[..., :-1].clone()
        sorted_idx_remove_cond[..., 0] = 0

        indices_to_remove = sorted_idx_remove_cond.scatter(-1, sorted_indices, sorted_idx_remove_cond)
        probs = probs.masked_fill(indices_to_remove, 0.0)
        norm_probs = probs / torch.sum(probs, dim=-1, keepdim=True)
        return norm_probs


def get_positional_encoding(inputs: torch.LongTensor, mode: str = '1d') -> torch.LongTensor:
    device = inputs.device
    if mode == '1d':
        B, N = inputs.shape
        xs_pos = torch.arange(N, device=device).repeat((B, 1))
    elif mode == '2d':
        B, H, W = inputs.shape
        xs_pos_h = torch.arange(H, device=device).repeat(B, W, 1).transpose(1, 2)
        xs_pos_w = torch.arange(W, device=device).repeat(B, H, 1)
        xs_pos = (xs_pos_h, xs_pos_w)
    else:
        raise ValueError('%s positional encoding invalid' % mode)
    return xs_pos


@torch.no_grad()
def sampling(model: torch.nn.Module,
             tokens: torch.LongTensor,
             top_k: Optional[float] = None,
             top_p: Optional[float] = None,
             softmax_temperature: float = 1.0,
             is_tqdm: bool = True,
             use_fp16: bool = True,
             max_seq_len: int = 256) -> torch.LongTensor:
    code = None
    past = None

    pbar = tqdm(range(max_seq_len), total=max_seq_len) if is_tqdm else range(max_seq_len)
    pos_enc_tokens = get_positional_encoding(tokens, mode='1d')

    for cnt, h in enumerate(pbar):
        if code is None:
            code_ = None
            pos_enc_code_ = None
        else:
            code_ = code.clone().detach()
            pos_enc_code_ = get_positional_encoding(code_, mode='1d')
            code_ = code_[:, cnt-1].unsqueeze(-1)
            pos_enc_code_ = pos_enc_code_[:, cnt-1].unsqueeze(-1)

        logits, present = model.sampling(images=code_,
                                         texts=tokens,
                                         pos_images=pos_enc_code_,
                                         pos_texts=pos_enc_tokens,
                                         use_fp16=use_fp16,
                                         past=past)
        logits = logits.to(dtype=torch.float32)
        logits = logits / softmax_temperature

        if isinstance(present, tuple):
            present1 = torch.stack(present[0]).clone().detach()
            present2 = torch.stack(present[1]).clone().detach()

            if past is None:
                past = ([present1], [present2])
            else:
                past[0].append(present1)
                past[1].append(present2)
        else:
            present = torch.stack(present).clone().detach()
            if past is None:
                past = [present]
            else:
                past.append(present)

        logits = cutoff_topk_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        probs = cutoff_topp_probs(probs, top_p)

        idx = torch.multinomial(probs, num_samples=1).clone().detach()
        code = idx if code is None else torch.cat([code, idx], axis=1)

    del past
    return code


@torch.no_grad()
def sampling_igpt(model: torch.nn.Module,
                  sos: torch.FloatTensor,
                  top_k: Optional[float] = None,
                  top_p: Optional[float] = None,
                  softmax_temperature: float = 1.0,
                  is_tqdm: bool = True,
                  use_fp16: bool = True,
                  max_seq_len: int = 256) -> torch.LongTensor:
    code = None
    past = None
    pbar = tqdm(range(max_seq_len), total=max_seq_len) if is_tqdm else range(max_seq_len)

    for cnt, h in enumerate(pbar):
        if code is None:
            code_ = None
            pos_enc_code_ = None
        else:
            code_ = code.clone().detach()
            pos_enc_code_ = get_positional_encoding(code_, mode='1d')
            code_ = code_[:, cnt-1].unsqueeze(-1)
            pos_enc_code_ = pos_enc_code_[:, cnt-1].unsqueeze(-1)

        logits, present = model.sampling(sos=sos,
                                         codes=code_,
                                         pos_codes=pos_enc_code_,
                                         use_fp16=use_fp16,
                                         past=past)
        logits = logits.to(dtype=torch.float32)
        logits = logits / softmax_temperature

        present = torch.stack(present).clone().detach()
        if past is None:
            past = [present]
        else:
            past.append(present)

        logits = cutoff_topk_logits(logits, top_k)
        probs = F.softmax(logits, dim=-1)
        probs = cutoff_topp_probs(probs, top_p)

        idx = torch.multinomial(probs, num_samples=1).clone().detach()
        code = idx if code is None else torch.cat([code, idx], axis=1)

    del past
    return code


@torch.no_grad()
def sampling_ihqgpt(model: torch.nn.Module,
                    num_candidates: int,
                    cond: torch.LongTensor,
                    top_k_top: Optional[float] = None,
                    top_p_top: Optional[float] = None,
                    top_k_bot: Optional[float] = None,
                    top_p_bot: Optional[float] = None,
                    softmax_temperature: List[float] = [1.0, 1.0],
                    is_tqdm: bool = True,
                    use_fp16: bool = True,
                    max_seq_len: int = 256,
                    model_stage1: Optional[torch.nn.Module] = None,
                    given_top_code: Optional[torch.LongTensor] = None) -> torch.LongTensor:
    codes_top = None
    codes_bot = None
    past = None
    pbar = tqdm(range(max_seq_len), total=max_seq_len) if is_tqdm else range(max_seq_len)

    if model.use_cls_cond:
        sos = torch.LongTensor([cond]).cuda()
        sos = sos.repeat(num_candidates)
        sos = model.sos(sos).unsqueeze(1)
    elif model.use_txt_cond:
        pos_txt = torch.arange(0, model.idx_pred).unsqueeze(0).cuda()
        sos = model.tok_emb_txt(cond)
        sos += model.pos_emb_txt(pos_txt)
    else:
        sos = model.sos.repeat(num_candidates, 1, 1)

    for cnt, h in enumerate(pbar):
        if codes_top is None:
            _code_top = None
            _code_bot = None
            _pos_code = None

        else:
            _code_top = codes_top.clone().detach()
            _code_bot = codes_bot.clone().detach()
            _pos_code = get_positional_encoding(_code_top, mode='1d')
            _code_top = _code_top[:, cnt-1:cnt]
            _code_bot = _code_bot[:, cnt-1, :]
            _pos_code = _pos_code[:, cnt-1:cnt]

        if (given_top_code is not None):
            _given_top_code = given_top_code[:, cnt]
        else:
            _given_top_code = None

        code_top, code_bot, present = model.sampling_step(sos=sos,
                                                          codes_t=_code_top,
                                                          codes_b=_code_bot,
                                                          pos_codes=_pos_code,
                                                          use_fp16=use_fp16,
                                                          top_k_top=top_k_top,
                                                          top_p_top=top_p_top,
                                                          top_k_bot=top_k_bot,
                                                          top_p_bot=top_p_bot,
                                                          softmax_temperature=softmax_temperature,
                                                          past=past,
                                                          model_stage1=model_stage1,
                                                          given_top_code=_given_top_code)

        present = torch.stack(present).clone().detach()
        if past is None:
            past = [present]
        else:
            past.append(present)

        codes_top = code_top if codes_top is None else torch.cat([codes_top, code_top], axis=1)  # [B, HW]
        codes_bot = code_bot if codes_bot is None else torch.cat([codes_bot, code_bot], axis=1)  # [B, HW, KerH*KerW]

    del past
    return codes_top, codes_bot


@torch.no_grad()
def sampling_hqtransformer(model: torch.nn.Module,
                           num_candidates: int,
                           cond: torch.LongTensor,
                           top_k: Optional[List[float]] = None,
                           top_p: Optional[List[float]] = None,
                           softmax_temperature: List[float] = [1.0, 1.0, 1.0],
                           is_tqdm: bool = True,
                           use_fp16: bool = True,
                           max_seq_len: int = 256,
                           model_stage1: Optional[torch.nn.Module] = None) -> torch.LongTensor:

    codes_level = None
    past = None
    pbar = tqdm(range(max_seq_len), total=max_seq_len) if is_tqdm else range(max_seq_len)

    if model.use_cls_cond:
        sos = torch.LongTensor([cond]).cuda()
        sos = sos.repeat(num_candidates)
        sos = model.sos(sos).unsqueeze(1)
    elif model.use_txt_cond:
        pos_txt = torch.arange(0, model.idx_pred).unsqueeze(0).cuda()
        sos = model.tok_emb_txt(cond)
        sos += model.pos_emb_txt(pos_txt)
    else:
        sos = model.sos.repeat(num_candidates)

    for cnt, h in enumerate(pbar):
        if codes_level is None:
            _codes = [None for i in range(0, model.code_level)]
            _pos_code = None
        else:
            _code_top = codes_level[0].clone().detach()

            _codes = [_code_top[:, cnt-1:cnt]]
            if (model.code_level > 1):
                _code_mid = codes_level[1].clone().detach()
                _codes.append(_code_mid[:, cnt-1:cnt, :])
            if (model.code_level > 2):
                _code_bot = codes_level[2].clone().detach()
                _codes.append(_code_bot[:, cnt-1:cnt, :])

            _pos_code = get_positional_encoding(_code_top, mode='1d')
            _pos_code = _pos_code[:, cnt-1:cnt]

        codes_step, present = model.sampling_step(sos=sos,
                                                  codes=_codes,
                                                  pos_codes=_pos_code,
                                                  use_fp16=use_fp16,
                                                  top_k=top_k,
                                                  top_p=top_p,
                                                  softmax_temperature=softmax_temperature,
                                                  past=past,
                                                  model_stage1=model_stage1)

        present = torch.stack(present).clone().detach()
        if past is None:
            past = [present]
        else:
            past.append(present)

        if codes_level is None:
            codes_level = codes_step
        else:
            codes_level = list(map(lambda level, step: torch.cat([level, step], axis=1), codes_level, codes_step))

    del past
    return codes_level
