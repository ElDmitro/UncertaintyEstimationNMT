import os
import json
import math
import argparse
from itertools import islice, zip_longest
from collections import namedtuple, defaultdict

import torch

from fairseq import progress_bar
from fairseq import bleu, data, options, tasks, tokenizer, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.sequence_generator import SequenceGenerator
from fairseq.sequence_scorer import SequenceScorer
from fairseq.models import FairseqIncrementalDecoder
from sacrebleu import sentence_bleu, corpus_bleu
from subword_nmt.apply_bpe import BPE
from sacremoses import MosesTokenizer
import codecs


import networkx as nx

import tqdm
from tqdm import tqdm_notebook
import pandas as pd
import numpy as np


def shannon_entropy(pk, dim=None):
    if dim is None:
        return -torch.sum(pk * torch.log(pk))

    return -torch.sum(pk * torch.log(pk), dim=dim)


Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')
def make_batches(lines, args, task, max_positions):
    tokens = [
        task.source_dictionary.encode_line(
            src_str, add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


class EnsembleModel(torch.nn.Module):
    """A wrapper around an ensemble of models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.incremental_states = None
        if all(hasattr(m, 'decoder') and isinstance(m.decoder, FairseqIncrementalDecoder) for m in models):
            self.incremental_states = {m: {} for m in models}

    def has_encoder(self):
        return hasattr(self.models[0], 'encoder')

    def max_decoder_positions(self):
        return min(m.max_decoder_positions() for m in self.models)

    @torch.no_grad()
    def forward_encoder(self, encoder_input):
        if not self.has_encoder():
            return None
        return [model.encoder(**encoder_input) for model in self.models]

    @torch.no_grad()
    def forward_decoder(self, tokens, encoder_outs, temperature=1., with_var=True):
        if len(self.models) == 1:
            probs, attn, pvars, pentropy = self._decode_one(
                tokens,
                self.models[0],
                encoder_outs[0] if self.has_encoder() else None,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
                with_var=True
            )
            if with_var:
                probs_vars = torch.zeros(probs.size(), device=probs.device)
                ens_var = probs.var(-1)
                return probs, attn, probs_vars, probs_vars, torch.stack([pvars], dim=0), torch.stack([pentropy], dim=0), ens_var, ens_var
            return probs, attn, None

        log_probs = []
        sing_var = []
        sing_entropy = []
        avg_attn = None
        for model, encoder_out in zip(self.models, encoder_outs):
            probs, attn, pvars, pentropy = self._decode_one(
                tokens,
                model,
                encoder_out,
                self.incremental_states,
                log_probs=True,
                temperature=temperature,
                with_var=True
            )
            log_probs.append(probs)
            sing_var.append(pvars)
            sing_entropy.append(pentropy)
            if attn is not None:
                if avg_attn is None:
                    avg_attn = attn
                else:
                    avg_attn.add_(attn)
        
        avg_probs = torch.logsumexp(torch.stack(log_probs, dim=0), dim=0) - math.log(len(self.models))

        e_avg_probs = torch.exp(avg_probs)
        ensemble_var = e_avg_probs.var(-1)
        ensemble_entropy = shannon_entropy(e_avg_probs, -1)

        e_probs = torch.exp(torch.stack(log_probs, dim=0))
        probs_mean = e_probs.mean(dim=0)
        probs_var = e_probs.var(dim=0)

        sing_var = torch.stack(sing_var, dim=0)
        sing_entropy = torch.stack(sing_entropy, dim=0)
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        if with_var:
            return avg_probs, avg_attn, probs_mean, probs_var, sing_var, sing_entropy, ensemble_var, ensemble_entropy, e_probs
        return avg_probs, avg_attn, probs_var

    def _decode_one(
        self, tokens, model, encoder_out, incremental_states, log_probs,
        temperature=1.,
        with_var=False
    ):
        if self.incremental_states is not None:
            decoder_out = list(model.forward_decoder(
                tokens, encoder_out=encoder_out, incremental_state=self.incremental_states[model],
            ))
        else:
            decoder_out = list(model.forward_decoder(tokens, encoder_out=encoder_out))
        decoder_out[0] = decoder_out[0][:, -1:, :]
        if temperature != 1.:
            decoder_out[0].div_(temperature)
        attn = decoder_out[1] if len(decoder_out) > 1 else None
        if type(attn) is dict:
            attn = attn.get('attn', None)
        if type(attn) is list:
            attn = attn[0]
        if attn is not None:
            attn = attn[:, -1, :]
        probs = model.get_normalized_probs(decoder_out, log_probs=log_probs)
        probs = probs[:, -1, :]
        e_probs = torch.exp(probs)
        if with_var:
            return probs, attn, e_probs.var(dim=1), shannon_entropy(e_probs, 1) 
        return probs, attn

    def reorder_encoder_out(self, encoder_outs, new_order):
        if not self.has_encoder():
            return
        return [
            model.encoder.reorder_encoder_out(encoder_out, new_order)
            for model, encoder_out in zip(self.models, encoder_outs)
        ]

    def reorder_incremental_state(self, new_order):
        if self.incremental_states is None:
            return
        for model in self.models:
            model.decoder.reorder_incremental_state(self.incremental_states[model], new_order)


class SourceSequenceGenerator(SequenceGenerator):
    @torch.no_grad()
    def generate(self, models, sample, **kwargs):
        """Generate a batch of translations.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        model = EnsembleModel(models)
        return self._generate(model, sample, **kwargs)

    @torch.no_grad()
    def _generate(
        self,
        model,
        sample,
        prefix_tokens=None,
        bos_token=None,
        with_var=False,
        **kwargs
    ):
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }
        
        return_all_tokens = False
        if 'return_all_tokens' in kwargs:
            return_all_tokens = kwargs['return_all_tokens']
            
        src_tokens = encoder_input['src_tokens']
        src_lengths = (src_tokens.ne(self.eos) & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        models_num = len(model.models)
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )
        assert self.min_len <= max_len, 'min_len cannot be larger than max_len, please adjust these!'

        # compute the encoder output for each beam
        encoder_outs = model.forward_encoder(encoder_input)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)

        # initialize buffers
        scores = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.new(bsz * beam_size, max_len + 2).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn, attn_buf = None, None

        # The blacklist indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then the blacklist would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        blacklist = src_tokens.new_zeros(bsz, beam_size).eq(-1)  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz
        
        if return_all_tokens:
            all_tokens = tokens.data.new(max_len + 2, 2 * bsz * beam_size).fill_(self.pad)
            all_scores = scores.data.new(max_len + 2, 2 * bsz * beam_size).fill_(0)
            all_softmaxes = torch.zeros(max_len + 2, beam_size, self.vocab_size)
            all_vars = scores.data.new(max_len + 2, 2 * bsz * beam_size).fill_(0)
            all_vars_vocab = torch.zeros(max_len + 2, beam_size, self.vocab_size)
            all_means = scores.data.new(max_len + 2, 2 * bsz * beam_size).fill_(0)
            all_means_vocab = torch.zeros(max_len + 2, beam_size, self.vocab_size)
            all_sing_vars = torch.zeros(max_len + 2, models_num, beam_size)
            all_sing_entropy = torch.zeros(max_len + 2, models_num, beam_size)
            all_ens_vars = torch.zeros(max_len + 2, beam_size)
            all_ens_entropy = torch.zeros(max_len + 2, beam_size)
            is_finalized = torch.zeros(max_len + 2, bsz * beam_size, dtype=torch.uint8)
            all_bbsz_idx = torch.zeros(max_len + 2, 2 * bsz * beam_size, dtype=torch.uint8)
            all_inens_dist = torch.zeros(max_len + 2, models_num, beam_size, self.vocab_size)
            all_lprobs = []

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfin_idx):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size or step == max_len:
                return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]  # skip the first index, which is EOS
            assert not tokens_clone.eq(self.eos).any()
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step+2] if attn is not None else None

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step+1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                if self.match_source_len and step > src_lengths[unfin_idx]:
                    score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i]
                    else:
                        hypo_attn = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': None,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                    if return_all_tokens:
                        is_finalized[step, idx] = 1

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfin_idx):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(corr.unsqueeze(-1) * beam_size)
                model.reorder_incremental_state(reorder_state)
                encoder_outs = model.reorder_encoder_out(encoder_outs, reorder_state)

            lprobs, avg_attn_scores, probs_means, probs_vars, sing_vars, sing_entropy, ens_vars, ens_entropy, inens_dist = model.forward_decoder(
                tokens[:, :step + 1], encoder_outs, temperature=self.temperature, with_var=True
            )
            lprobs[lprobs != lprobs] = -math.inf

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if prefix_tokens is not None and step < prefix_tokens.size(1) and step < max_len:
                prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(1, beam_size).view(-1)
                prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_vars = probs_vars.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_means = probs_means.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_mask = prefix_toks.ne(self.pad)
                lprobs[prefix_mask] = -math.inf
                # TODO
                lprobs[prefix_mask] = lprobs[prefix_mask].scatter_(
                    -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_lprobs[prefix_mask]
                )
                probs_vars[prefix_mask] = probs_vars[prefix_mask].scatter_(
                    -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_vars[prefix_mask]
                )
                probs_means[prefix_mask] = probs_means[prefix_mask].scatter_(
                    -1, prefix_toks[prefix_mask].unsqueeze(-1), prefix_means[prefix_mask]
                )
                # if prefix includes eos, then we should make sure tokens and
                # scores are the same across all beams
                eos_mask = prefix_toks.eq(self.eos)
                if eos_mask.any():
                    # validate that the first beam matches the prefix
                    first_beam = tokens[eos_mask].view(-1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
                    eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
                    target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
                    assert (first_beam == target_prefix).all()

                    def replicate_first_beam(tensor, mask):
                        tensor = tensor.view(-1, beam_size, tensor.size(-1))
                        tensor[mask] = tensor[mask][:, :1, :]
                        return tensor.view(-1, tensor.size(-1))

                    # copy tokens, scores and lprobs from the first beam to all beams
                    tokens = replicate_first_beam(tokens, eos_mask_batch_dim)
                    scores = replicate_first_beam(scores, eos_mask_batch_dim)
                    lprobs = replicate_first_beam(lprobs, eos_mask_batch_dim)
                    probs_vars = replicate_first_beam(probs_vars, eos_mask_batch_dim)
                    probs_means = replicate_first_beam(probs_means, eos_mask_batch_dim)
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            if self.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                gen_ngrams = [{} for bbsz_idx in range(bsz * beam_size)]
                for bbsz_idx in range(bsz * beam_size):
                    gen_tokens = tokens[bbsz_idx].tolist()
                    for ngram in zip(*[gen_tokens[i:] for i in range(self.no_repeat_ngram_size)]):
                        gen_ngrams[bbsz_idx][tuple(ngram[:-1])] =                                 gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), []) + [ngram[-1]]

            # Record attention scores
            if type(avg_attn_scores) is list:
                avg_attn_scores = avg_attn_scores[0]
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1), max_len + 2)
                    attn_buf = attn.clone()
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)

            self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:
                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    ngram_index = tuple(tokens[bbsz_idx, step + 2 - self.no_repeat_ngram_size:step + 1].tolist())
                    return gen_ngrams[bbsz_idx].get(ngram_index, [])

                if step + 2 - self.no_repeat_ngram_size >= 0:
                    # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
                    banned_tokens = [calculate_banned_tokens(bbsz_idx) for bbsz_idx in range(bsz * beam_size)]
                else:
                    banned_tokens = [[] for bbsz_idx in range(bsz * beam_size)]

                for bbsz_idx in range(bsz * beam_size):
                    lprobs[bbsz_idx, banned_tokens[bbsz_idx]] = -math.inf


            clean_lprobs = lprobs.clone().detach()
            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
            )
            
            ishape = cand_indices.shape[1]
            cand_vars = torch.ones((bsz, ishape))
            if with_var:
                boffsets = (torch.cumsum(
                    torch.full((bsz, ), ishape, dtype=torch.int64, device=cand_indices.device) - ishape,
                    dim=0
                )).unsqueeze_(-1).T

                boffset_idxs = (cand_indices + boffsets).flatten()
                cand_vars = probs_vars.flatten()[boffset_idxs].view(bsz, -1)
                cand_means = probs_means.flatten()[boffset_idxs].view(bsz, -1)

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            
            if return_all_tokens:
                all_scores[step + 1] = cand_scores #-scores[cand_beams,step-1]
                all_softmaxes[step + 1] = clean_lprobs
                all_vars[step + 1] = cand_vars
                all_vars_vocab[step + 1] = probs_vars
                all_means[step + 1] = cand_means
                all_means_vocab[step + 1] = probs_means
                all_tokens[step + 1] = cand_indices
                all_bbsz_idx[step] = cand_bbsz_idx
                all_sing_vars[step + 1] = sing_vars
                all_sing_entropy[step + 1] = sing_entropy
                all_ens_vars[step + 1] = ens_vars
                all_ens_entropy[step + 1] = ens_entropy
                all_inens_dist[step + 1] = inens_dist



            # finalize hypotheses that end in eos, except for blacklisted ones
            # or candidates with a score of -inf
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][blacklist] = 0

            # only consider eos when it's among the top beam_size indices
            torch.masked_select(
                cand_bbsz_idx[:, :beam_size],
                mask=eos_mask[:, :beam_size],
                out=eos_bbsz_idx,
            )

            finalized_sents = set()
            if eos_bbsz_idx.numel() > 0:
                torch.masked_select(
                    cand_scores[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_scores,
                )
                finalized_sents = finalize_hypos(step, eos_bbsz_idx, eos_scores)
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                blacklist = blacklist[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos or
            # blacklisted hypos and values < cand_size indicate candidate
            # active hypos. After this, the min values per row are the top
            # candidate active hypos.
            active_mask = buffer('active_mask')
            eos_mask[:, :beam_size] |= blacklist
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, new_blacklist = buffer('active_hypos'), buffer('new_blacklist')
            torch.topk(
                active_mask, k=beam_size, dim=1, largest=False,
                out=(new_blacklist, active_hypos)
            )

            # update blacklist to ignore any finalized hypos
            blacklist = new_blacklist.ge(cand_size)[:, :beam_size]
            assert (~blacklist).any(dim=1).all()

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx, dim=1, index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1], dim=0, index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(
                cand_indices, dim=1, index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step], dim=0, index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores, dim=1, index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2], dim=0, index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent in range(len(finalized)):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r['score'], reverse=True)
            
        if return_all_tokens:
            all_scores = all_scores[:step + 2]
            all_softmaxes = all_softmaxes[:step + 2]
            all_vars = all_vars[:step + 2]
            all_vars_vocab = all_vars_vocab[:step + 2]
            all_means = all_means[:step + 2]
            all_means_vocab = all_means_vocab[:step + 2]
            all_sing_vars = all_sing_vars[:step + 2]
            all_sing_entropy = all_sing_entropy[:step + 2]
            all_ens_vars = all_ens_vars[:step + 2]
            all_ens_entropy = all_ens_entropy[:step + 2]
            all_inens_dist = all_inens_dist[:step + 2]
            if with_var:
                    return finalized, all_tokens[:step + 2].cpu(), all_scores.cpu(), is_finalized[:step + 1].cpu(), all_bbsz_idx[:step+1].cpu(), all_softmaxes.cpu(), all_means.cpu(), all_means_vocab.cpu(), all_vars.cpu(), all_vars_vocab.cpu(), all_sing_vars.cpu(), all_sing_entropy.cpu(), all_ens_vars.cpu(), all_ens_entropy.cpu(), all_inens_dist.cpu()
                
            return finalized, all_tokens[:step + 2].cpu(), all_scores.cpu(), is_finalized[:step + 1].cpu(), all_bbsz_idx[:step+1].cpu()
        return finalized


def get_stats_distribution(ref_tokens, tgt_tokens, token_cmp, inens_mean_vocab, inens_var_vocab, *args):
    ref_len = ref_tokens.shape[0]
    tgt_len = tgt_tokens.shape[0]
    max_len = min(ref_len, tgt_len)
   
    idxs = torch.arange(max_len)
    mask = token_cmp(ref_tokens[:max_len], tgt_tokens[:max_len])
    ffalse_idx = max_len
    if not mask.all():
        ffalse_idx = idxs[~mask][0]
    mask[ffalse_idx:] = False

    stats_true = dict()
    stats_false = dict()
    stats_falsetrue = dict()
    for name, score in args:
        stats_true[name] = score[:max_len][mask].tolist()
        if ffalse_idx < max_len:
            stats_false[name] = [score[:max_len][ffalse_idx].tolist()]


    if ffalse_idx < max_len:
        stats_falsetrue['inens_mean'] = [inens_mean_vocab[:max_len][ffalse_idx][ref_tokens[ffalse_idx]].tolist()]
        stats_falsetrue['inens_var'] = [inens_var_vocab[:max_len][ffalse_idx][ref_tokens[ffalse_idx]].tolist()]

    stats_true['tokens'] = tgt_tokens[:max_len][mask].tolist()
    if ffalse_idx < max_len:
        stats_false['tokens'] = [tgt_tokens[:max_len][ffalse_idx].tolist()]
        
    return stats_true, stats_false, stats_falsetrue
        

def get_translation_stats(tgt_tokens, beam_tokens, bbsz_idx, beam_scores, beam_means, beam_means_vocab, beam_vars, beam_vars_vocab, beam_softmaxes, inens_dist):
    tgt_len = tgt_tokens.shape[0]
    beam_size = beam_softmaxes.size(1)

    last_idx = 0
    tscores = []
    tvars = []
    tmeans = []
    tsoftmaxes = []
    tmeans_vocab = []
    tvars_vocab = []
    tinens_dist = []
    idx = torch.arange(beam_tokens.shape[1])
    for i in range(1, tgt_len + 1):
        mask = (beam_tokens[i] == tgt_tokens[i - 1]) & (bbsz_idx[i - 1] == last_idx)
        half_mask = mask[:beam_size]

        tscores.append(beam_scores[i][mask][0])
        tvars.append(beam_vars[i][mask][0])
        tmeans.append(beam_means[i][mask][0])
        tsoftmaxes.append(beam_softmaxes[i][half_mask][0])
        tmeans_vocab.append(beam_means_vocab[i][half_mask][0])
        tvars_vocab.append(beam_vars_vocab[i][half_mask][0])
        tinens_dist.append(inens_dist[i][:, half_mask][:, 0][:, tgt_tokens[i - 1]])
        last_idx = idx[mask][0]

    return torch.stack(tscores), torch.stack(tmeans), torch.stack(tmeans_vocab), torch.stack(tvars), torch.stack(tvars_vocab), torch.stack(tsoftmaxes), torch.stack(tinens_dist)


def get_eos_stats_distribution(tgt_dict, tokens, probs, inens_vars):
    n, m = tokens.shape

    eos_token = tgt_dict.eos()
    idxs = torch.repeat_interleave(torch.arange(n), m).view(n, m)
    eos_mask = tokens == eos_token

    return {
        'probs': probs[eos_mask].tolist(),
        'inens_vars': inens_vars[eos_mask].tolist(),
        'token_idxs': idxs[eos_mask].tolist(),
    }


def get_wrong_suff_stats_distribution(ref_tokens, tgt_tokens, *args):
    ref_len = ref_tokens.shape[0]
    tgt_len = tgt_tokens.shape[0]
    max_len = min(ref_len, tgt_len)

    mask = ref_tokens[:max_len] == tgt_tokens[:max_len]
    true_mask = mask.clone()
    idxs = torch.arange(max_len)

    if mask.all():
        return dict()

    sidx = idxs[~mask][0]
    mask[sidx:] = False
    mask = torch.cat((
        mask,
        torch.full((tgt_len - max_len, ), False, dtype=torch.bool)
    ))
    true_mask = torch.cat((
        true_mask,
        torch.full((tgt_len - max_len, ), False, dtype=torch.bool)
    ))

    stats = dict()
    for name, score in args:
        stats[name] = score[~mask].tolist()
    stats['tokens'] = tgt_tokens[~mask].tolist()
    assert 2 in stats['tokens']
    stats['is_true'] = true_mask[~mask].tolist()

    return stats


# PARSER  ---------------------------------------------------------------------------------------------------------------
inparser = argparse.ArgumentParser()
inparser.add_argument('--model-path', type=str, required=True)
inparser.add_argument('--checkpoint-paths', nargs='*', type=str, required=True)
inparser.add_argument('--data-path', type=str, required=True)
inparser.add_argument('--bpe-path', type=str, required=True)
inparser.add_argument('--beam', type=str, default='5')
inparser.add_argument('--lenpen', type=str, default='0')
inparser.add_argument('--diverse-beam-strength', type=str, default='0')
inparser.add_argument('--shared-bpe', action='store_true', default=False)
inparser.add_argument('--src-lang', type=str, required=True)
inparser.add_argument('--tgt-lang', type=str, required=True)
inparser.add_argument('--cpu', action='store_true', default=False)
inparser.add_argument('--log-dir', type=str, default='.')
inparser.add_argument('--device-id', type=int, default=0)
inparser.add_argument('--max-sentences', type=int, default=0)

inargs = inparser.parse_args()
#  -----------------------------------------------------------------------------------------------------------------------
checkpoint_paths = [os.path.join(inargs.model_path, path) for path in inargs.checkpoint_paths]
model_path = ':'.join(checkpoint_paths)

BPECODES_PATH = inargs.bpe_path
SHARED_BPE = inargs.shared_bpe
SRS = inargs.src_lang
TGT = inargs.tgt_lang

tkn = {}
bpe = {}
if not SHARED_BPE:
    for l in [SRS, TGT]:
        with codecs.open(BPECODES_PATH) as src_codes:
            tkn[l] = MosesTokenizer(lang=l)
            bpe[l] = BPE(src_codes)
else:
    l = SRS
    with codecs.open(BPECODES_PATH) as src_codes:
        tkn[l] = MosesTokenizer(lang=l)
        bpe[l] = BPE(src_codes)

def prepare_input(s, l='en'):
    return [bpe[l].process_line(tkn[l].tokenize(s, return_str=True))]


parser = options.get_generation_parser(interactive=True)
args = options.parse_args_and_arch(parser, input_args=[
    inargs.data_path,
    '--path', model_path,
    '--diverse-beam-strength', inargs.diverse_beam_strength,
    '--lenpen', inargs.lenpen,
    '--remove-bpe',
    '--beam', inargs.beam
])

print(inargs)
print('device-id', torch.cuda.current_device())
print('device-environ', os.environ['CUDA_VISIBLE_DEVICES'])
use_cuda = torch.cuda.is_available() and not inargs.cpu

task = tasks.setup_task(args)
model_paths = args.path.split(':')
models, model_args = utils.load_ensemble_for_inference(
        model_paths,
        task,
        model_arg_overrides=eval(args.model_overrides)
)
src_dict = task.source_dictionary
tgt_dict = task.target_dictionary


for model in models:
    model.make_generation_fast_(
        beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
        need_attn=args.print_alignment,
    )
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()


align_dict = utils.load_align_dict(args.replace_unk)
max_positions = utils.resolve_max_positions(
    task.max_positions(),
    *[model.max_positions() for model in models]
)
# ----------------------------------------------------------------------------------------

translator = SourceSequenceGenerator(
    tgt_dict=tgt_dict,
    beam_size=args.beam,
    min_len=args.min_len,
    normalize_scores=(not args.unnormalized),
    len_penalty=args.lenpen,
    unk_penalty=args.unkpen
)
task.load_dataset(args.gen_subset)
itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=inargs.max_sentences if inargs.max_sentences > 0 else args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
).next_epoch_itr(shuffle=False)
progress = progress_bar.build_progress_bar(
        args,
        itr
)

global_positive_stats = defaultdict(list)
global_negative_stats = defaultdict(list)
global_negative_true_stats = defaultdict(list)
global_eos_stats = defaultdict(list)
global_wrong_suff_stats = defaultdict(list)
global_positive_stats_by_len = defaultdict(dict)
global_negative_stats_by_len = defaultdict(dict)
for batch in progress:
    nsentences = batch['nsentences']
    for i in tqdm.tqdm(range(nsentences)):
        sample = {
                'net_input': {
                    'src_tokens': batch['net_input']['src_tokens'][i:i+1],
                    'src_lengths': batch['net_input']['src_lengths'][i:i+1],
                }
        }
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        ref_tokens = batch['target'][i]

        translations, all_tokens, all_scores, is_finalized, all_bbsz_idx, all_softmaxes, all_means, all_means_vocab, all_vars, all_vars_vocab, all_sing_vars, all_sing_entropy, all_ens_vars, all_ens_entropy, all_inens_dist = translator.generate(
            models=models,
            sample=sample,
            return_all_tokens=True,
            with_var=True
        )

        all_sing_vars = all_sing_vars.mean(-1)
        all_sing_entropy = all_sing_entropy.mean(-1)
        all_ens_vars = all_ens_vars.mean(-1)
        all_ens_entropy = all_ens_entropy.mean(-1)

        tgt_tokens = translations[0][0]['tokens'].cpu()
        tgt_len = tgt_tokens.shape[0]
        probs, inens_mean, inens_mean_vocab, inens_var, inens_var_vocab, ens_softmaxes, inens_dist = get_translation_stats(tgt_tokens, all_tokens, all_bbsz_idx, all_scores, all_means, all_means_vocab, all_vars, all_vars_vocab, all_softmaxes, all_inens_dist)
        all_ens_vars = all_ens_vars[1:tgt_len + 1]
        all_ens_entropy = all_ens_entropy[1:tgt_len + 1]
        all_sing_vars = all_sing_vars[1:tgt_len + 1]
        all_sing_entropy = all_sing_entropy[1:tgt_len + 1]

        # TODO(eldmitro): collect ens softmax wisely
        scores = [
            ('prob', probs),
            ('inens_var', inens_var),
            ('inens_mean', inens_mean),
            ('ens_softmax', torch.exp(ens_softmaxes)),
            ('inens_dist', inens_dist),
            ('ens_svar', all_ens_vars),
            ('ens_entropy', all_ens_entropy),
        ]
        for i in range(len(models)):
            name = 'm{}_svar'.format(i)
            scores.append((name, all_sing_vars[:, i]))
            name = 'm{}_entropy'.format(i)
            scores.append((name, all_sing_entropy[:, i]))
        
        positive_stats, negative_stats, negative_true_stats = get_stats_distribution(
            ref_tokens,
            tgt_tokens,
            lambda x, y: x == y,
            inens_mean_vocab,
            inens_var_vocab,
            *scores
        )
        eos_stats = get_eos_stats_distribution(
            tgt_dict,
            all_tokens,
            all_scores,
            all_vars
        )
        wrong_suff_stats = get_wrong_suff_stats_distribution(
            ref_tokens,
            tgt_tokens,
            *scores
        )

        for key in positive_stats:
            global_positive_stats[key].extend(positive_stats[key])
            if key not in global_positive_stats_by_len[tgt_len]:
                global_positive_stats_by_len[tgt_len][key] = list()
            global_positive_stats_by_len[tgt_len][key].extend(positive_stats[key])
        for key in negative_stats:
            global_negative_stats[key].extend(negative_stats[key])
            if key not in global_negative_stats_by_len[tgt_len]:
                global_negative_stats_by_len[tgt_len][key] = list()
            global_negative_stats_by_len[tgt_len][key].extend(negative_stats[key])
        for key in negative_true_stats:
            global_negative_true_stats[key].extend(negative_true_stats[key])
        for key in eos_stats:
            global_eos_stats[key].extend(eos_stats[key])
        for key in wrong_suff_stats:
            global_wrong_suff_stats[key].extend(wrong_suff_stats[key])
    break


log_dir = inargs.log_dir
with open(os.path.join(log_dir, 'positive_stats.json'), 'w') as stream_output:
    json.dump(global_positive_stats, stream_output)
with open(os.path.join(log_dir, 'negative_stats.json'), 'w') as stream_output:
    json.dump(global_negative_stats, stream_output)
with open(os.path.join(log_dir, 'negative_true_stats.json'), 'w') as stream_output:
    json.dump(global_negative_true_stats, stream_output)
with open(os.path.join(log_dir, 'eos_stats.json'), 'w') as stream_output:
    json.dump(global_eos_stats, stream_output)
with open(os.path.join(log_dir, 'wrong_suff_stats.json'), 'w') as stream_output:
    json.dump(global_wrong_suff_stats, stream_output)
with open(os.path.join(log_dir, 'positive_stats_by_len.json'), 'w') as stream_output:
    json.dump(global_positive_stats_by_len, stream_output)
with open(os.path.join(log_dir, 'negative_stats_by_len.json'), 'w') as stream_output:
    json.dump(global_negative_stats_by_len, stream_output)
