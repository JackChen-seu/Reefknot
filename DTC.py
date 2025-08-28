# --- DTC.py (rewritten, safe for fp16 & actually patches LLaVA class) ---
import math
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.utils import (
    GenerationMixin,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput,
)


from llava.model.language_model import llava_llama


def _stash_dtc_to_config(self, kwargs: dict):

    for k in ("apha", "threshold", "layer"):
        if k in kwargs:
            setattr(self.generation_config, f"dtc_{k}", kwargs.pop(k))


def DTC_function():
   
    _orig_generate = llava_llama.LlavaLlamaForCausalLM.generate

    def _generate_patch(self, *args, **kwargs):
        _stash_dtc_to_config(self, kwargs)
        return _orig_generate(self, *args, **kwargs)

    llava_llama.LlavaLlamaForCausalLM.generate = _generate_patch

    
    def greedy_search_redefine(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList = None,
        stopping_criteria: StoppingCriteriaList = None,
        max_length: int = None,
        pad_token_id: int = None,
        eos_token_id=None,
        output_attentions: bool = None,
        output_hidden_states: bool = None,
        output_scores: bool = None,
        return_dict_in_generate: bool = None,
        synced_gpus: bool = False,
        streamer=None,
        **model_kwargs,  
    ):
        
        apha = getattr(self.generation_config, "dtc_apha", None)            
        threshold = getattr(self.generation_config, "dtc_threshold", None)
        layer = getattr(self.generation_config, "dtc_layer", None)

        
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)

        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = (
            torch.tensor(eos_token_id, device=input_ids.device) if eos_token_id is not None else None
        )

        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        
        output_hidden_states = True
        return_dict_in_generate = True

       
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None

        
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        this_peer_finished = False  # for ZeRO stage 3
        first = True
        layer_scores = {}

        while True:
            if synced_gpus:
                
                flag = torch.tensor(0.0 if this_peer_finished else 1.0, device=input_ids.device)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.SUM)
                if flag.item() == 0.0:
                    break

            
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=True,
            )

            
            final_layer_idx = len(outputs.hidden_states) - 1

           
            if layer is None:
                base_layer_idx = final_layer_idx
            else:
                base_layer_idx = max(0, min(int(layer), final_layer_idx))

            
            final_logits_step = self.lm_head(outputs.hidden_states[final_layer_idx])[:, -1, :]  # [bsz, vocab]
            base_logits_step = self.lm_head(outputs.hidden_states[base_layer_idx])[:, -1, :]    # [bsz, vocab]

            
            softmax_final = F.softmax(final_logits_step.float(), dim=-1)
           
            yes_prob = softmax_final.flatten()[3869].item()
            no_prob = softmax_final.flatten()[1939].item()

            if threshold is not None and yes_prob > 0.0 and no_prob > 0.0:
                yes_no_entropy = -(yes_prob * math.log2(yes_prob) + no_prob * math.log2(no_prob))
            else:
                yes_no_entropy = 0.0

            
            use_dtc = (threshold is not None) and (apha is not None) and (yes_no_entropy >= float(threshold))
            if use_dtc:
                relative_top = 0.1


                final_logits_norm = final_logits_step.float().log_softmax(dim=-1)  # [bsz, vocab]
                base_logits_norm = base_logits_step.float().log_softmax(dim=-1)    # [bsz, vocab]

                if relative_top > 0.0:

                    sorted_logits, _ = torch.sort(final_logits_norm, descending=True)
                    min_thresh = sorted_logits[..., 0]  # [bsz]
                    probs_max = torch.max(final_logits_norm, dim=-1).values  # [bsz]
                    probs_thresh = torch.min(min_thresh, probs_max + math.log(relative_top)).unsqueeze(-1)  # [bsz,1]
                    mask = final_logits_norm < probs_thresh  # [bsz, vocab]

       
                    final_logits_norm = final_logits_norm.masked_fill(mask, float("-inf"))
                    base_logits_norm = base_logits_norm.masked_fill(mask, float("-inf"))

                process_logits = (1.0 + float(apha)) * final_logits_norm - float(apha) * base_logits_norm
                next_token_logits = process_logits.to(final_logits_step.dtype)  # 还原到原 dtype（可能是 half）
            else:
               
                next_token_logits = final_logits_step

            
            if first:
                for i, hs in enumerate(outputs.hidden_states):
                    layer_logits_i = self.lm_head(hs)[:, -1, :]  # [bsz, vocab]
                    layer_scores[i] = logits_processor(input_ids, layer_logits_i)
                first = False

            
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            
            if return_dict_in_generate and output_scores:
                scores += (next_tokens_scores,)

            if return_dict_in_generate:
                if output_attentions:
                    if self.config.is_encoder_decoder:
                        decoder_attentions += (outputs.decoder_attentions,)
                        cross_attentions += (outputs.cross_attentions,)
                    else:
                        decoder_attentions += (outputs.attentions,)
                if output_hidden_states:
                    if self.config.is_encoder_decoder:
                        decoder_hidden_states += (outputs.decoder_hidden_states,)
                    else:
                        decoder_hidden_states += (outputs.hidden_states,)

            
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

           
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if streamer is not None:
                streamer.put(next_tokens.cpu())

            
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

          
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                    .ne(eos_token_id_tensor.unsqueeze(1))
                    .prod(dim=0)
                )
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        
        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return layer_scores, GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return layer_scores, GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids

    
    GenerationMixin.greedy_search = greedy_search_redefine
    llava_llama.LlavaLlamaForCausalLM.greedy_search = greedy_search_redefine

    print(
        "[DTC] Patched LlavaLlamaForCausalLM.generate + greedy_search; "
        "custom params read from generation_config.",
        flush=True,
    )
