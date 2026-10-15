# Ablations: paper table → command

Every row below runs on Qwen2.5-VL-7B-Instruct with greedy decoding and alpha 1.5 unless
the row says otherwise. Run them with

```bash
vlmcfg-eval --model qwen2_5_vl_vlmcfg --model_args "<args>" --tasks <task> --batch_size 1 --log_samples
```

An empty `<args>` is the paper's method: image deletion, hidden-state blend, the switch
reference at K = 8 and the APC guard (beta 0.1). The unguided rows use
`--model qwen2_5_vl_base`. Each knob reproduces the research code the paper was run with
bitwise (identical decision logits); its defaults are that code's defaults.

## Where the reference stands (Tables `reference`, `app_schedules`, `app_ksweep`, `app_window`, `app_arms`)

| row | `--model_args` |
|---|---|
| TF | `switch_k=0,guard=none` |
| i-1 | `reference=cached,guard=none` |
| i-1, strength oracle | `switch_k=0,rule=strength_oracle,guard=none` |
| i-1, decay schedule | `switch_k=0,rule=decay,guard=none` |
| i-1, capped at step 0 | `switch_k=0,rule=cap,guard=none` |
| i-1, first token only | `reference=cached,guide_steps=1,guard=none` |
| sw, K = 4 / 8 / 16 / 32 | `switch_k=<K>,guard=none` |
| prefix window w | `reference=window,window=<w>,guard=none` |
| + EOS | add `guard=eos` (i-1 + EOS: `reference=window,window=0,guard=eos`) |
| + APC | add `guard=apc` |

`feature=logprobs` blends fp32 log-probabilities instead of the hidden state, as the
earliest teacher-forced runs did.

## Operators and scale (Tables `operators`, `app_alpha_sweep`)

All on sw (K = 8) without a guard: `guard=none` plus

| row | `--model_args` |
|---|---|
| alpha ladder | `alpha=<a>` |
| APG, eta = 0.5 / 2 | `operator=apg,apg_eta=<eta>` |
| CFG-Zero* | `operator=zerostar` |
| TCFG | `operator=tcfg` |
| EP-CFG | `operator=epcfg` |

## Where the blend acts (Table `sites`)

| row | `--model_args` |
|---|---|
| logits | `reference=cached,guard=none,feature=logits` |
| last hidden state | `reference=cached,guard=none` |

## Negative streams (Tables `negatives`, `app_tuned`, `app_weak_geometry`)

On the cached reference: `reference=cached,guard=none` plus

| row | `--model_args` |
|---|---|
| early exit, layer L | `negative=early_exit,exit_layer=<L>` |
| smaller model (3B) | `feature=logits,negative=weak_model,weak_model=Qwen/Qwen2.5-VL-3B-Instruct` |
| MUL, gamma | `negative=attention,attn_op=mul,attn_gamma=<gamma>` |
| SEG | `negative=attention,attn_op=blur,attn_blur_sigma_frac=<f>,attn_renorm=preserve_ratio` |
| Over-blur, a | `negative=attention,attn_op=sharpen,attn_blur_sigma_frac=<f>,attn_alpha=<a>` |
| Soft PAG, lambda / PAG | `negative=attention,attn_op=peak,attn_peak_lambda=<lambda>` (PAG: 1.0) |
| Masked | `negative=attention,attn_op=masked` |

SEG and Over-blur also take the kernel width in patches (`attn_blur_sigma`) instead of a
fraction; `attn_layers="[lo-hi]"` restricts any edit to a layer range (default: all but
the first layer).

## Region knowledge (Tables `app_localized`, `app_attn_edit`)

The region rows read the task's annotation: `region=seg,region_field=segmentation` on
VizWiz, `region=bbox` with `region_field=bboxes` (V*), `target_instances` (TreeBench) or
`bboxs` (OpenImages). These rows use the logit blend, as the paper's runs did:
`reference=cached,guard=none,feature=logits`.

| row | model | `--model_args` (besides the region) |
|---|---|---|
| region attention, mul gamma | base | `cond_attn_op=mul,cond_attn_gamma=<gamma>,cond_attn_renorm=preserve_ratio` |
| random region | base | the same with `random_region=True` |
| boost background | base | `cond_attn_op=mul,cond_attn_gamma=0,cond_attn_gamma2=0.5,cond_attn_renorm=preserve_ratio` |
| localized negative | vlmcfg | `negative=region` |
| region attention + vanilla guidance | vlmcfg | the attention edit plus `alpha=<a>` |
| region attention + localized negative | vlmcfg | the attention edit plus `negative=region` |
| attention edit on the conditional (no region) | vlmcfg | `cond_attn_op=mul|add|abs,cond_attn_gamma=<gamma>`, optionally `cond_attn_renorm=cap_ratio,cond_attn_cap=0.2` |

## Guidance against reasoning (Table `cfg_vs_reasoning_phd`)

The reasoning row is the unguided model on `phd_sub10_ccs_rationale`; the other two rows
are `qwen2_5_vl_base` and `qwen2_5_vl_vlmcfg` on `phd_sub10_ccs`.
