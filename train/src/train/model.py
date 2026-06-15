import lightning as L
from diffusers.pipelines import FluxPipeline, FluxFillPipeline
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model_state_dict
import os
import prodigyopt

from ..flux.transformer import tranformer_forward
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, encode_images_fill, prepare_text_input
import time

class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config

        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)

        self.transformer = self.flux_fill_pipe.transformer
        self.text_encoder = self.flux_fill_pipe.text_encoder
        self.text_encoder_2 = self.flux_fill_pipe.text_encoder_2
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()
        # Freeze the Flux pipeline
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.flux_fill_pipe.vae.requires_grad_(False).eval()
        self.use_offset_noise = use_offset_noise

        if use_offset_noise:
            print('[debug] use OFFSET NOISE.')

        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_config:
            # add_adapter registers peft_config['default'], required by save_lora
            self.transformer.add_adapter(LoraConfig(**lora_config))
            if lora_path:
                self._load_lora_state_dict(lora_path)
        else:
            # No config provided — fall back to diffusers loader (standard LoRA only)
            self.transformer.load_lora_adapter(lora_path)
        lora_layers = filter(lambda p: p.requires_grad, self.transformer.parameters())
        return list(lora_layers)

    def _load_lora_state_dict(self, lora_path: str, adapter_name: str = "default"):
        """Load MoE LoRA weights from a safetensors file into the already-injected adapter.

        The file uses flat keys (e.g. lora_A.weight) for standard LoRA layers and
        expert-namespaced keys (lora_A.expert_0.weight) for MoE layers.  We need to
        insert the adapter name into the flat keys so they match the in-memory
        ModuleDict layout (lora_A.<adapter_name>.weight).
        """
        from safetensors.torch import load_file
        raw = load_file(lora_path)
        # Remove the 'transformer.' prefix added by FluxFillPipeline.save_lora_weights
        raw = {k[len("transformer."):]: v for k, v in raw.items() if k.startswith("transformer.")}
        sd = {}
        flat_lora_modules = {"lora_A", "lora_B", "lora_route", "lora_embedding_A", "lora_embedding_B"}
        for k, v in raw.items():
            parts = k.split(".")
            # Flat key: ...lora_X.weight  (no adapter name between module and weight)
            # Insert adapter_name: ...lora_X.<adapter_name>.weight
            if len(parts) >= 2 and parts[-1] in ("weight", "bias") and parts[-2] in flat_lora_modules:
                k = ".".join(parts[:-1]) + f".{adapter_name}." + parts[-1]
            sd[k] = v
        missing, unexpected = self.transformer.load_state_dict(sd, strict=False)
        if unexpected:
            print(f"[LoRA load] {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
        if missing:
            print(f"[LoRA load] {len(missing)} missing keys (first 3): {missing[:3]}")

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        if self.model_config['use_sep']:
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # In RL mode only the routing networks are trained; expert weights stay frozen.
        # In supervised mode all LoRA parameters are trained.
        rl_config = getattr(self, "rl_config", None)
        if rl_config is not None and not rl_config.train_lora:
            self.trainable_params = [
                p for n, p in self.transformer.named_parameters()
                if "lora_route" in n
            ]
            route_param_count = sum(p.numel() for p in self.trainable_params)
            total_lora_count = sum(p.numel() for p in self.lora_layers)
            print(
                f"[RL] Training routing params only: "
                f"{route_param_count:,} / {total_lora_count:,} total LoRA params"
            )
        else:
            self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        rl_config = getattr(self, "rl_config", None)
        if rl_config is not None:
            step_loss = self.rl_training_step(batch, rl_config)
        else:
            step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    # ── RL helpers ────────────────────────────────────────────────────────────

    def _encode_batch(self, batch):
        """Encode text + images; return tensors shared across rollouts."""
        imgs = batch["image"]
        mask_imgs = batch["condition"]
        prompts = batch["description"]
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )
            x_0, x_cond, img_ids = encode_images_fill(
                self.flux_fill_pipe, imgs, mask_imgs,
                prompt_embeds.dtype, prompt_embeds.device,
            )
        return x_0, x_cond, img_ids, prompt_embeds, pooled_prompt_embeds, text_ids

    def _transformer_forward(self, x_t, x_cond, t, guidance, img_ids,
                             prompt_embeds, pooled_prompt_embeds, text_ids):
        return self.transformer(
            hidden_states=torch.cat((x_t, x_cond), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]

    def rl_training_step(self, batch, rl_config) -> torch.Tensor:
        """
        GRPO-style policy-gradient update for MoE routing networks.

        Pass 1 (greedy, with grad): supervised diffusion loss.

        Pass 2 (stochastic, no_grad × G): collect rewards and routing inputs.
          - capture_routing_inputs records the hidden-state input to every MoE
            layer on CPU (avoids holding large GPU tensors across G rollouts).
          - Rewards are accumulated as Python floats, then normalized via GRPO.

        Policy-gradient loss: computed outside the full-transformer graph using
          compute_routing_log_prob, which re-runs only the tiny routing linears
          (with their own lightweight checkpoint) to get a gradient w.r.t.
          lora_route.weight.  This avoids the G× memory overhead of keeping
          stochastic rollout checkpoint states alive for backward.

        CheckpointError is prevented by two layers of defence:
          1. transformer_flux.py create_custom_forward captures routing_mode in
             its closure at forward time, so backward recompute always replays
             the same mode regardless of global module state.
          2. set_routing_mode("greedy") before return keeps module state clean.
        """
        from .rl_utils import (
            set_routing_mode, set_routing_masks,
            capture_layer_seq_lens, generate_global_routing_masks,
            save_actions, grpo_advantages,
            capture_routing_inputs, capture_routing_logits, compute_aux_losses,
            compute_routing_log_prob_nograd,
            compute_routing_log_prob,
        )

        G = rl_config.num_rollouts
        total_loss = 0

        # ── Pass 1: supervised diffusion loss (greedy routing, with grad) ────────
        set_routing_mode(self.transformer, "greedy")
        routing_logits = {}
        layer_seq_lens = {}  # {layer_id: T} — captured once, reused across rollouts
        with capture_routing_logits(self.transformer, routing_logits), \
             capture_layer_seq_lens(self.transformer, layer_seq_lens):
            (
                supervised_loss,
                target,
                t,
                x_t,
                x_cond,
                guidance,
                pooled_prompt_embeds,
                prompt_embeds,
                text_ids,
                img_ids,
            ) = self.step(batch, return_hidden_states=True)

        total_loss += rl_config.supervised_coeff * supervised_loss

        if rl_config.z_coeff != 0 or rl_config.load_balancing_coeff != 0:
            z_loss, lb_loss = compute_aux_losses(self.transformer, routing_logits)
            total_loss += rl_config.z_coeff * z_loss + rl_config.load_balancing_coeff * lb_loss

        # ── Pass 2: G stochastic rollouts — all no_grad ───────────────────────
        if rl_config.rl_coeff != 0:
            set_routing_mode(self.transformer, "stochastic")

            rollout_rewards    = []   # Python floats, one per rollout
            rollout_inputs     = []   # CPU routing-layer inputs, one dict per rollout
            rollout_actions    = []   # sampled top-k indices, one dict per rollout
            rollout_masks      = []   # CPU bool [B,T] masks keyed by layer id
            rollout_log_probs_old = []  # log P(action|state) under rollout-time weights

            for g in range(G):
                # Generate one mask spanning all layers' token budgets, then split
                # per layer.  Each layer receives a [B, T_layer] slice drawn from
                # the same global pool, so some layers may get zero stochastic
                # tokens while others get more — natural cross-layer variation.
                g_per_layer_masks = generate_global_routing_masks(
                    layer_seq_lens, x_t.shape[0], rl_config.stochastic_ratio, self.device
                )
                set_routing_masks(self.transformer, g_per_layer_masks)

                g_inputs = {}
                with capture_routing_inputs(self.transformer, g_inputs):
                    with torch.no_grad():
                        pred = self.transformer(
                            hidden_states=torch.cat((x_t, x_cond), dim=2),
                            timestep=t,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=img_ids,
                            return_dict=False,
                        )[0]

                pred_loss = F.mse_loss(pred, target, reduction="mean")
                rollout_rewards.append((pred_loss - supervised_loss.detach()).item())

                saved = save_actions(self.transformer)
                g_actions = {lid: acts for lid, (acts, _, _) in saved.items()}
                g_masks   = {lid: mask for lid, (_, _, mask) in saved.items()}

                rollout_inputs.append(g_inputs)
                rollout_actions.append(g_actions)
                rollout_masks.append(g_masks)
                # Anchor old-policy log-prob at rollout-time weights so the PPO
                # ratio is non-trivial when weights have been updated (e.g. multi-
                # epoch PPO or gradient-accumulation across batches).
                rollout_log_probs_old.append(
                    compute_routing_log_prob_nograd(
                        self.transformer, g_inputs, g_actions, g_masks
                    )
                )

            # Normalize rewards across the G rollouts (zero-mean, unit-std)
            advantages = grpo_advantages(rollout_rewards)  # [G] float32 CPU tensor

            # ── Policy-gradient loss ──────────────────────────────────────────────
            # compute_routing_log_prob re-runs only the routing linears with grad
            # (via its own lightweight checkpoint) — never touches the block-level
            # transformer checkpoints, so no routing-mode conflict during backward.
            # Masks restrict gradients to the stochastic token positions only.
            # log_prob_old was captured at rollout time, so the PPO ratio reflects
            # true policy drift if weights have been updated since the rollout.
            rl_loss_terms = []
            for g in range(G):
                adv_g = advantages[g].to(device=self.device, dtype=self.dtype)

                log_prob_new = compute_routing_log_prob(
                    self.transformer, rollout_inputs[g], rollout_actions[g], rollout_masks[g]
                )
                log_prob_old = rollout_log_probs_old[g]

                ratio   = torch.exp(log_prob_new - log_prob_old)
                clipped = torch.clamp(ratio, 1.0 - rl_config.clip_eps, 1.0 + rl_config.clip_eps)
                rl_loss_terms.append(-torch.min(ratio * adv_g, clipped * adv_g))

            rl_loss = torch.stack(rl_loss_terms).mean()
            self.last_rl_loss = rl_loss.item()

            # Reset to greedy and clear per-layer routing masks.
            set_routing_mode(self.transformer, "greedy")
            set_routing_masks(self.transformer, None)
            
            total_loss += rl_config.rl_coeff * rl_loss

        return total_loss

    def step(self, batch, return_hidden_states=False):
        imgs = batch["image"]
        mask_imgs = batch["condition"]
        condition_types = batch["condition_type"]
        prompts = batch["description"]
        position_delta = batch["position_delta"][0]

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )

            x_0, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)

            if self.use_offset_noise:
                x_1 = x_1 + 0.1 * torch.randn(x_1.shape[0], 1, x_1.shape[2]).to(self.device).to(self.dtype)

            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

            target = x_1 - x_0

        # Forward pass
        transformer_out = self.transformer(
            hidden_states=torch.cat((x_t, x_cond), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]

        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, target, reduction="mean")
        self.last_t = t.mean().item()

        if not return_hidden_states:
            return loss
        else:
            return (
                loss,
                target,
                t,
                x_t,
                x_cond,
                guidance,
                pooled_prompt_embeds,
                prompt_embeds,
                text_ids,
                img_ids,
            )
