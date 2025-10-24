#author: Vipul Ramtekkar

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from lerobot.policies.act.modeling_act import (
    ACT,
    ACTION,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_STATE,
)
from collections import deque
from torch import Tensor

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_ENV_STATE
from lerobot.policies.act_text.configuration_act_text import ACTTextConfig


class ACTText(ACT):
    """
    Language-guided ACT that:
      1) Encodes a natural-language instruction to a single vector
      2) Projects it to ACT dim
      3) Prepends it as a 1-D token before the latent token
      4) Grows the 1-D positional embedding bank by +1 to keep alignment
    """

    def __init__(self, config):
        super().__init__(config)
        if not getattr(config, "use_language", False):
            raise ValueError("ACTText requires config.use_language=True")

        # ---- Language encoder (frozen by default) ----
        self.lang_encoder = AutoModel.from_pretrained(config.language_model_name)
        if getattr(config, "freeze_language_encoder", True):
            for p in self.lang_encoder.parameters():
                p.requires_grad = False
        self.lang_proj = nn.Linear(self.lang_encoder.config.hidden_size, config.dim_model)
        self.language_pooling = getattr(config, "language_pooling", "cls")

        # ---- Grow 1-D positional embedding bank by +1 (reserve index 0 for language) ----
        # Baseline uses nn.Embedding for 1-D tokens and builds the list via .weight.unsqueeze(1).
        if hasattr(self, "encoder_1d_feature_pos_embed") and isinstance(
            self.encoder_1d_feature_pos_embed, nn.Embedding
        ):
            old = self.encoder_1d_feature_pos_embed
            new = nn.Embedding(old.num_embeddings + 1, old.embedding_dim)
            with torch.no_grad():
                nn.init.zeros_(new.weight)       # pos[0] (language) starts at zeros (or init as you like)
                new.weight[1:] = old.weight      # shift old: latent -> pos[1], robot -> pos[2], env -> pos[3]
            self.encoder_1d_feature_pos_embed = new

    # ---- pooling helper identical to intent in your snippet ----
    def _pool_lang(self, last_hidden, attn_mask):
        if str(self.language_pooling).lower() == "cls":
            return last_hidden[:, 0]  # (B, H_text)
        mask = attn_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom  # (B, H_text)
    
    def _get_language_batch(self, batch: dict):
        # Prefer flat keys (recommended)
        if "language_tokens" in batch and "language_attention_mask" in batch:
            return batch["language_tokens"], batch["language_attention_mask"]
        # Fallback to nested if someone kept the older processor
        obs = batch.get("observation", {})
        lang = obs.get("language", {})
        if "tokens" in lang and "attention_mask" in lang:
            return lang["tokens"], lang["attention_mask"]
        # Helpful error with next steps
        raise KeyError(f"{batch}"
            "Language tokens not found in batch. "
            "Ensure the act_text preprocessor is used and it injects "
            "`language_tokens` and `language_attention_mask` at top-level."
        )
    
    def forward(self, batch: dict[str, torch.Tensor]):
        # --------- (A) VAE path unchanged (copied from baseline) ---------
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        batch_size = batch[OBS_IMAGES][0].shape[0] if OBS_IMAGES in batch else batch[OBS_ENV_STATE].shape[0]

        if self.config.use_vae and ACTION in batch and self.training:
            # Build VAE encoder inputs exactly as baseline
            cls_embed = torch.repeat_interleave(self.vae_encoder_cls_embed.weight, repeats=batch_size, dim=0)
            cls_embed = cls_embed.unsqueeze(1)  # (B,1,D)

            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE]).unsqueeze(1)  # (B,1,D)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B,S,D)

            vae_encoder_input = [cls_embed, action_embed] if not self.config.robot_state_feature else [
                cls_embed, robot_state_embed, action_embed
            ]
            vae_encoder_input = torch.cat(vae_encoder_input, dim=1)  # (B, S+1/2, D)

            pos_embed = self.vae_encoder_pos_enc.clone().detach()     # (1, S+1/2, D)

            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat([cls_joint_is_pad, batch["action_is_pad"]], dim=1)

            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # (B, D)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # --------- (B) NEW: build language token ---------
        lang_ids, lang_mask = self._get_language_batch(batch)  # (B, L), (B, L)

        lang_out = self.lang_encoder(input_ids=lang_ids, attention_mask=lang_mask)
        lang_vec = self._pool_lang(lang_out.last_hidden_state, lang_mask)  # (B, H_text)
        lang_tok = self.lang_proj(lang_vec)  # (B, D)

        # --------- (C) Encoder inputs: reuse baseline order, just prepend language ---------
        # Build pos list explicitly to keep alignment: [lang, latent, (robot), (env)]
        pos_list = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))  # each item: (1, D)

        encoder_in_tokens = [lang_tok, self.encoder_latent_input_proj(latent_sample)]  # (B,D) each
        encoder_in_pos_embed = [pos_list[0], pos_list[1]]                              # (1,D) each

        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
            encoder_in_pos_embed.append(pos_list[2])

        if self.config.env_state_feature:
            # index 3 if robot_state is present; else index 2
            idx = 3 if self.config.robot_state_feature else 2
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))
            encoder_in_pos_embed.append(pos_list[idx])

        # Image features path is copied as-is from baseline
        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                cam_features = cam_features.permute(0, 2, 3, 1).reshape(cam_features.size(0), -1, cam_features.size(1))
                cam_pos_embed = cam_pos_embed.permute(0, 2, 3, 1).reshape(cam_pos_embed.size(0), -1, cam_pos_embed.size(1))
                # rearrange to (seq, B, D) later; here we just extend lists with (B,D) items
                for f, p in zip(cam_features.unbind(dim=1), cam_pos_embed.unbind(dim=1)):
                    encoder_in_tokens.append(f)
                    encoder_in_pos_embed.append(p.unsqueeze(0))  # match (1,D) shape convention

        # Stack to (Seq, B, D) exactly like baseline
        encoder_in_tokens = torch.stack(encoder_in_tokens, dim=0)         # (S_enc, B, D)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, dim=0)   # (S_enc, 1, D)

        # --------- (D) Transformer + decoder path unchanged ---------
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)

        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        decoder_out = decoder_out.transpose(0, 1)                 # (B, S, D)
        actions = self.action_head(decoder_out)                   # (B, S, action_dim)
        return actions, (mu, log_sigma_x2)




class ACTTextPolicy(PreTrainedPolicy):
    config_class = ACTTextConfig
    name = "act_text"

    def __init__(self, config: ACTTextConfig):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACTText(config)

        if config.temporal_ensemble_coeff is not None:
            # reuse same ensembler as baseline
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

        self.reset()

    def get_optim_params(self):
        # identical pattern to ACTPolicy
        return [
            {"params": [p for n, p in self.named_parameters()
                        if not n.startswith("model.backbone") and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters()
                        if n.startswith("model.backbone") and p.requires_grad],
             "lr": self.config.optimizer_lr_backbone},
        ]

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue.clear()

    @torch.no_grad()
    def select_action(self, batch: dict) -> Tensor:
        self.eval()
        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            return self.temporal_ensembler.update(actions)

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> Tensor:
        self.eval()
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]
        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict):
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (F.l1_loss(batch[ACTION], actions_hat, reduction="none")
                   * ~batch["action_is_pad"].unsqueeze(-1)).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            mean_kld = (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - log_sigma_x2_hat.exp())).sum(-1).mean()
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss
        return loss, loss_dict
