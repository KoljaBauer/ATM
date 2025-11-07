import numpy as np
from collections import deque
import robomimic.utils.tensor_utils as TensorUtils
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torchvision.transforms as T

from einops import rearrange, repeat
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()

from atm.model import *
from atm.model.track_patch_embed import TrackPatchEmbed
from atm.policy.vilt_modules.transformer_modules import *
from atm.policy.vilt_modules.rgb_modules import *
from atm.policy.vilt_modules.language_modules import *
from atm.policy.vilt_modules.extra_state_modules import ExtraModalityTokens
from atm.policy.vilt_modules.policy_head import *
from atm.utils.flow_utils import ImageUnNormalize, sample_double_grid, tracks_to_video

import hydra
import os
from omegaconf import DictConfig

###############################################################################
#
# A ViLT Policy
#
###############################################################################
import tempfile

from typing import List
from pathlib import Path

def consolidate_ckpt(checkpoint_path: str | Path):
    if not isinstance(checkpoint_path, Path):
        checkpoint_path = Path(checkpoint_path)

    # Create a temporary file that we are guaranteed to have write access to
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Consolidate distributed checkpoint into the temporary file
        torch.distributed.checkpoint.format_utils.dcp_to_torch_save(checkpoint_path, tmp_path)

        # Load the state dict from the temporary file
        state_dict = torch.load(tmp_path, map_location="cpu")["model"]
    except Exception as e:
        print(f"Checkpoint consolidation failed: {e}")
        state_dict = None
    finally:
        # Clean up the temporary file
        tmp_path.unlink(missing_ok=True)

    # Return the full consolidated state dict (nothing is written to disk)
    return state_dict

def get_sd(ckpt_dir: str, drop_keys: List[str] = [], tag: str = None):
    if tag is None:  # get latest ckpt
        tag = sorted(os.listdir(os.path.join(ckpt_dir, "checkpoints")), key=lambda x: int(x.split("-")[-1]))[-1]

    ckpt_path = os.path.join(ckpt_dir, "checkpoints", tag)
    ckpt_files = os.listdir(ckpt_path)
    print(f"Loading state dict from {ckpt_path} ...", flush=True)

    if "model.pt" in ckpt_files:
        sd = torch.load(os.path.join(ckpt_path, "model.pt"), map_location="cpu")
    elif "inference.pt" in ckpt_files:
        sd = torch.load(os.path.join(ckpt_path, "inference.pt"), map_location="cpu")
    else:
        sd = consolidate_ckpt(ckpt_path)  # consolidate distributed checkpoint into inference ckpt

    for drop_key in drop_keys:
        print(f"Dropping key {drop_key}", flush=True)
        sd = {k: v for k, v in sd.items() if not k.startswith(drop_key)}
    return sd


class BCViLTPolicy(nn.Module):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self, obs_cfg, img_encoder_cfg, language_encoder_cfg, extra_state_encoder_cfg, track_cfg,
                 spatial_transformer_cfg, temporal_transformer_cfg,
                 policy_head_cfg, load_path=None, use_traj_gen: bool = False, traj_gen_path: str = None, ae_dir: str = None, nfe: int=50):
        super().__init__()

        self._process_obs_shapes(**obs_cfg)

        # 1. encode image
        self._setup_image_encoder(**img_encoder_cfg)

        # 2. encode language (spatial)
        self.language_encoder_spatial = self._setup_language_encoder(output_size=self.spatial_embed_size, **language_encoder_cfg)

        # 3. Track Transformer module
        self.use_traj_gen = use_traj_gen
        if self.use_traj_gen:
            assert traj_gen_path is not None and ae_dir is not None, "When using trajectory generator, traj_gen_path and ae_dir must be provided."
            import sys
            sys.path.append("/export/home/ra48gaq/code/trajectory-ae/diffusion")
            from diffusion.model.trajectory_gen_text import TrajRFText
            # from diffusion.model.trajectory_gen import get_sd
            
            ### LOAD MODEL ###

            cfg_traj_gen = DictConfig(
                OmegaConf.load(os.path.join(traj_gen_path, ".hydra", "config.yaml")), flags={"allow_objects": True}
            )
            OmegaConf.resolve(cfg_traj_gen)
            cfg_traj_gen_model = cfg_traj_gen.model
            cfg_traj_gen_model.ae_ckpt_cfg.ae_ckpt = ae_dir
            cfg_traj_gen_model.model_ckpt_cfg = {}
            cfg_traj_gen_model.multi_view = True
            traj_gen = hydra.utils.instantiate(cfg_traj_gen_model)
            traj_gen.eval()

            ### LOAD CHECKPOINT ###
            sd = get_sd(traj_gen_path, drop_keys=[])
            traj_gen.load_state_dict(sd, strict=True)

            # Compile UNET forward for speed
            traj_gen.unet.forward = torch.compile(traj_gen.unet.forward, mode="reduce-overhead", dynamic=True)

            self.track = traj_gen

            self.num_track_ts = 1 # our latents completely compress temporal dim
            self.num_track_ids = 1 # we dont have explicit trajectories, just one latent
            self.num_track_patches_per_view = 16 * 16 # our latent dim is 16x16
            self.policy_num_track_ts = 16 * 16 # our latent dim is 16x16
            self.policy_num_track_ids = 8 # required to match input dim of policy head

            self.track_up_proj = nn.Linear(16, 128)
            self.nfe = nfe
        else:
            self._setup_track(**track_cfg)

        # 3. define spatial positional embeddings, modality embeddings, and spatial token for summary
        self._setup_spatial_positional_embeddings()

        # 4. define spatial transformer
        self._setup_spatial_transformer(**spatial_transformer_cfg)

        ### 5. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = self._setup_extra_state_encoder(extra_embedding_size=self.temporal_embed_size, **extra_state_encoder_cfg)

        # 6. encode language (temporal), this will also act as the TEMPORAL_TOKEN, i.e., CLS token for action prediction
        self.language_encoder_temporal = self._setup_language_encoder(output_size=self.temporal_embed_size, **language_encoder_cfg)

        # 7. define temporal transformer
        self._setup_temporal_transformer(**temporal_transformer_cfg)

        # 8. define policy head
        self._setup_policy_head(**policy_head_cfg)

        if load_path is not None:
            drop_keys = ['policy_head'] if self.use_traj_gen else []
            self.load(load_path, strict=(not self.use_traj_gen), drop_keys=drop_keys)
            if not self.use_traj_gen:
                self.track.load(f"{track_cfg.track_fn}/model_best.ckpt")

    def _process_obs_shapes(self, obs_shapes, num_views, extra_states, img_mean, img_std, max_seq_len):
        self.img_normalizer = T.Normalize(img_mean, img_std)
        self.img_unnormalizer = ImageUnNormalize(img_mean, img_std)
        self.obs_shapes = obs_shapes
        self.policy_num_track_ts = obs_shapes["tracks"][0]
        self.policy_num_track_ids = obs_shapes["tracks"][1]
        self.num_views = num_views
        self.extra_state_keys = extra_states
        self.max_seq_len = max_seq_len
        # define buffer queue for encoded latent features
        self.latent_queue = deque(maxlen=max_seq_len)
        self.track_obs_queue = deque(maxlen=max_seq_len)

    def _setup_image_encoder(self, network_name, patch_size, embed_size, no_patch_embed_bias):
        self.spatial_embed_size = embed_size
        self.image_encoders = []
        for _ in range(self.num_views):
            input_shape = self.obs_shapes["rgb"]
            self.image_encoders.append(eval(network_name)(input_shape=input_shape, patch_size=patch_size,
                                                          embed_size=self.spatial_embed_size,
                                                          no_patch_embed_bias=no_patch_embed_bias))
        self.image_encoders = nn.ModuleList(self.image_encoders)

        self.img_num_patches = sum([x.num_patches for x in self.image_encoders])

    def _setup_language_encoder(self, network_name, **language_encoder_kwargs):
        return eval(network_name)(**language_encoder_kwargs)

    def _setup_track(self, track_fn, policy_track_patch_size=None, use_zero_track=False):
        """
        track_fn: path to the track model
        policy_track_patch_size: The patch size of TrackPatchEmbedding in the policy, if None, it will be assigned the same patch size as TrackTransformer by default
        use_zero_track: whether to zero out the tracks (ie use only the image)
        """
        track_cfg = OmegaConf.load(f"{track_fn}/config.yaml")
        self.use_zero_track = use_zero_track

        track_cfg.model_cfg.load_path = f"{track_fn}/model_best.ckpt"
        track_cls = eval(track_cfg.model_name)
        self.track = track_cls(**track_cfg.model_cfg)
        # freeze
        self.track.eval()
        for param in self.track.parameters():
            param.requires_grad = False

        self.num_track_ids = self.track.num_track_ids
        self.num_track_ts = self.track.num_track_ts
        self.policy_track_patch_size = self.track.track_patch_size if policy_track_patch_size is None else policy_track_patch_size


        self.track_proj_encoder = TrackPatchEmbed(
            num_track_ts=self.policy_num_track_ts,
            num_track_ids=self.num_track_ids,
            patch_size=self.policy_track_patch_size,
            in_dim=2 + self.num_views,  # X, Y, one-hot view embedding
            embed_dim=self.spatial_embed_size)

        self.track_id_embed_dim = 16
        self.num_track_patches_per_view = self.track_proj_encoder.num_patches_per_track
        self.num_track_patches = self.num_track_patches_per_view * self.num_views

    def _setup_spatial_positional_embeddings(self):
        # setup positional embeddings
        spatial_token = nn.Parameter(torch.randn(1, 1, self.spatial_embed_size))  # SPATIAL_TOKEN
        img_patch_pos_embed = nn.Parameter(torch.randn(1, self.img_num_patches, self.spatial_embed_size))

        if not self.use_traj_gen:
            track_patch_pos_embed = nn.Parameter(torch.randn(1, self.num_track_patches, self.spatial_embed_size-self.track_id_embed_dim))
            self.register_parameter("track_patch_pos_embed", track_patch_pos_embed)

        modality_embed = nn.Parameter(
            torch.randn(1, len(self.image_encoders) + self.num_views + 1, self.spatial_embed_size)
        )  # IMG_PATCH_TOKENS + TRACK_PATCH_TOKENS + SENTENCE_TOKEN

        self.register_parameter("spatial_token", spatial_token)
        self.register_parameter("img_patch_pos_embed", img_patch_pos_embed)
        self.register_parameter("modality_embed", modality_embed)

        # for selecting modality embed
        modality_idx = []
        for i, encoder in enumerate(self.image_encoders):
            modality_idx += [i] * encoder.num_patches
        for i in range(self.num_views):
            modality_idx += [modality_idx[-1] + 1] * self.num_track_ids * self.num_track_patches_per_view  # for track embedding
        modality_idx += [modality_idx[-1] + 1]  # for sentence embedding
        self.modality_idx = torch.LongTensor(modality_idx)

    def _setup_extra_state_encoder(self, **extra_state_encoder_cfg):
        if len(self.extra_state_keys) == 0:
            return None
        else:
            return ExtraModalityTokens(
                use_joint=("joint_states" in self.extra_state_keys),
                use_gripper=("gripper_states" in self.extra_state_keys),
                use_ee=("ee_states" in self.extra_state_keys),
                **extra_state_encoder_cfg
            )

    def _setup_spatial_transformer(self, num_layers, num_heads, head_output_size, mlp_hidden_size, dropout,
                                   spatial_downsample, spatial_downsample_embed_size, use_language_token=True):
        self.spatial_transformer = TransformerDecoder(
            input_size=self.spatial_embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_output_size=head_output_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,
        )

        if spatial_downsample:
            self.temporal_embed_size = spatial_downsample_embed_size
            self.spatial_downsample = nn.Linear(self.spatial_embed_size, self.temporal_embed_size)
        else:
            self.temporal_embed_size = self.spatial_embed_size
            self.spatial_downsample = nn.Identity()

        self.spatial_transformer_use_text = use_language_token

    def _setup_temporal_transformer(self, num_layers, num_heads, head_output_size, mlp_hidden_size, dropout, use_language_token=True):
        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(input_size=self.temporal_embed_size)

        self.temporal_transformer = TransformerDecoder(
            input_size=self.temporal_embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_output_size=head_output_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,)
        self.temporal_transformer_use_text = use_language_token

        action_cls_token = nn.Parameter(torch.zeros(1, 1, self.temporal_embed_size))
        nn.init.normal_(action_cls_token, std=1e-6)
        self.register_parameter("action_cls_token", action_cls_token)

    def _setup_policy_head(self, network_name, **policy_head_kwargs):
        policy_head_kwargs["input_size"] \
            = self.temporal_embed_size + self.num_views * self.policy_num_track_ts * self.policy_num_track_ids * 2

        action_shape = policy_head_kwargs["output_size"]
        self.act_shape = action_shape
        self.out_shape = np.prod(action_shape)
        policy_head_kwargs["output_size"] = self.out_shape
        self.policy_head = eval(network_name)(**policy_head_kwargs)

    @torch.no_grad()
    def preprocess(self, obs, track, action):
        """
        Preprocess observations, according to an observation dictionary.
        Return the feature and state.
        """
        b, v, t, c, h, w = obs.shape

        action = action.reshape(b, t, self.out_shape)

        obs = self._preprocess_rgb(obs)

        return obs, track, action

    @torch.no_grad()
    def _preprocess_rgb(self, rgb):
        rgb = self.img_normalizer(rgb / 255.)
        return rgb

    def _get_view_one_hot(self, tr):
        """ tr: b, v, t, tl, n, d -> (b, v, t), tl n, d + v"""
        b, v, t, tl, n, d = tr.shape
        tr = rearrange(tr, "b v t tl n d -> (b t tl n) v d")
        one_hot = torch.eye(v, device=tr.device, dtype=tr.dtype)[None, :, :].repeat(tr.shape[0], 1, 1)
        tr_view = torch.cat([tr, one_hot], dim=-1)  # (b t tl n) v (d + v)
        tr_view = rearrange(tr_view, "(b t tl n) v c -> b v t tl n c", b=b, v=v, t=t, tl=tl, n=n, c=d + v)
        return tr_view

    def track_encode(self, track_obs, task_emb):
        """
        Args:
            track_obs: b v t tt_fs c h w
            task_emb: b e
        Returns: b v t track_len n 2
        """
        assert self.num_track_ids == 32
        b, v, t, *_ = track_obs.shape

        if self.use_zero_track:
            recon_tr = torch.zeros((b, v, t, self.num_track_ts, self.num_track_ids, 2), device=track_obs.device, dtype=track_obs.dtype)
        else:
            track_obs_to_pred = rearrange(track_obs, "b v t fs c h w -> (b v t) fs c h w")

            grid_points = sample_double_grid(4, device=track_obs.device, dtype=track_obs.dtype)
            grid_sampled_track = repeat(grid_points, "n d -> b v t tl n d", b=b, v=v, t=t, tl=self.num_track_ts)
            grid_sampled_track = rearrange(grid_sampled_track, "b v t tl n d -> (b v t) tl n d")

            expand_task_emb = repeat(task_emb, "b e -> b v t e", b=b, v=v, t=t)
            expand_task_emb = rearrange(expand_task_emb, "b v t e -> (b v t) e")
            with torch.no_grad():
                pred_tr, _ = self.track.reconstruct(track_obs_to_pred, grid_sampled_track, expand_task_emb, p_img=0)  # (b v t) tl n d
                recon_tr = rearrange(pred_tr, "(b v t) tl n d -> b v t tl n d", b=b, v=v, t=t)

        recon_tr = recon_tr[:, :, :, :self.policy_num_track_ts, :, :]  # truncate the track to a shorter one
        _recon_tr = recon_tr.clone()  # b v t tl n 2
        with torch.no_grad():
            tr_view = self._get_view_one_hot(recon_tr)  # b v t tl n c

        tr_view = rearrange(tr_view, "b v t tl n c -> (b v t) tl n c")
        tr = self.track_proj_encoder(tr_view)  # (b v t) track_patch_num n d
        tr = rearrange(tr, "(b v t) pn n d -> (b t n) (v pn) d", b=b, v=v, t=t, n=self.num_track_ids)  # (b t n) (v patch_num) d

        return tr, _recon_tr

    
    def track_encode_traj_gen(self, track_obs, task_str: list[str]):
        with torch.autocast(device_type=track_obs.device.type, dtype=torch.bfloat16):
            B, v, t = track_obs.shape[:3]
            # v = 1 # for now we only use one view (side view)
            sample_tensor = torch.randn(B * v * t, *self.track.val_shape).to(device=track_obs.device, dtype=track_obs.dtype)

            points_per_traj = 64

            grid_points = sample_double_grid(4, device=track_obs.device, dtype=track_obs.dtype)
            query_pos = repeat(grid_points, "n c -> b_new n c", b_new=B * v * t)

            track_conds = torch.zeros((B * v * t, 1, 5), device=track_obs.device, dtype=track_obs.dtype)

            # for now, we use only the side view and the last of the 10 frames
            track_obs = track_obs[:, :, :, 0, ...]  # b v t tt_fs c h w -> b v t c h w

            view_ids = torch.cat([torch.zeros((B, 1, t), device=track_obs.device, dtype=track_obs.dtype), 
                                    torch.ones((B, 1, t), device=track_obs.device, dtype=track_obs.dtype)], dim=1)  # b v t
            view_ids = rearrange(view_ids, "b v t -> (b v t)")

            track_obs = rearrange(track_obs, "b v t c h w -> (b v t) h w c")
            track_obs = ((track_obs / 255.0) - 0.5) * 2 # map to [-1, 1] range

            with torch.no_grad():
                txt_emb = self.track.text_embedder(task_str).to(device=track_obs.device)
                txt_emb = repeat(txt_emb, "b l e -> (b v t) l e", v=v, t=t)

                # call super().sample to avoid decoding, as we do not need it here
                track_embs = self.track.sample(sample_tensor, 
                                                points_per_traj=points_per_traj, 
                                                query_pos=query_pos, 
                                                track_conds=track_conds, 
                                                start_frame=track_obs, 
                                                txt_emb=txt_emb,
                                                sample_steps=self.nfe,
                                                decode_latent=False,
                                                view_id=view_ids)
            
            n = track_embs.shape[-2]

            reshaped_latent = rearrange(track_embs, "(b v t) n c -> b v t 1 n c", b=B, v=v, t=t, n=n)
            
            track_embs = self.track_up_proj(track_embs)
            track_embs = rearrange(track_embs, "(b v t) n c -> b t (v n) c", b=B, v=v, t=t)

        return track_embs, reshaped_latent


    def spatial_encode(self, obs, track_obs, task_emb, extra_states, return_recon=False, task_str=None):
        """
        Encode the images separately in the videos along the spatial axis.
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w, (0, 255)
            task_emb: b e
            extra_states: {k: b t n}
        Returns: out: (b t 2+num_extra c), recon_track: (b v t tl n 2)
        """
        # 1. encode image
        img_encoded = []
        for view_idx in range(self.num_views):
            img_encoded.append(
                rearrange(
                    TensorUtils.time_distributed(
                        obs[:, view_idx, ...], self.image_encoders[view_idx]
                    ),
                    "b t c h w -> b t (h w) c",
                )
            )  # (b, t, num_patches, c)

        img_encoded = torch.cat(img_encoded, -2)  # (b, t, 2*num_patches, c)
        img_encoded += self.img_patch_pos_embed.unsqueeze(0)  # (b, t, 2*num_patches, c)
        B, T = img_encoded.shape[:2]

        # 2. encode task_emb
        text_encoded = self.language_encoder_spatial(task_emb)  # (b, c)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (b, t, 1, c)

        # 3. encode track
        if self.use_traj_gen:
            assert task_str is not None, "When using trajectory generator, task_str must be provided."
            track_encoded, _recon_track = self.track_encode_traj_gen(track_obs, task_str)  # track_encoded: ((b t n), 2*patch_num, c)  _recon_track: (b, v, track_len, n, 2)
        else:
            track_encoded, _recon_track = self.track_encode(track_obs, task_emb)  # track_encoded: ((b t n), 2*patch_num, c)  _recon_track: (b, v, track_len, n, 2)

            # patch position embedding
            tr_feat, tr_id_emb = track_encoded[:, :, :-self.track_id_embed_dim], track_encoded[:, :, -self.track_id_embed_dim:]
            tr_feat += self.track_patch_pos_embed  # ((b t n), 2*patch_num, c)
            # track id embedding
            tr_id_emb[:, 1:, -self.track_id_embed_dim:] = tr_id_emb[:, :1, -self.track_id_embed_dim:]  # guarantee the permutation invariance
            track_encoded = torch.cat([tr_feat, tr_id_emb], dim=-1)
            track_encoded = rearrange(track_encoded, "(b t n) pn d -> b t (n pn) d", b=B, t=T)  # (b, t, 2*num_track*num_track_patch, c)

        # 3. concat img + track + text embs then add modality embeddings
        if self.spatial_transformer_use_text:
            img_track_text_encoded = torch.cat([img_encoded, track_encoded, text_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch + 1, c)
            img_track_text_encoded += self.modality_embed[None, :, self.modality_idx, :]
        else:
            img_track_text_encoded = torch.cat([img_encoded, track_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch, c)
            img_track_text_encoded += self.modality_embed[None, :, self.modality_idx[:-1], :]

        # 4. add spatial token
        spatial_token = self.spatial_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c)
        encoded = torch.cat([spatial_token, img_track_text_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch + 2, c)

        # 5. pass through transformer
        encoded = rearrange(encoded, "b t n c -> (b t) n c")  # (b*t, 2*num_img_patch + 2*num_track*num_track_patch + 2, c)
        out = self.spatial_transformer(encoded)
        out = out[:, 0]  # extract spatial token as summary at o_t
        out = self.spatial_downsample(out).view(B, T, 1, -1)  # (b, t, 1, c')

        # 6. encode extra states
        if self.extra_encoder is None:
            extra = None
        else:
            extra = self.extra_encoder(extra_states)  # (B, T, num_extra, c')

        # 7. encode language, treat it as action token
        text_encoded_ = self.language_encoder_temporal(task_emb)  # (b, c')
        text_encoded_ = text_encoded_.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (b, t, 1, c')
        action_cls_token = self.action_cls_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c')
        if self.temporal_transformer_use_text:
            out_seq = [action_cls_token, text_encoded_, out]
        else:
            out_seq = [action_cls_token, out]

        if self.extra_encoder is not None:
            out_seq.append(extra)
        output = torch.cat(out_seq, -2)  # (b, t, 2 or 3 + num_extra, c')

        if return_recon:
            output = (output, _recon_track)

        return output

    def temporal_encode(self, x):
        """
        Args:
            x: b, t, num_modality, c
        Returns:
        """
        pos_emb = self.temporal_position_encoding_fn(x)  # (t, c)
        x = x + pos_emb.unsqueeze(1)  # (b, t, 2+num_extra, c)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (b, t*num_modality, c)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)  # (b, t, num_modality, c)
        return x[:, :, 0]  # (b, t, c)

    def forward(self, obs, track_obs, track, task_emb, extra_states, task_str=None):
        """
        Return feature and info.
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2, not used for training, only preserved for unified interface
            extra_states: {k: b t e}
            task_str: List of task strings (length b)
        """
        x, recon_track = self.spatial_encode(obs, track_obs, task_emb, extra_states, return_recon=True, task_str=task_str)  # x: (b, t, 2+num_extra, c), recon_track: (b, v, t, tl, n, 2)
        x = self.temporal_encode(x)  # (b, t, c)
        recon_track = rearrange(recon_track, "b v t tl n d -> b t (v tl n d)")
        x = torch.cat([x, recon_track], dim=-1)  # (b, t, c + v*tl*n*2)
        dist = self.policy_head(x)  # only use the current timestep feature to predict action
        return dist

    def forward_loss(self, obs, track_obs, track, task_emb, extra_states, action, task_str=None):
        """
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2, not used for training, only preserved for unified interface
            task_emb: b emb_size
            action: b t act_dim
            task_str: List of task strings (length b)
        """
        obs, track, action = self.preprocess(obs, track, action)
        dist = self.forward(obs, track_obs, track, task_emb, extra_states, task_str)
        loss = self.policy_head.loss_fn(dist, action, reduction="mean")

        ret_dict = {
            "bc_loss": loss.sum().item(),
        }

        if not self.policy_head.deterministic:
            # pseudo loss
            sampled_action = dist.sample().detach()
            mse_loss = F.mse_loss(sampled_action, action)
            ret_dict["pseudo_sampled_action_mse_loss"] = mse_loss.sum().item()

        ret_dict["loss"] = ret_dict["bc_loss"]
        return loss.sum(), ret_dict

    def forward_vis(self, obs, track_obs, track, task_emb, extra_states, action, task_str=None):
        """
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2
            task_emb: b emb_size
        Returns:
        """
        _, track, _ = self.preprocess(obs, track, action)
        track = track[:, :, 0, :, :, :]  # (b, v, track_len, n, 2) use the track in the first timestep

        b, v, t, track_obs_t, c, h, w = track_obs.shape
        if t >= self.num_track_ts:
            track_obs = track_obs[:, :, :self.num_track_ts, ...]
            track = track[:, :, :self.num_track_ts, ...]
        else:
            last_obs = track_obs[:, :, -1:, ...]
            pad_obs = repeat(last_obs, "b v 1 track_obs_t c h w -> b v t track_obs_t c h w", t=self.num_track_ts-t)
            track_obs = torch.cat([track_obs, pad_obs], dim=2)
            last_track = track[:, :, -1:, ...]
            pad_track = repeat(last_track, "b v 1 n d -> b v tl n d", tl=self.num_track_ts-t)
            track = torch.cat([track, pad_track], dim=2)

        grid_points = sample_double_grid(4, device=track_obs.device, dtype=track_obs.dtype)
        grid_track = repeat(grid_points, "n d -> b v tl n d", b=b, v=v, tl=self.num_track_ts)

        all_ret_dict = {}
        for view in range(self.num_views):
            gt_track = track[:1, view]  # (1 tl n d)
            gt_track_vid = tracks_to_video(gt_track, img_size=h)
            combined_gt_track_vid = (track_obs[:1, view, 0, :, ...] * .25 + gt_track_vid * .75).cpu().numpy().astype(np.uint8)

            _, ret_dict = self.track.forward_vis(track_obs[:1, view, 0, :, ...], grid_track[:1, view], task_emb[:1], p_img=0)
            ret_dict["combined_track_vid"] = np.concatenate([combined_gt_track_vid, ret_dict["combined_track_vid"]], axis=-1)

            all_ret_dict = {k: all_ret_dict.get(k, []) + [v] for k, v in ret_dict.items()}

        for k, v in all_ret_dict.items():
            if k == "combined_image" or k == "combined_track_vid":
                all_ret_dict[k] = np.concatenate(v, axis=-2)  # concat on the height dimension
            else:
                all_ret_dict[k] = np.mean(v)
        return None, all_ret_dict

    def act(self, obs, task_emb, extra_states, task_str=None):
        """
        Args:
            obs: (b, v, h, w, c)
            task_emb: (b, em_dim)
            extra_states: {k: (b, state_dim,)}
        """
        self.eval()
        B = obs.shape[0]

        # expand time dimenstion
        obs = rearrange(obs, "b v h w c -> b v 1 c h w").copy()
        extra_states = {k: rearrange(v, "b e -> b 1 e") for k, v in extra_states.items()}

        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        obs = torch.Tensor(obs).to(device=device, dtype=dtype)
        task_emb = torch.Tensor(task_emb).to(device=device, dtype=dtype)
        extra_states = {k: torch.Tensor(v).to(device=device, dtype=dtype) for k, v in extra_states.items()}

        if (obs.shape[-2] != self.obs_shapes["rgb"][-2]) or (obs.shape[-1] != self.obs_shapes["rgb"][-1]):
            obs = rearrange(obs, "b v fs c h w -> (b v fs) c h w")
            obs = F.interpolate(obs, size=self.obs_shapes["rgb"][-2:], mode="bilinear", align_corners=False)
            obs = rearrange(obs, "(b v fs) c h w -> b v fs c h w", b=B, v=self.num_views)

        while len(self.track_obs_queue) < self.max_seq_len:
            self.track_obs_queue.append(torch.zeros_like(obs))
        self.track_obs_queue.append(obs.clone())
        track_obs = torch.cat(list(self.track_obs_queue), dim=2)  # b v fs c h w
        track_obs = rearrange(track_obs, "b v fs c h w -> b v 1 fs c h w")

        obs = self._preprocess_rgb(obs)

        with torch.no_grad():
            x, rec_tracks = self.spatial_encode(obs, track_obs, task_emb=task_emb, extra_states=extra_states, return_recon=True, task_str=task_str)  # x: (b, 1, 4, c), recon_track: (b, v, 1, tl, n, 2)
            self.latent_queue.append(x)
            x = torch.cat(list(self.latent_queue), dim=1)  # (b, t, 4, c)
            x = self.temporal_encode(x)  # (b, t, c)

            feat = torch.cat([x[:, -1], rearrange(rec_tracks[:, :, -1, :, :, :], "b v tl n d -> b (v tl n d)")], dim=-1)

            action = self.policy_head.get_action(feat)  # only use the current timestep feature to predict action
            action = action.detach().cpu()  # (b, act_dim)

        action = action.reshape(-1, *self.act_shape)
        action = torch.clamp(action, -1, 1)
        return action.float().cpu().numpy(), (None, rec_tracks[:, :, -1, :, :, :])  # (b, *act_shape)

    def reset(self):
        self.latent_queue.clear()
        self.track_obs_queue.clear()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, strict: bool = True, drop_keys: list[str] = []):
        sd = torch.load(path, map_location="cpu")
        for drop_key in drop_keys:
            print(f"Dropping key {drop_key}", flush=True)
            sd = {k: v for k, v in sd.items() if not k.startswith(drop_key)}
        self.load_state_dict(sd, strict=strict)

    def train(self, mode=True):
        super().train(mode)
        self.track.eval()

    def eval(self):
        super().eval()
        self.track.eval()
