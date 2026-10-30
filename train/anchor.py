"""
E30 (10-21): 옛 정책 붙잡기(앵커) — PPO 손실에 β·KL(π_옛 ‖ π)를 더함

- mode "rollout" (E30 C): 이번 미니배치의 rollout 상태 전부에서 KL
- mode "states"  (E30 D): 고정된 상태 묶음(npz, `train/e30_anchor_states.py`)에서 미니배치마다 batch_size개를 뽑아 KL
- 옛 정책은 고정 (학습 안 함). β = 0 또는 앵커를 안 주면 train_ppo는 원래 PPO 클래스를 그대로 씀 → 예전과 같음
- train()은 SB3 2.9.0 PPO.train을 그대로 옮기고 'loss +=' 한 줄과 기록 한 줄만 더함 (버전이 다르면 멈춤)
"""

import numpy as np
import stable_baselines3
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

assert stable_baselines3.__version__ == "2.9.0", "anchor.py의 train()은 SB3 2.9.0 PPO.train을 옮긴 것 — 버전이 바뀌면 다시 옮길 것"


def kl_old_new(old_policy, policy, obs):
    """KL(옛 ‖ 지금), 상태마다 계산해 평균. 옛 정책 쪽은 기울기 없음"""
    with th.no_grad():
        p_old = old_policy.get_distribution(obs).distribution.probs
    logp_new = th.log_softmax(policy.get_distribution(obs).distribution.logits, dim=-1)
    return (p_old * (th.log(p_old + 1e-8) - logp_new)).sum(-1).mean()


class AnchorPPO(PPO):
    def setup_anchor(self, old_path, coef, mode, states_path=None, seed=0):
        self.anchor_coef = float(coef)
        self.anchor_mode = mode
        self.anchor_old = PPO.load(old_path, device=self.device).policy
        self.anchor_old.set_training_mode(False)
        for p in self.anchor_old.parameters():
            p.requires_grad_(False)
        self.anchor_obs = None
        if mode == "states":
            d = np.load(states_path)
            obs = {"screen": d["screen"], "goal": d["goal"]}
            self.anchor_obs, _ = self.policy.obs_to_tensor(obs)
            self.anchor_n = len(d["screen"])
            # 상태 뽑기 난수는 학습 난수와 따로 (PPO 쪽 난수 흐름을 바꾸지 않게)
            self.anchor_gen = th.Generator().manual_seed(seed)

    def _anchor_batch(self, rollout_obs):
        if self.anchor_mode == "rollout":
            return rollout_obs
        idx = th.randint(0, self.anchor_n, (self.batch_size,), generator=self.anchor_gen)
        return {k: v[idx] for k, v in self.anchor_obs.items()}

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        anchor_kls = []

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                # --- E30 추가: 옛 정책 붙잡기 ---
                anchor_kl = kl_old_new(self.anchor_old, self.policy, self._anchor_batch(rollout_data.observations))
                anchor_kls.append(anchor_kl.item())
                loss = loss + self.anchor_coef * anchor_kl

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/anchor_kl", np.mean(anchor_kls))   # E30 추가
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
