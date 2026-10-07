"""
체크포인트별 이동 지도: 학습이 진행되면서 "어디를 밟는지"가 어떻게 바뀌는지 한 장으로 보기

왜 필요한가
- TensorBoard의 visited_tiles/visited_maps는 "얼마나"만 알려줌
- watch.py는 한 모델의 한 판만 보여줌
- 이 스크립트는 여러 체크포인트를 같은 조건으로 돌려서, 밟은 칸을 맵별 히트맵으로 그리고
  한 장에 위(초반) → 아래(후반)로 쌓음
  → "마을 안만 돌던 에이전트가 29번도로로 넓어진다" 같은 변화가 그림으로 보임
  → 보상 설계별(v1, v2...) 그림을 나란히 두면 실험 비교 자료가 됨 (포트폴리오용)

그림 읽는 법
- 행 = 체크포인트 (위가 초반), 열 = 맵 (왼쪽부터 진행 순서)
- 색이 밝을수록 오래 머문 칸 (로그 스케일. 몇 번 밟은 칸도 보이게)
- 흰 칸 = 한 번도 안 밟은 칸 / 회색 칸 = 그 체크포인트에서 아예 안 들어간 맵

실행 (학습 중에도 돌릴 수 있음. 그동안 학습은 조금 느려짐)
- uv run python -m train.progress_map v1_explore
- 체크포인트 수/판 수 조절:  ... --max-ckpts 8 --episodes 3
결과: notes/img/<실험이름>_progress.png

한 번 계산한 체크포인트는 runs/progress_map/에 저장해두고 재사용함
→ 학습이 더 진행된 뒤 다시 돌리면 새 체크포인트만 계산함
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 창 없이 파일로만 그림
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.gold_env import GoldEnv
from envs.map_names import MAP_ORDER, map_name, map_size

# 윈도우 기본 한글 폰트. 없으면 맵 이름이 네모로 깨짐
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def list_checkpoints(model_dir, max_ckpts):
    """ppo_<스텝>_steps.zip들을 스텝 순으로 정렬 + final.zip. 너무 많으면 고르게 골라냄"""
    ckpts = []
    for p in model_dir.glob("ppo_*_steps.zip"):
        steps = int(re.search(r"ppo_(\d+)_steps", p.name).group(1))
        ckpts.append((steps, p))
    ckpts.sort()

    # 체크포인트가 20개면 다 돌리기엔 오래 걸림 → 처음/끝을 포함해서 고르게 max_ckpts개만
    if len(ckpts) > max_ckpts:
        idx = np.linspace(0, len(ckpts) - 1, max_ckpts).round().astype(int)
        ckpts = [ckpts[i] for i in sorted(set(idx))]

    labels = [(f"{s // 1000}k", p) for s, p in ckpts]
    final = model_dir / "final.zip"
    if final.exists():
        labels.append(("final", final))
    return labels


def collect(model_path, env, episodes):
    """모델로 episodes판을 동시에 돌려서 (맵그룹, 맵번호, x, y)별 밟은 횟수를 셈"""
    model = PPO.load(model_path, device="cpu")
    counts = Counter()
    obs = env.reset()
    finished = [False] * episodes

    # 환경마다 한 판씩만 셈. 끝난 환경은 자동으로 새 판을 시작하는데 그건 무시함
    while not all(finished):
        # 학습 때와 같은 확률적 행동 (판마다 결과가 달라서 여러 판을 합침)
        actions, _ = model.predict(obs)
        obs, _, dones, infos = env.step(actions)
        for i, (done, info) in enumerate(zip(dones, infos)):
            if finished[i]:
                continue
            counts[info["position"]] += 1
            if done:
                finished[i] = True
    return counts


def load_or_collect(label, path, env, episodes, cache_dir):
    # 같은 체크포인트를 다시 계산하지 않으려고 결과를 파일로 저장해둠
    cache = cache_dir / f"{path.stem}_ep{episodes}.json"
    # final.zip은 이어서 학습하면 내용이 바뀌니까 캐시 안 씀
    if cache.exists() and label != "final":
        rows = json.loads(cache.read_text())
        return Counter({tuple(r[:4]): r[4] for r in rows})

    print(f"  {label} 계산 중 ({episodes}판)...")
    counts = collect(path, env, episodes)
    cache.write_text(json.dumps([[*k, v] for k, v in counts.items()]))
    return counts


def draw(results, out_path, title):
    # 열 = 모든 체크포인트를 통틀어 한 번이라도 들어간 맵 (진행 순서대로)
    visited = {k[:2] for counts in results.values() for k in counts}
    maps = [m for m in MAP_ORDER if m in visited] + sorted(visited - set(MAP_ORDER))

    # 맵 크기: 표에 있으면 그걸 쓰고, 없으면 밟은 좌표 최댓값으로 추정
    sizes = {}
    for m in maps:
        xs = [k[2] for c in results.values() for k in c if k[:2] == m]
        ys = [k[3] for c in results.values() for k in c if k[:2] == m]
        w, h = map_size(m) or (0, 0)
        # 표 크기를 벗어난 좌표가 있으면 늘림 (문 밖 1칸 등)
        sizes[m] = (max(w, max(xs) + 1), max(h, max(ys) + 1))

    # 색 범위를 모든 칸에서 공통으로 맞춤 → 체크포인트끼리 밝기를 직접 비교할 수 있게
    vmax = np.log1p(max(v for c in results.values() for v in c.values()))

    n_rows, n_cols = len(results), len(maps)
    # 열 너비를 맵 가로 크기에 비례하게 → 29번도로처럼 긴 맵이 찌그러지지 않게
    widths = [sizes[m][0] for m in maps]
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(8, sum(widths) * 0.12), 1.6 * n_rows + 1),
        gridspec_kw={"width_ratios": widths},
        squeeze=False,
    )

    for r, (label, counts) in enumerate(results.items()):
        n_tiles = len(counts)
        n_maps = len({k[:2] for k in counts})
        for c, m in enumerate(maps):
            ax = axes[r][c]
            w, h = sizes[m]
            grid = np.zeros((h, w))
            for (g, n, x, y), v in counts.items():
                if (g, n) == m:
                    grid[y, x] += v

            if grid.sum() == 0:
                # 이 체크포인트에서 아예 안 들어간 맵
                ax.imshow(np.ones((h, w)), cmap="Greys", vmin=0, vmax=4)
            else:
                # 안 밟은 칸(0)은 마스킹해서 흰색으로
                shown = np.ma.masked_equal(np.log1p(grid), 0)
                cmap = plt.get_cmap("magma_r").copy()
                cmap.set_bad("white")
                ax.imshow(shown, cmap=cmap, vmin=0, vmax=vmax)

            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(map_name(m), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{label}\n칸 {n_tiles} / 맵 {n_maps}", fontsize=9, rotation=0, ha="right", va="center")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    print(f"저장: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="실험 이름 (models/<name>/)")
    parser.add_argument("--max-ckpts", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=2, help="체크포인트당 판 수")
    args = parser.parse_args()

    model_dir = Path("models") / args.name
    ckpts = list_checkpoints(model_dir, args.max_ckpts)
    if not ckpts:
        print(f"{model_dir}에 체크포인트가 없음")
        return

    cache_dir = Path("runs/progress_map") / args.name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 판 수만큼 환경을 동시에 돌림. 환경은 한 번만 만들고 체크포인트끼리 재사용
    # (만들 때마다 프로세스를 새로 띄우면 시간이 꽤 걸림)
    env = make_vec_env(GoldEnv, n_envs=args.episodes, vec_env_cls=SubprocVecEnv)
    results = {}
    for label, path in ckpts:
        results[label] = load_or_collect(label, path, env, args.episodes, cache_dir)
    env.close()

    title = f"{args.name}: 체크포인트별 이동 지도 (체크포인트당 {args.episodes}판 합계, 색 = 머문 스텝 수)"
    draw(results, Path("notes/img") / f"{args.name}_progress.png", title)


# SubprocVecEnv 때문에 윈도우에서 필요한 가드 (train_ppo.py와 같은 이유)
if __name__ == "__main__":
    main()
