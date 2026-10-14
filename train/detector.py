"""
실패 감지기 (E26 트랙 A, 10-14) — "정책이 지금 목표를 못 하고 있다"를 사람 판단 없이 알아냄

기준 (E26 확정 설계, 초기값. 값은 사람이 미리 정함 = 미리 제공한 것)
- goal_steps: 같은 목표로 정책 걸음 300 넘게 도착 못 함
- stall: 정책 걸음 60 동안 새 칸 없음 (같은 목표 안에서)
- revisit: 같은 최종 목표 안에서 떠났던 맵으로 2번 이상 되돌아감 (예: 연구소 → 연두마을 → 연구소). 10-14: 다음 목표 기준 → 최종 목표 기준으로 바꿈 (new_leg)
- rule_cap (10-14 추가, 사용자 확인 항목): 규칙 상한으로 판이 끊김 — run_story가 직접 fired로 넣음
걸음 수에서 빼는 것: 전투(자동 전투가 처리한 걸음), 대화·이벤트(run_event), 규칙 걸음 (설계 보강 1)
→ 호출하는 쪽이 정책 걸음만 step()으로 알려주고, 맵 변화는 매 반복 on_map()으로 알려줌 (규칙 걸음 중 맵 변화도 셈)

감지하면 이유와 함께, 그 목표가 시작된 순간의 상태(경계 상태)를 돌려줌 → 트랙 A가 거기서 Go-Explore
"""

GOAL_STEPS, STALL, REVISITS = 300, 60, 2
# 감지로 쓰는 이유 (10-14 사용자 결정, 결과 보기 전): 걸음 상한 + 규칙 상한(rule_cap)만.
#   재방문·새 칸 없음은 기록만 (observed) — 보정 측정(시드 3000~, M2B 통합 실행 21판)에서 성공 판 5/6에서 울려서 (회복 가능한 머뭇거림)
TRIGGERS = ("goal_steps", "rule_cap")


class FailureDetector:
    def __init__(self, goal_steps=GOAL_STEPS, stall=STALL, revisits=REVISITS, triggers=TRIGGERS):
        self.limits = {"goal_steps": goal_steps, "stall": stall, "revisits": revisits}
        self.triggers = set(triggers)
        self.observed = []          # 감지로 안 쓰는 이유가 울린 기록 (이유, 목표, 걸음)
        self.goal = None
        self.fired = None

    def _hit(self, reason):
        if reason in self.triggers:
            self.fired = reason
        elif not any(o[0] == reason and o[1] == self.goal for o in self.observed[-5:]):
            self.observed.append((reason, self.goal, getattr(self, "steps", 0)))
        return self.fired

    def new_leg(self, target, cur_map):
        """최종 목표(스토리 목표 또는 회복하러 가는 센터)가 바뀔 때: 재방문 셈을 처음부터
        10-14 시험: 재방문을 '다음 목표 맵' 기준으로 세면, 무궁시티 ↔ 30번도로를 100번 넘게 오가도
        맵이 바뀔 때마다 다음 목표가 바뀌어(센터 ↔ 무궁시티) 셈이 0으로 돌아가 못 잡았음 → 최종 목표 기준으로"""
        self.leg = tuple(target)
        self.left, self.cur_map, self.revisits = set(), tuple(cur_map), 0

    def new_goal(self, goal, cur_map, pos, state_bytes):
        """정책 목표(다음 목표 맵)가 바뀔 때. 경계 상태를 기억하고 걸음·새 칸 셈을 처음부터 (재방문은 new_leg 기준)"""
        self.goal, self.start_map, self.start_pos, self.boundary = tuple(goal), tuple(cur_map), tuple(pos), state_bytes
        self.steps, self.since_new = 0, 0
        self.tiles = {tuple(pos)}
        if not hasattr(self, "leg"):
            self.new_leg(goal, cur_map)
        self.fired = None

    def on_map(self, cur_map):
        """매 반복 (규칙 걸음 중에도). 떠났던 맵으로 돌아오면 재방문"""
        if self.goal is None or self.fired:
            return self.fired
        cur_map = tuple(cur_map)
        if cur_map != self.cur_map:
            self.left.add(self.cur_map)
            if cur_map in self.left:
                self.revisits += 1
            self.cur_map = cur_map
        if self.revisits >= self.limits["revisits"]:
            self._hit("revisit")
        return self.fired

    def step(self, pos):
        """정책 걸음 하나 (전투·이벤트로 처리된 걸음은 부르지 않음)"""
        if self.goal is None or self.fired:
            return self.fired
        self.steps += 1
        pos = tuple(pos)
        if pos in self.tiles:
            self.since_new += 1
        else:
            self.tiles.add(pos)
            self.since_new = 0
        if self.steps > self.limits["goal_steps"]:
            self._hit("goal_steps")
        elif self.since_new >= self.limits["stall"]:
            self._hit("stall")
        return self.fired

    def report(self):
        return {"reason": self.fired, "goal": list(self.goal) if self.goal else None, "leg": list(getattr(self, "leg", ()) or []),
                "start_map": list(self.start_map),
                "start_pos": list(self.start_pos), "policy_steps": self.steps, "revisits": self.revisits,
                "tiles": len(self.tiles), "limits": self.limits, "triggers": sorted(self.triggers),
                "observed": [list(o[:1]) + [list(o[1]) if o[1] else None, o[2]] for o in self.observed[-20:]]}
