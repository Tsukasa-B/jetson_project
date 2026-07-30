"""
model_registry.py — models/manifest.yaml に基づくポリシーの解決・検証・推論

設計方針
  * manifest.yaml の値を「正」とする。
  * 起動時に ONNX の実体（入力数 / obs次元 / 出力数）と突き合わせ、
    1つでも食い違ったら RuntimeError で停止する。
    → モデルを差し替えるたびに手でパラメータを直す必要がなくなり、
      かつ「間違ったまま動いてしまう」経路を塞ぐ。
  * frame stacking (モデルE) のリングバッファはここに閉じ込める。
    sim側 user0/porcaro_2026_env.py と同じ規約:
        obs_history = zeros(k, base_obs_dim)          # reset時ゼロ埋め
        obs_history = roll(obs_history, -1); obs_history[-1] = new_frame
        obs = obs_history.reshape(-1)                 # [最古 ... 最新]
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(REPO_ROOT, "models", "manifest.yaml")

# base_obs = q(2) + qd(2) + prev_action(3) + sin + cos + bpm = 10, + lookahead(L)
BASE_OBS_WITHOUT_LOOKAHEAD = 10
ACTION_DIM = 3


# =============================================================================
# 定義
# =============================================================================
@dataclass
class ModelSpec:
    key: str                 # "IROS/B" のような完全キー
    group: str               # "IROS" / "RAL"
    name: str                # "A".."E"
    path: str                # onnxの絶対パス
    arch: str                # "lstm" | "mlp"
    lookahead_horizon: float
    frame_stack: int
    control_dt: float
    p_max: float
    target_force: float
    label: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def lookahead_steps(self) -> int:
        return int(round(self.lookahead_horizon / self.control_dt))

    @property
    def base_obs_dim(self) -> int:
        return BASE_OBS_WITHOUT_LOOKAHEAD + self.lookahead_steps

    @property
    def obs_dim(self) -> int:
        return self.base_obs_dim * self.frame_stack

    @property
    def is_recurrent(self) -> bool:
        return self.arch == "lstm"

    def describe(self) -> str:
        return (
            f"{self.key} [{self.label or self.arch}]\n"
            f"    file            : {os.path.relpath(self.path, REPO_ROOT)}\n"
            f"    arch            : {self.arch}\n"
            f"    lookahead       : {self.lookahead_horizon}s = {self.lookahead_steps} steps\n"
            f"    frame_stack     : {self.frame_stack}\n"
            f"    base_obs_dim    : {self.base_obs_dim}\n"
            f"    obs_dim         : {self.obs_dim}"
        )


# =============================================================================
# manifest 読み込み
# =============================================================================
def load_manifest(path: str = MANIFEST_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_models(manifest: dict | None = None) -> list[str]:
    manifest = manifest or load_manifest()
    keys = []
    for group, entries in manifest.items():
        if group == "defaults" or not isinstance(entries, dict):
            continue
        for name in entries:
            keys.append(f"{group}/{name}")
    return keys


def resolve(model_key: str, manifest: dict | None = None) -> ModelSpec:
    """ "IROS/B" / "RAL/E" 形式のキーを ModelSpec に解決する。
        グループ省略時("B")は RAL -> IROS の順に探す。 """
    manifest = manifest or load_manifest()
    defaults = manifest.get("defaults", {}) or {}

    if "/" in model_key:
        group, name = model_key.split("/", 1)
    else:
        name = model_key
        group = None
        for g in ("RAL", "IROS"):
            if name in (manifest.get(g) or {}):
                group = g
                break
        if group is None:
            raise KeyError(
                f"モデル '{model_key}' が manifest に見つかりません。"
                f" 利用可能: {list_models(manifest)}"
            )

    entries = manifest.get(group) or {}
    if name not in entries:
        raise KeyError(
            f"モデル '{group}/{name}' が manifest に見つかりません。"
            f" 利用可能: {list_models(manifest)}"
        )
    e = dict(entries[name])

    required = ("file", "arch", "lookahead_horizon", "frame_stack")
    missing = [k for k in required if k not in e]
    if missing:
        raise KeyError(f"manifest の {group}/{name} に必須項目がありません: {missing}")
    if e["arch"] not in ("lstm", "mlp"):
        raise ValueError(f"{group}/{name}: arch は 'lstm' か 'mlp'。実際: {e['arch']}")

    path = e["file"]
    if not os.path.isabs(path):
        path = os.path.join(REPO_ROOT, "models", path)

    return ModelSpec(
        key=f"{group}/{name}",
        group=group,
        name=name,
        path=path,
        arch=e["arch"],
        lookahead_horizon=float(e["lookahead_horizon"]),
        frame_stack=int(e["frame_stack"]),
        control_dt=float(defaults.get("control_dt", 0.02)),
        p_max=float(defaults.get("p_max", 0.6)),
        target_force=float(defaults.get("target_force", 20.0)),
        label=e.get("label", ""),
        extra={k: v for k, v in e.items() if k not in required + ("label",)},
    )


# =============================================================================
# 実行器（ONNX検証 + framestack + LSTM隠れ状態）
# =============================================================================
class PolicyRunner:
    def __init__(self, spec: ModelSpec, providers: list[str] | None = None, verbose: bool = True):
        import onnxruntime as ort  # 遅延import（検証だけしたい場面で重くしない）

        self.spec = spec
        if not os.path.exists(spec.path):
            raise FileNotFoundError(f"ONNX が見つかりません: {spec.path}")

        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        available = ort.get_available_providers()
        providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(spec.path, providers=providers)

        self._validate()

        self.input_names = [i.name for i in self.session.get_inputs()]
        if spec.is_recurrent:
            hshape = [1 if isinstance(d, str) or d is None else d
                      for d in self.session.get_inputs()[1].shape]
            self.hidden_shape = hshape
        else:
            self.hidden_shape = None

        self.reset()

        if verbose:
            print(f"[Policy] {spec.describe()}")
            print(f"    providers       : {self.session.get_providers()}")
            if self.hidden_shape:
                print(f"    hidden_shape    : {self.hidden_shape}")
            print("[Policy] manifest と ONNX の整合性チェック: OK")

    # ---------------------------------------------------------------- 検証
    def _validate(self) -> None:
        """manifest の宣言と ONNX の実体を突き合わせる。不一致は即例外。"""
        spec = self.spec
        ins = self.session.get_inputs()
        outs = self.session.get_outputs()
        errs = []

        n_in_expected = 3 if spec.is_recurrent else 1
        if len(ins) != n_in_expected:
            errs.append(
                f"入力数が不一致: manifest arch='{spec.arch}' なら {n_in_expected} 個のはずが "
                f"ONNX は {len(ins)} 個 ({[i.name for i in ins]})。"
                f" arch の指定（lstm/mlp）が逆になっていませんか？"
            )

        obs_shape = ins[0].shape
        actual_obs = obs_shape[-1] if len(obs_shape) >= 1 else None
        if isinstance(actual_obs, int) and actual_obs != spec.obs_dim:
            hint = ""
            base = actual_obs
            if spec.frame_stack > 1 and actual_obs % spec.frame_stack == 0:
                base = actual_obs // spec.frame_stack
            guessed_L = base - BASE_OBS_WITHOUT_LOOKAHEAD
            if guessed_L > 0:
                hint = (f" ONNXの{actual_obs}次元から逆算すると "
                        f"lookahead_horizon={guessed_L * spec.control_dt:.2f}s "
                        f"(frame_stack={spec.frame_stack} 前提) が妥当。")
            errs.append(
                f"obs次元が不一致: manifest から計算した {spec.obs_dim} "
                f"(= (10 + {spec.lookahead_steps}) x {spec.frame_stack}) に対し "
                f"ONNX の入力は {actual_obs}。{hint}"
            )

        n_out_expected = 3 if spec.is_recurrent else 1
        if len(outs) != n_out_expected:
            errs.append(f"出力数が不一致: 期待 {n_out_expected}, 実際 {len(outs)}")

        act_shape = outs[0].shape
        actual_act = act_shape[-1] if len(act_shape) >= 1 else None
        if isinstance(actual_act, int) and actual_act != ACTION_DIM:
            errs.append(f"action次元が不一致: 期待 {ACTION_DIM}, 実際 {actual_act}")

        if errs:
            raise RuntimeError(
                f"[Policy] manifest と ONNX が矛盾しています ({spec.key} -> {spec.path})\n  - "
                + "\n  - ".join(errs)
                + "\n  models/manifest.yaml を修正してください。"
            )

    # ---------------------------------------------------------------- 状態
    def reset(self) -> None:
        """エピソード開始時に呼ぶ。sim の _reset_idx と同じくゼロ埋め。"""
        s = self.spec
        self.frame_buffer = np.zeros((s.frame_stack, s.base_obs_dim), dtype=np.float32)
        if s.is_recurrent:
            self.h = np.zeros(self.hidden_shape, dtype=np.float32)
            self.c = np.zeros(self.hidden_shape, dtype=np.float32)

    def build_obs(self, base_frame: np.ndarray) -> np.ndarray:
        """base_obs_dim の1フレームを積んで、モデルに入れる obs を返す。"""
        s = self.spec
        base_frame = np.asarray(base_frame, dtype=np.float32).reshape(-1)
        if base_frame.shape[0] != s.base_obs_dim:
            raise ValueError(
                f"base frame の次元が違います: 期待 {s.base_obs_dim}, 実際 {base_frame.shape[0]}"
            )
        if s.frame_stack == 1:
            return base_frame.reshape(1, -1)
        # sim: roll(-1) して末尾に最新を書く → [最古 ... 最新]
        self.frame_buffer = np.roll(self.frame_buffer, -1, axis=0)
        self.frame_buffer[-1] = base_frame
        return self.frame_buffer.reshape(1, -1)

    # ---------------------------------------------------------------- 推論
    def infer(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if self.spec.is_recurrent:
            feeds = {self.input_names[0]: obs,
                     self.input_names[1]: self.h,
                     self.input_names[2]: self.c}
            out = self.session.run(None, feeds)
            self.h, self.c = out[1], out[2]
        else:
            out = self.session.run(None, {self.input_names[0]: obs})
        return out[0].reshape(-1)

    def step(self, base_frame: np.ndarray) -> np.ndarray:
        """1フレーム入れて action(3) を返す。frame stack / 隠れ状態は内部で処理。"""
        return self.infer(self.build_obs(base_frame))

    # ---------------------------------------------------------------- 変換
    def action_to_pressure(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """action -> (clip後action, 圧力指令[MPa] x3)"""
        a = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
        p = (a + 1.0) / 2.0 * self.spec.p_max
        return a, p
