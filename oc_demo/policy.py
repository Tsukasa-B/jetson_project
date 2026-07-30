"""ポリシー。ONNX実物と、実機/モデルが無いとき用のスクリプト版。

どちらも act(obs) -> action(3,) を返す。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from . import adapter


class Policy:
    kind = "base"
    obs_dim: int = 0
    lookahead_steps: int = 0
    is_rnn: bool = False

    def reset(self) -> None: ...
    def act(self, obs: np.ndarray, step: int) -> np.ndarray: ...


class OnnxPolicy(Policy):
    """models/*.onnx をそのまま読む。正規化はONNX内に焼き込み済み。"""

    kind = "onnx"

    def __init__(self, path: str | Path, providers: Optional[list] = None):
        import onnxruntime as ort  # 実機のみ必要

        self.path = str(path)
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        avail = ort.get_available_providers()
        providers = [p for p in providers if p in avail] or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(self.path, providers=providers)
        self.providers = providers

        self.inputs = {i.name: i for i in self.sess.get_inputs()}
        # obs入力の特定: 名前に obs を含むもの > 2次元のもの > 先頭
        obs_name = next((n for n in self.inputs if "obs" in n.lower()), None)
        if obs_name is None:
            obs_name = next((n for n in self.inputs if len(self.inputs[n].shape) == 2), None)
        if obs_name is None:
            obs_name = next(iter(self.inputs))
        self.obs_name = obs_name
        shape = self.inputs[obs_name].shape
        self.obs_dim = int(shape[-1])
        self.lookahead_steps = self.obs_dim - 10  # frame stacking なしの場合

        # ★frame stacking(論文E)は未対応。静かに間違うのを防ぐため明示的に落とす。
        #   obs = 10 + lookahead_steps なので、lookahead が 1.5秒(75step)を超えるのは
        #   スタック済みobsを1フレームとして誤解釈している証拠。
        if self.lookahead_steps < 0 or self.lookahead_steps > 75:
            raise ValueError(
                f"obs次元 {self.obs_dim} を lookahead={self.lookahead_steps}step と解釈しました。"
                "frame stacking 付きモデル（論文E）はこのデモでは未対応です。"
                "oc_demo/policy.py にリングバッファを実装してください。"
            )
        # h0/c0 があれば RNN
        self.rnn_inputs = [n for n in self.inputs if n != obs_name]
        self.is_rnn = len(self.rnn_inputs) >= 2
        self._rnn_state: dict[str, np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        self._rnn_state = {}
        for name in self.rnn_inputs:
            shape = [1 if (isinstance(d, str) or d is None) else int(d)
                     for d in self.inputs[name].shape]
            self._rnn_state[name] = np.zeros(shape, dtype=np.float32)

    def act(self, obs: np.ndarray, step: int) -> np.ndarray:
        feed = {self.obs_name: obs.reshape(1, -1).astype(np.float32)}
        feed.update(self._rnn_state)
        outs = self.sess.run(None, feed)
        action = np.asarray(outs[0]).reshape(-1)[:3]
        if self.is_rnn and len(outs) >= 1 + len(self.rnn_inputs):
            for i, name in enumerate(self.rnn_inputs):
                self._rnn_state[name] = np.asarray(outs[1 + i], dtype=np.float32)
        return action


class ScriptedPolicy(Policy):
    """モデルファイルが無いとき用。目標力から素直に圧力パルスを作る。

    「それらしく叩く」だけのもので、学習済みポリシーではない。
    画面には必ず SCRIPTED と表示すること。
    """

    kind = "scripted"

    # 打面に当たるまでの機構遅れを含めた補正。MockPlant と対で調整する。
    STRIKE_OFFSET = -3  # 打点ステップからこのぶんずらして振り出す
    PULSE_STEPS = 6  # 振り下ろしの圧力パルス長

    def __init__(self, target_force: np.ndarray, lookahead_steps: int = 25,
                 lead_steps: Optional[int] = None, jitter_steps: float = 1.1,
                 seed: int = 0):
        self.target = np.asarray(target_force, dtype=np.float64)
        self.lookahead_steps = int(lookahead_steps)
        self.obs_dim = 10 + self.lookahead_steps
        self.lead = self.STRIKE_OFFSET if lead_steps is None else int(lead_steps)
        self._rng = np.random.default_rng(seed)
        self._jitter_steps = float(jitter_steps)
        self.drive = self._build_drive()

    def _onsets(self) -> list[int]:
        """目標力の立ち上がりを打点ステップとみなす。"""
        peak = float(self.target.max()) or 1.0
        thr = 0.45 * peak
        above = self.target >= thr
        return [i for i in range(1, len(above)) if above[i] and not above[i - 1]]

    def _build_drive(self) -> np.ndarray:
        """各制御ステップの振り下ろし強度 u∈[0,1] を先に作っておく。"""
        drive = np.zeros(len(self.target))
        peak = float(self.target.max()) or 1.0
        for m in self._onsets():
            amp = float(np.clip(self.target[m : m + 4].max() / peak, 0.2, 1.0))
            jitter = int(round(self._rng.normal(0.0, self._jitter_steps)))
            s0 = m + self.lead + jitter
            for k in range(self.PULSE_STEPS):
                i = s0 + k
                if 0 <= i < len(drive):
                    # 前半で立ち上げ、後半で緩める
                    shape = 1.0 if k < self.PULSE_STEPS - 2 else 0.45
                    drive[i] = max(drive[i], amp * shape)
        return drive

    def reset(self) -> None:
        pass

    def act(self, obs: np.ndarray, step: int) -> np.ndarray:
        u = float(self.drive[step]) if 0 <= step < len(self.drive) else 0.0
        # DF(伸展)とF(屈曲)を拮抗させる: u が大きいほど F 側に圧を入れる
        a_df = -1.0 + 2.0 * (0.35 - 0.25 * u)
        a_f = -1.0 + 2.0 * (0.10 + 0.80 * u)
        a_g = -1.0 + 2.0 * 0.45  # グリップは一定
        return np.array([a_df, a_f, a_g], dtype=np.float64)


def discover_models(models_dir: str | Path) -> list[dict]:
    """models/ 以下の .onnx を列挙する。manifest があればラベルを付ける。"""
    models_dir = Path(models_dir)
    manifest = adapter.load_manifest(models_dir)
    found: list[dict] = []
    if not models_dir.is_dir():
        return found
    for p in sorted(models_dir.rglob("*.onnx")):
        rel = p.relative_to(models_dir).as_posix()
        entry = {"id": rel, "path": str(p), "label": p.stem}
        # manifest から lookahead / obs_dim を拾えれば拾う
        for _group, items in (manifest or {}).items():
            if isinstance(items, dict):
                for key, meta in items.items():
                    if isinstance(meta, dict) and Path(str(meta.get("file", ""))).name == p.name:
                        entry["label"] = str(meta.get("label", key))
                        entry["lookahead"] = meta.get("lookahead_horizon")
                        entry["obs_dim"] = meta.get("obs_dim")
        found.append(entry)
    return found
