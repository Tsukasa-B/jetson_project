# models/RAL/ — RA-L用モデル置き場

sim側リポジトリ `actuated-design-lab/porcaro_2026` で学習した最新チェックポイントを
ONNX に export して、ここに置く。

## ファイル名

論文名そのまま。`modelA.onnx` 〜 `modelE.onnx`。
複数seedを持ち込む場合は `modelE_seed1000.onnx` のようにサフィックスを付け、
`manifest.yaml` 側で `E_seed1000:` のようにキーを分ける。

## モデル定義（sim側 `analysis/eval/run_eval_matrix.py` が正）

| 名前 | agent | lookahead_horizon | use_frame_stacking | frame_stack_k | obs次元 |
|---|---|---|---|---|---|
| A | lstm | 0.1 | False | 1 | 15 |
| B | lstm | 0.5 | False | 1 | 35 |
| C | lstm | 1.0 | False | 1 | 60 |
| D | mlp  | 0.5 | False | 1 | 35 |
| E | mlp  | 0.5 | **True** | **5** | **175** |

## export で必ず守ること

1. **empirical normalization を ONNX の中に焼き込む**。IROS時の ONNX は先頭に
   `Sub` / `Div` が入っており、実機側では正規化を一切していない。同じ形にすること。
   （焼き込まないと実機側が無正規化のまま推論して静かに壊れる）
2. **LSTM は入力3・出力3**（`obs, h_in, c_in` → `actions, h_out, c_out`）。
   **MLP は入力1・出力1**（`obs` → `actions`）。`model_registry.py` がここを検査する。
3. **E は frame stack 済みの 175次元を obs 入力に取る**こと。
   sim の積み方は `torch.roll(hist, -1); hist[-1] = new; hist.reshape(-1)`
   ＝ **`[最古 … 最新]` で最新が末尾**。ゼロ埋めスタート。
   実機側 `model_registry.PolicyRunner.build_obs` が同じ規約で積む。
4. batch次元は 1 固定でよい（`[1, obs_dim]`）。

## 置いたあとの手順

```bash
# 1. models/manifest.yaml の RAL: セクションのコメントを外して該当モデルを登録
#    （checkpoint パスと seed も書いておくと後で論文に書ける）

# 2. 整合性チェック（manifest の宣言と ONNX の実体を突き合わせる）
python3 tools/check_models.py

# 3. 観測構築の検証（旧実装とのビット一致 + framestack順序）
python3 tools/parity_test.py

# 4. 実機なしのドライラン
python3 src/deploy_policy.py --model RAL/E --midi songs/test_single4_bpm60.mid --mock

# 5. ここまで全部通ってから実機
python3 src/deploy_policy.py --model RAL/E --midi songs/gmd_02_mid_bpm105.mid --trial 1
```

## sim との数値パリティ（E を実機で回す前に必須）

`hardware_validation_plan.md` の通り、**物理ロボットに触る前に**
同一の観測系列を sim経路 / 実機経路の両方に通して action が一致することを確認する。

```bash
# 実機側の obs/action 系列を保存
python3 src/deploy_policy.py --model RAL/E --midi songs/test_single4_bpm60.mid \
        --mock --dump_obs /tmp/real_path_E.npz
# → sim側で同じ obs 系列をポリシーに流し、action を突き合わせる
```
