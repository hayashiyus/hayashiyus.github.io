# 再現用入力と実行手順

原著リポジトリのcommit `4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`、Qwen3.5-0.8Bのrevision `2fc06364715b967f1860aea9cf38778875588b17` を使用する。編集ファイルのSHA-256は `e92a92aafd1f1068b4fd94aa0fbe081ba3ddea2047ee7bcac7161d861be299e7`。

`environment.md` は使用環境の記録であり、全ハードウェアで検証したlockfileではない。モデルとMATH-500全体は同梱しない。対照8文章は原著公開ログから抽出したもので、出典は `RikiyaT/lm_understand-logs/exp8_math/runs/p50c`。原著のライセンスを `UPSTREAM_LICENSE` に保存している。

## 入力準備

対応したTransformers・huggingface_hub・NumPyを用意し、リポジトリのルートで実行する。準備処理はGPU推論をしない。

```bash
python3 blog/repro/prepare.py --out experiments/data/repro_inputs --prepare-inputs
```

既存のモデルとベンチマークを利用する場合：

```bash
python3 blog/repro/prepare.py --out experiments/data/repro_inputs --skip-model --prepare-inputs --model-dir path/to/pinned-model --math500 path/to/math500.json
```

ダウンロード物は保存前にハッシュ照合する。既存の異なる内容は上書きしない。MATH-500は正規化JSONのハッシュで全500行を照合する。並べ替えは `numpy.default_rng(20261105).permutation(500)`。先頭16件は較正、次の32件は幾何評価、先頭8件は生成試運転に使う。**生成本番は全500件**であり、残り452件だけを評価する設定ではない。

生成する `math_prefixes_cal.json` と `math_prefixes_eval.json` は、辞書ではなくtokenizerのchat templateで整形した文字列のリスト。実験に使った16/32 prefixと、文字列が全件一致することを独立に確認済み。

## E0/E1/E3

```bash
WSP_DATA=experiments/data/repro_inputs
for WSP_STAGE in e0 e1 e1fp32 e3; do
  python3 experiments/wsp_geometry/wsp_runner.py --stage "$WSP_STAGE" --model "$WSP_DATA/model" --edit-npz "$WSP_DATA/runs/p50c/edit/edit.npz" --manifest "$WSP_DATA/runs/p50c/edit/manifest.json" --texts "$WSP_DATA/contrast_texts.json" --math-prefixes "$WSP_DATA/math_prefixes_eval.json" --output "experiments/results/wsp_geometry/$WSP_STAGE.jsonl"
done
python3 experiments/wsp_geometry/wsp_runner.py --stage e3 --model "$WSP_DATA/model" --edit-npz "$WSP_DATA/runs/p50c/edit/edit.npz" --manifest "$WSP_DATA/runs/p50c/edit/manifest.json" --texts "$WSP_DATA/contrast_texts.json" --math-prefixes "$WSP_DATA/math_prefixes_cal.json" --output experiments/results/wsp_geometry/e3_calibration.jsonl
```

モデルはローカルファイルのみ読み込む。既存モデルを再利用した場合、`--model`をその場所へ変更する。E1-FP32はBF16保存重みをFP32へ変換して検証する。全語彙logitsの保存を続けず、最終next-tokenの統計を集計する。

## E2

```bash
python3 experiments/wsp_controls/e2_controls.py --model "$WSP_DATA/model" --edit-npz "$WSP_DATA/runs/p50c/edit/edit.npz" --manifest "$WSP_DATA/runs/p50c/edit/manifest.json" --data "$WSP_DATA/math500_ordered.json" --output experiments/results/wsp_geometry/e2.json
```

これは合成反復のteacher-forced対照であり、自然な生成におけるgate必要性の検証ではない。全3対照が対象効果の較正に失敗した結果を保持する。

## 保存重みの編集と生成評価

原著の `apply_edit.py` に元の単一shard、編集ディレクトリ、出力shardの順で渡す。モデルのメタデータを別ディレクトリへ複製し、その中の重みshardを置き換える。適用スクリプトは編集前後の対象テンソルのハッシュを照合する。

```bash
cp -a "$WSP_DATA/model" "$WSP_DATA/model_edited"
python3 "$WSP_DATA/code/apply_edit.py" "$WSP_DATA/model/model.safetensors-00001-of-00001.safetensors" "$WSP_DATA/runs/p50c/edit" "$WSP_DATA/model_edited/model.safetensors-00001-of-00001.safetensors"
```

対応したvLLM環境と原著採点器の依存関係（math-verify 0.9.0等）を使う。共有環境の書換えは不要であり、独立環境を推奨する。

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 python3 experiments/wsp_generation/runner.py --model "$WSP_DATA/model" --arm base --data "$WSP_DATA/math500_ordered.json" --out experiments/results/wsp_geometry/generation_base --samples 4 --chunk-size 8 --scorer-dir "$WSP_DATA/code" --gpu-memory-utilization 0.20 --max-num-seqs 32
VLLM_USE_FLASHINFER_SAMPLER=0 python3 experiments/wsp_generation/runner.py --model "$WSP_DATA/model_edited" --arm edited --data "$WSP_DATA/math500_ordered.json" --out experiments/results/wsp_geometry/generation_edited --samples 4 --chunk-size 8 --scorer-dir "$WSP_DATA/code" --gpu-memory-utilization 0.20 --max-num-seqs 32
```

メモリ割合は検証機での値であり、実行直前に空き容量を確認して調整する。1ジョブずつ実行。完了chunkは再利用できる。`--max-seconds`はchunk間で確認するため、厳密な実時間上限にはプロセス単位の外部timeoutも使う。`--aggregate-only`で完了chunkを再集計できる。

生成成績の本番再測定・P-Astra E4は、図版改訂時点では未完了。ここに記載したコマンドの存在を、本番結果が取得済みという意味にはしない。
