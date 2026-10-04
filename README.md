# treeattn: sub-quadratic tree attention (clean rewrite)

Pure sub-quadratic code, no dense attention anywhere (benchmark against nanoGPT externally).

## Layout
- `treeattn/tree.py`      positional binary tree (index vectors = sums), causal beam descent, neighbour search, budget split, routing loss
- `treeattn/spike.py`     causal spike detector (trailing 64-key window) + graded amplification
- `treeattn/attention.py` the attention module: splitter (qkv), budget predictor, key selection, exact sparse attention, decode cache
- `treeattn/model.py`     standard FFN + blocks + GPT + cached `generate`
- `treeattn/tasks.py`     needle-in-a-haystack generator, `evaluate.py` recall metric
- `scripts/`              `train_needle.py`, `eval_needle.py`, `prepare_data.py`, `train_lm.py`
- `tests/test_core.py`    causality, tree coverage, decode == parallel, gradients to both trainable functions

## Kaggle
```
pip install -q -r requirements.txt
python -m pytest tests -q
python scripts/train_needle.py --lengths 512,1024,2048,4096 --steps 600 --eval_lengths 1024,4096,16384
python scripts/train_needle.py --out runs/needle_nospike --spike 0 --route_weight 0     # baseline for comparison
python scripts/eval_needle.py --ckpt runs/needle/ckpt.pt --lengths 4096,16384,65536
python scripts/prepare_data.py && python scripts/train_lm.py                             # LM stage
```

## Design decisions to know about
- Padding zeros go at the end (future side). A causal descent never reads those nodes, so this equals uniform padding.
- Causal tree = left siblings along the root-to-t path ("fractal" rule); training and inference are the same code, causal in both.
- Tree indexes only the content dims; `rot_dims` (default 16 of 64) carry rotary position and are used only in exact attention, so tree sums are position-free and lengths beyond training work.
- Graded amplification: mult = 1 + amp*d*min(relu(score/thr-1), cap). Threshold = running quantile of natural scores (training), frozen at inference. Only index sums are amplified.
- Anti-dilution global vector = exclusive prefix sum of the amplified index keys, fed to the budget predictor.
- Routing loss (needle task): at the question position, ancestors of the needle must outrank other past nodes at every level. Gradient flows into q, k and the amplification multiplier, which is how the splitter can learn to make important keys spiky.
- Neighbour search borrows each earlier query's best leaves and compares scores against that query's own score on the same group (`nb_level` 1 or 2 = groups of 2 or 4 keys). Left side only: in autoregressive use the right side is the future.

## Not done / honest limits
- Written without a GPU or torch available: syntax-checked only. Run the tests first.
- Selection is pure PyTorch (no Triton). Beam width is fixed at `wmax`; the budget limits keys read, not search compute. Adaptive per-query beam width is the next step for real compute savings.
- Training at 16k+ will be slow on a T4; start with the default curriculum.
