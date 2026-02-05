## Run Profiler

```bash
cd tokenizers
cargo run --release --example profiler -- ../data/tokenizer.json ../data/brown_corpus.txt
```
# Optimisations made
- Changed hashmap to static array, lower overhead