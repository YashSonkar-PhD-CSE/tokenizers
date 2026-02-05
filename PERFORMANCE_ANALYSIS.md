# ByteLevel Pre-tokenization Performance

Added timing checkpoints to see where the time goes in pre-tokenization.

## What changed

Modified `byte_level.rs` to track timing for three phases:
- Prefix space handling
- Regex splitting  
- Byte-level transformation

Modified `profiler.rs` to reset stats before running and print results after.

## Results (1000 lines from brown_corpus.txt)

Overall pipeline:
- Normalization: 1,783 ns (7.8%)
- Pre-tokenization: 15,615 ns (68.1%)
- Model Tokenization: 4,758 ns (20.7%)
- Post-processing: 781 ns (3.4%)

Inside pre-tokenization (per call):
- Prefix space handling: 24.66 ns (0.1%)
- Regex splitting: 8,856 ns (53.7%)
- Byte-level transform: 7,603 ns (46.1%)

## What's slow

Regex splitting takes 53.7% of pre-tokenization time. The GPT-2 regex pattern `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` has to match Unicode categories which means Unicode database lookups.

Byte transformation takes 46.1%. It converts each UTF-8 byte to a special Unicode character via `BYTES_CHAR` lookup, iterating over every byte.

Prefix space handling is basically free at 0.1%.

## Possible optimizations

- Cache or optimize the regex pattern matching
- SIMD vectorization for byte transformation loop
- Skip transformations for unused tokens

## Run it

```bash
cargo run --release --example profiler -- ../data/tokenizer.json ../data/brown_corpus.txt
```
