use std::time::Instant;
use tokenizers::tokenizer::Tokenizer;
use tokenizers::pre_tokenizers::byte_level::{reset_byte_level_stats, print_byte_level_stats};
use std::env;
use std::fs;
// cargo run --example profiler -- <tokenizer_path> <text_path>
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = env::args().collect();

    let tokenizer_path = &args[1];
    let text_path = &args[2];

    println!("Loading tokenizer from: {}", tokenizer_path);
    let tokenizer = Tokenizer::from_file(tokenizer_path)?;

    println!("Reading text from: {}", text_path);
    let content = fs::read_to_string(text_path)?;
    let lines: Vec<&str> = content.lines().collect();

    println!("Profiling {} lines...", lines.len());

    let mut total_normalization = 0u128;
    let mut total_pre_tokenization = 0u128;
    let mut total_model = 0u128;
    let mut total_post_processing = 0u128;
    let mut total_end_to_end = 0u128;

    // Reset ByteLevel statistics before profiling
    reset_byte_level_stats();

    for &line in &lines {
        // End-to-end timing
        let start_e2e = Instant::now();
        let _ = tokenizer.encode(line, true)?;
        total_end_to_end += start_e2e.elapsed().as_nanos();

        // Manual pipeline timing
        // 1. Normalization
        let start_norm = Instant::now();
        let mut normalized = tokenizers::tokenizer::NormalizedString::from(line);
        if let Some(normalizer) = tokenizer.get_normalizer() {
            use tokenizers::tokenizer::Normalizer;
            normalizer.normalize(&mut normalized)?;
        }
        total_normalization += start_norm.elapsed().as_nanos();

        // 2. Pre-tokenization
        let start_pre = Instant::now();
        let mut pre_tokenized = tokenizers::tokenizer::PreTokenizedString::from(normalized);
        if let Some(pre_tok) = tokenizer.get_pre_tokenizer() {
            use tokenizers::tokenizer::PreTokenizer;
            pre_tok.pre_tokenize(&mut pre_tokenized)?;
        }
        total_pre_tokenization += start_pre.elapsed().as_nanos();

        // 3. Model
        let start_model = Instant::now();
        use tokenizers::tokenizer::Model;
        // We need to call it for each part of pre_tokenized.
        // Replicating internal do_tokenize logic roughly.
        let mut encodings = vec![];
        for (subseq, _offsets, _is_word) in pre_tokenized.get_splits(tokenizers::tokenizer::OffsetReferential::Normalized, tokenizers::tokenizer::OffsetType::Byte) {
             let tokens = tokenizer.get_model().tokenize(subseq)?;
             // We won't build full Encoding here to keep it simple and focus on model time
             encodings.push(tokens);
        }
        total_model += start_model.elapsed().as_nanos();

        // 4. Post-processing
        let start_full_no_post = Instant::now();
        let encoding_no_post = tokenizer.encode(line, false)?;
        let _time_no_post = start_full_no_post.elapsed().as_nanos();

        if let Some(post) = tokenizer.get_post_processor() {
             let start_actual_post = Instant::now();
             use tokenizers::tokenizer::PostProcessor;
             let _ = post.process(encoding_no_post, None, true)?;
             total_post_processing += start_actual_post.elapsed().as_nanos();
        }
    }

    let n = lines.len() as f64;
    println!("--- Profiling Results (Average per line) ---");
    println!("Normalization:      {:>10.2} ns", total_normalization as f64 / n);
    println!("Pre-tokenization:   {:>10.2} ns", total_pre_tokenization as f64 / n);
    println!("Model Tokenization: {:>10.2} ns", total_model as f64 / n);
    println!("Post-processing:    {:>10.2} ns", total_post_processing as f64 / n);
    println!("Total (Pipeline):   {:>10.2} ns", (total_normalization + total_pre_tokenization + total_model + total_post_processing) as f64 / n);
    println!("Total (End-to-End): {:>10.2} ns", total_end_to_end as f64 / n);

    // Print fine-grained ByteLevel statistics
    print_byte_level_stats();

    // Batch profiling
    println!("Profiling batch encode...");
    let start_batch = Instant::now();
    let _ = tokenizer.encode_batch(lines.clone(), true)?;
    let duration_batch = start_batch.elapsed().as_nanos();
    println!("Batch encode:       {:>10.2} ns (Total: {} ms)", duration_batch as f64 / n, duration_batch / 1_000_000);

    Ok(())
}
