use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let bpe = BPE::from_file("data/gpt2-vocab.json", "data/gpt2-merges.txt").build()?;
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
    tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
    tokenizer.save("data/gpt2-tokenizer.json", true)?;
    println!("Saved data/gpt2-tokenizer.json");
    Ok(())
}
