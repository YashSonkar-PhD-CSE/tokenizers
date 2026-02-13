use crate::processors::byte_level::bytes_char;
use crate::tokenizer::{NormalizedString, Normalizer, Result};
use crate::utils::macro_rules_attribute;
use ahash::AHashSet;
use std::sync::LazyLock;

#[derive(Clone, Debug)]
#[macro_rules_attribute(impl_serde_type!)]
pub struct ByteLevel;

static BYTES_CHAR: LazyLock<[char; 256]> = LazyLock::new(bytes_char);

impl Default for ByteLevel {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteLevel {
    pub fn new() -> Self {
        Self {}
    }

    pub fn alphabet() -> AHashSet<char> {
        BYTES_CHAR.iter().copied().collect()
    }
}

#[inline]
fn byte_to_transformation(byte: u8) -> (char, isize) {
    // Safety: `byte` is in `0..=255`, so indexing into the 256-entry lookup table is always valid.
    let mapped = unsafe { *BYTES_CHAR.get_unchecked(byte as usize) };
    // In UTF-8, continuation bytes are exactly `10xxxxxx`.
    let shift = isize::from((byte & 0b1100_0000) == 0b1000_0000);
    (mapped, shift)
}

#[inline]
fn extend_transformations_scalar(bytes: &[u8], transformations: &mut Vec<(char, isize)>) {
    transformations.extend(bytes.iter().copied().map(byte_to_transformation));
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn extend_transformations_avx2(bytes: &[u8], transformations: &mut Vec<(char, isize)>) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut i = 0usize;
    let len = bytes.len();

    let mask_c0 = _mm256_set1_epi8(0b1100_0000_u8 as i8);
    let mask_80 = _mm256_set1_epi8(0b1000_0000_u8 as i8);

    while i + 32 <= len {
        let ptr = bytes.as_ptr().add(i) as *const __m256i;
        let lane = _mm256_loadu_si256(ptr);
        let high_bits = _mm256_and_si256(lane, mask_c0);
        let continuation = _mm256_cmpeq_epi8(high_bits, mask_80);
        let continuation_mask = _mm256_movemask_epi8(continuation) as u32;

        for offset in 0..32 {
            let byte = *bytes.get_unchecked(i + offset);
            let mapped = *BYTES_CHAR.get_unchecked(byte as usize);
            let shift = ((continuation_mask >> offset) & 1) as isize;
            transformations.push((mapped, shift));
        }

        i += 32;
    }

    if i < len {
        extend_transformations_scalar(&bytes[i..], transformations);
    }
}

#[inline]
fn extend_transformations(bytes: &[u8], transformations: &mut Vec<(char, isize)>) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: guarded by runtime AVX2 feature detection.
            unsafe {
                extend_transformations_avx2(bytes, transformations);
            }
            return;
        }
    }

    extend_transformations_scalar(bytes, transformations);
}

impl Normalizer for ByteLevel {
    /// Strip the normalized string inplace
    fn normalize(&self, normalized: &mut NormalizedString) -> Result<()> {
        if !normalized.is_empty() {
            let s = normalized.get();
            let mut transformations: Vec<(char, isize)> = Vec::with_capacity(s.len());
            extend_transformations(s.as_bytes(), &mut transformations);
            normalized.transform(transformations, 0);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_byte_level_normalize() {
        let original = "Hello 我今天能为你做什么";
        let normalized = "HelloĠæĪĳä»Ĭå¤©èĥ½ä¸ºä½łåģļä»Ģä¹Ī";
        assert_ne!(original, normalized);
        let mut n = NormalizedString::from(original);
        let byte_level = ByteLevel::new();
        byte_level.normalize(&mut n).unwrap();
        assert_eq!(&n.get(), &normalized);
        assert_eq!(
            n,
            NormalizedString::new(
                original.to_string(),
                normalized.to_string(),
                vec![
                    (0, 1),
                    (1, 2),
                    (2, 3),
                    (3, 4),
                    (4, 5),
                    (5, 6),
                    (5, 6),
                    (6, 9),
                    (6, 9),
                    (6, 9),
                    (6, 9),
                    (6, 9),
                    (6, 9),
                    (9, 12),
                    (9, 12),
                    (9, 12),
                    (9, 12),
                    (9, 12),
                    (9, 12),
                    (12, 15),
                    (12, 15),
                    (12, 15),
                    (12, 15),
                    (12, 15),
                    (12, 15),
                    (15, 18),
                    (15, 18),
                    (15, 18),
                    (15, 18),
                    (15, 18),
                    (15, 18),
                    (18, 21),
                    (18, 21),
                    (18, 21),
                    (18, 21),
                    (18, 21),
                    (18, 21),
                    (21, 24),
                    (21, 24),
                    (21, 24),
                    (21, 24),
                    (21, 24),
                    (21, 24),
                    (24, 27),
                    (24, 27),
                    (24, 27),
                    (24, 27),
                    (24, 27),
                    (24, 27),
                    (27, 30),
                    (27, 30),
                    (27, 30),
                    (27, 30),
                    (27, 30),
                    (27, 30),
                    (30, 33),
                    (30, 33),
                    (30, 33),
                    (30, 33),
                    (30, 33),
                    (30, 33)
                ],
                0
            )
        );
        assert_eq!(
            n.alignments_original(),
            vec![
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 7),
                (7, 13),
                (7, 13),
                (7, 13),
                (13, 19),
                (13, 19),
                (13, 19),
                (19, 25),
                (19, 25),
                (19, 25),
                (25, 31),
                (25, 31),
                (25, 31),
                (31, 37),
                (31, 37),
                (31, 37),
                (37, 43),
                (37, 43),
                (37, 43),
                (43, 49),
                (43, 49),
                (43, 49),
                (49, 55),
                (49, 55),
                (49, 55),
                (55, 61),
                (55, 61),
                (55, 61)
            ]
        );
    }
}
