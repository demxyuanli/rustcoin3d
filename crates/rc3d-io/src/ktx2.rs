//! Decode `KHR_texture_basisu` KTX2 / Basis Universal payloads to RGBA8.

use basisu::{DecodeFlags, TargetFormat, Transcoder};

/// KTX 2.0 file identifier (`\xABKTX 20\xBB\r\n\x1A\n`).
const KTX2_IDENTIFIER: [u8; 12] = [
    0xAB, b'K', b'T', b'X', b' ', b'2', b'0', 0xBB, 0x0D, 0x0A, 0x1A, 0x0A,
];

pub struct DecodedKtx2 {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

pub fn looks_like_ktx2(bytes: &[u8]) -> bool {
    bytes.len() >= KTX2_IDENTIFIER.len() && bytes.starts_with(&KTX2_IDENTIFIER)
}

pub fn decode_to_rgba(bytes: &[u8]) -> Result<DecodedKtx2, String> {
    let tex = Transcoder::new(bytes).map_err(|e| format!("KTX2/Basis open failed: {e:?}"))?;
    if !tex.supports(TargetFormat::Rgba32) {
        return Err("KTX2 texture cannot transcode to RGBA32".into());
    }
    let info = tex
        .image_level_info(0)
        .map_err(|e| format!("KTX2 level 0 missing: {e:?}"))?;
    let pixels = tex
        .transcode(0, TargetFormat::Rgba32, DecodeFlags::NONE)
        .map_err(|e| format!("KTX2 transcode failed: {e:?}"))?;
    let width = info.width.max(1);
    let height = info.height.max(1);
    let rgba = crop_rgba32(&pixels, width, height, info.num_blocks_x);
    Ok(DecodedKtx2 {
        width,
        height,
        rgba,
    })
}

fn crop_rgba32(src: &[u8], width: u32, height: u32, num_blocks_x: u32) -> Vec<u8> {
    let expected = width as usize * height as usize * 4;
    if src.len() == expected {
        return src.to_vec();
    }
    let stride = if num_blocks_x > 0 {
        num_blocks_x as usize * 4
    } else {
        width as usize
    };
    let row_bytes = stride.saturating_mul(4);
    if row_bytes == 0 || src.len() < row_bytes {
        let mut out = vec![0u8; expected];
        let n = src.len().min(expected);
        out[..n].copy_from_slice(&src[..n]);
        return out;
    }
    let mut out = vec![0u8; expected];
    let copy_w = (width as usize).min(stride) * 4;
    for y in 0..height as usize {
        let src_off = y * row_bytes;
        let dst_off = y * width as usize * 4;
        if src_off + copy_w > src.len() || dst_off + copy_w > out.len() {
            break;
        }
        out[dst_off..dst_off + copy_w].copy_from_slice(&src[src_off..src_off + copy_w]);
    }
    out
}
