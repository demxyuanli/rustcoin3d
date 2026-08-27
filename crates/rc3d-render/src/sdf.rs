//! Coverage-mask to signed-distance field (Felzenszwalb & Huttenlocher 1D EDT).

const INF: f32 = 1.0e20;

/// Encode a coverage mask as an 8-bit SDF. `0.5` is the glyph edge; values `> 0.5` are inside.
/// `spread` is the distance (in pixels) mapped to the full `0..1` range.
pub fn coverage_to_sdf(mask: &[u8], width: u32, height: u32, spread: f32) -> Vec<u8> {
    let w = width.max(1) as usize;
    let h = height.max(1) as usize;
    let spread = spread.max(1.0);
    let pad = spread.ceil() as usize;
    let pw = w + pad * 2;
    let ph = h + pad * 2;
    let len = pw * ph;
    let mut to_inside = vec![INF; len];
    let mut to_outside = vec![INF; len];
    for y in 0..h {
        for x in 0..w {
            let on = mask[y * w + x] > 127;
            let i = (y + pad) * pw + (x + pad);
            if on {
                to_inside[i] = 0.0;
            } else {
                to_outside[i] = 0.0;
            }
        }
    }
    edt_2d(&mut to_inside, pw, ph);
    edt_2d(&mut to_outside, pw, ph);

    let mut out = vec![0u8; w * h];
    let inv_spread = 1.0 / spread;
    for y in 0..h {
        for x in 0..w {
            let i = (y + pad) * pw + (x + pad);
            let d_in = to_inside[i].sqrt();
            let d_out = to_outside[i].sqrt();
            let signed = d_out - d_in;
            let encoded = (0.5 + signed * inv_spread).clamp(0.0, 1.0);
            out[y * w + x] = (encoded * 255.0 + 0.5) as u8;
        }
    }
    out
}

fn edt_2d(grid: &mut [f32], w: usize, h: usize) {
    let n = w.max(h);
    let mut f = vec![0.0; n];
    let mut d = vec![0.0; n];
    let mut v = vec![0i32; n];
    let mut z = vec![0.0; n + 1];
    for y in 0..h {
        f[..w].copy_from_slice(&grid[y * w..y * w + w]);
        edt_1d(&f[..w], &mut d[..w], &mut v, &mut z);
        grid[y * w..y * w + w].copy_from_slice(&d[..w]);
    }
    for x in 0..w {
        for y in 0..h {
            f[y] = grid[y * w + x];
        }
        edt_1d(&f[..h], &mut d[..h], &mut v, &mut z);
        for y in 0..h {
            grid[y * w + x] = d[y];
        }
    }
}

fn edt_1d(f: &[f32], d: &mut [f32], v: &mut [i32], z: &mut [f32]) {
    let n = f.len();
    if n == 0 {
        return;
    }
    let mut k: i32 = 0;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    for q in 1..n {
        let mut s;
        loop {
            let p = v[k as usize] as usize;
            let qf = f[q] + (q * q) as f32;
            let pf = f[p] + (p * p) as f32;
            s = (qf - pf) / (2 * q as i32 - 2 * p as i32) as f32;
            if s > z[k as usize] {
                break;
            }
            k -= 1;
            if k < 0 {
                k = 0;
                break;
            }
        }
        k += 1;
        v[k as usize] = q as i32;
        z[k as usize] = s;
        z[k as usize + 1] = f32::INFINITY;
    }
    k = 0;
    for q in 0..n {
        while z[k as usize + 1] < q as f32 {
            k += 1;
        }
        let p = v[k as usize];
        let dq = q as i32 - p;
        d[q] = (dq * dq) as f32 + f[p as usize];
    }
}
