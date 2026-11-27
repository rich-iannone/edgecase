// The nodes shaded as spheres, which `plot(shade=True)` draws alike in each backend: a radial
// gradient from a highlight up and to the left of the center, lighter there and darker toward
// the far side, with the color's own in between. Its stops mix in sRGB, as SVG mixes them, and
// the viewer's shader (`static/viewer.js`) follows the same numbers.
use super::css::{self, Rgba};
use pyo3::prelude::*;

/// Where the highlight is, as a share of the radius from the center, both right and down (so up
/// and to the left)
pub const HIGHLIGHT: f64 = -0.35;
/// The gradient's radius, as a multiple of the node's
pub const REACH: f64 = 1.5;
/// Where the node's own color is, from the highlight (0) to the gradient's end (1)
pub const MID: f64 = 0.45;
/// How far the highlight goes toward white, and the far side toward black
const LIGHTEN: f32 = 0.5;
const DARKEN: f32 = 0.35;

/// The highlight's color and the far side's for `color`, which keep its alpha
pub fn stops(color: Rgba) -> (Rgba, Rgba) {
    let toward = |to: f32, t: f32| {
        let mut c = color;
        c[..3].iter_mut().for_each(|v| *v += (to - *v) * t);
        c
    };
    (toward(1.0, LIGHTEN), toward(0.0, DARKEN))
}

/// The gradient's color at `t`, from the highlight (0) to its end (1) and past it
pub fn at(color: Rgba, (hi, lo): (Rgba, Rgba), t: f64) -> Rgba {
    let mix = |a: Rgba, b: Rgba, t: f32| [0, 1, 2, 3].map(|k| a[k] + (b[k] - a[k]) * t);
    let t = t.clamp(0.0, 1.0);
    if t < MID {
        mix(hi, color, (t / MID) as f32)
    } else {
        mix(color, lo, ((t - MID) / (1.0 - MID)) as f32)
    }
}

/// `color` as hex, with its alpha where it has one
fn hex([r, g, b, a]: Rgba) -> String {
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    let rgb = format!("#{:02x}{:02x}{:02x}", byte(r), byte(g), byte(b));
    if a < 1.0 {
        format!("{rgb}{:02x}", byte(a))
    } else {
        rgb
    }
}

/// The colors of the highlight and of the far side of a node of `color` shaded as a sphere, as
/// hex: the stops of the SVG's gradient on either side of `color`'s own.
#[pyfunction]
pub fn shade_stops(color: &str) -> PyResult<(String, String)> {
    let (hi, lo) = stops(css::parse(color)?);
    Ok((hex(hi), hex(lo)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_lighten_and_darken() {
        assert_eq!(
            shade_stops("#4f79a7").unwrap(),
            ("#a7bcd3".to_string(), "#334f6d".to_string())
        );
        assert_eq!(
            shade_stops("rgb(255 1 1 / 0.4)").unwrap(),
            ("#ff808066".to_string(), "#a6010166".to_string())
        );
    }

    #[test]
    fn gradient_runs_through_the_color() {
        let c = [0.2, 0.4, 0.6, 1.0];
        let s = stops(c);
        assert_eq!(at(c, s, 0.0), s.0);
        assert_eq!(at(c, s, MID), c);
        assert_eq!(at(c, s, 1.0), s.1);
        assert_eq!(at(c, s, 3.0), s.1);
    }
}
