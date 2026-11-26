// The colors that the encodings give: the numeric scales, mixed in OKLab, and the darker borders
// of the nodes. The arithmetic follows the Python renderer's operation for operation, so that
// each color comes out as the same hex digits
use super::geometry::{py_max, py_min};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

/// A color for the light theme and one for the dark theme, as hex
pub type Pair = (String, String);

/// A stop of a numeric scale: its position and its color in each theme
pub type Stop = (f64, String, String);

/// The red, green and blue of a hex color, as Python's `int(h[i : i + 2], 16)` reads them
fn channels(hex: &str) -> PyResult<[u8; 3]> {
    let h = hex.trim_start_matches('#');
    let mut out = [0; 3];
    for (k, c) in out.iter_mut().enumerate() {
        *c = h
            .get(2 * k..2 * k + 2)
            .and_then(|digits| u8::from_str_radix(digits, 16).ok())
            .ok_or_else(|| PyValueError::new_err(format!("invalid hex color: {hex:?}")))?;
    }
    Ok(out)
}

fn to_hex(rgb: [i64; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
}

fn srgb_to_oklab(hex: &str) -> PyResult<[f64; 3]> {
    let [r, g, b] = channels(hex)?.map(|v| {
        let c = v as f64 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    });
    let l_ = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b).powf(1.0 / 3.0);
    let m_ = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b).powf(1.0 / 3.0);
    let s_ = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b).powf(1.0 / 3.0);
    Ok([
        0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
        1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
        0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
    ])
}

fn oklab_to_srgb([l, a, b]: [f64; 3]) -> String {
    let l_ = (l + 0.3963377774 * a + 0.2158037573 * b).powf(3.0);
    let m_ = (l - 0.1055613458 * a - 0.0638541728 * b).powf(3.0);
    let s_ = (l - 0.0894841775 * a - 1.2914855480 * b).powf(3.0);
    let rgb = [
        4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_,
        -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_,
        -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_,
    ];
    to_hex(rgb.map(|c| {
        let c = py_min(py_max(c, 0.0), 1.0);
        let c = if c <= 0.0031308 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (c * 255.0).round_ties_even() as i64
    }))
}

/// The color `t` of the way from hex `c1` to hex `c2`, mixed in OKLab
pub fn mix(c1: &str, c2: &str, t: f64) -> PyResult<String> {
    let (a, b) = (srgb_to_oklab(c1)?, srgb_to_oklab(c2)?);
    Ok(oklab_to_srgb([0, 1, 2].map(|k| a[k] + (b[k] - a[k]) * t)))
}

/// Hex `color` darkened by `factor`
pub fn darken(color: &str, factor: f64) -> PyResult<String> {
    Ok(to_hex(
        channels(color)?.map(|c| (c as f64 * (1.0 - factor)) as i64),
    ))
}

/// A numeric color scale, which keeps the colors it has mixed: there are at most a few hundred,
/// one for each hundredth of the scale
pub struct Scale {
    stops: Vec<Stop>,
    mixed: HashMap<u64, Pair>,
}

impl Scale {
    pub fn new(stops: Vec<Stop>) -> Self {
        Scale {
            stops,
            mixed: HashMap::new(),
        }
    }

    /// The (light, dark) colors at position `t`, in steps of a hundredth of the scale
    pub fn at(&mut self, t: f64) -> PyResult<Pair> {
        let (lo, hi) = (self.stops[0].0, self.stops[self.stops.len() - 1].0);
        let t = (py_min(py_max(t, lo), hi) * 100.0).round_ties_even() / 100.0;
        if let Some(pair) = self.mixed.get(&t.to_bits()) {
            return Ok(pair.clone());
        }
        let mut pair = None;
        for w in self.stops.windows(2) {
            let ((p0, l0, d0), (p1, l1, d1)) = (&w[0], &w[1]);
            if t <= *p1 {
                let f = (t - p0) / (p1 - p0);
                pair = Some((mix(l0, l1, f)?, mix(d0, d1, f)?));
                break;
            }
        }
        let last = &self.stops[self.stops.len() - 1];
        let pair = pair.unwrap_or_else(|| (last.1.clone(), last.2.clone()));
        self.mixed.insert(t.to_bits(), pair.clone());
        Ok(pair)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stops(list: &[(f64, &str, &str)]) -> Vec<Stop> {
        list.iter()
            .map(|&(p, l, d)| (p, l.to_string(), d.to_string()))
            .collect()
    }

    #[test]
    fn colors_are_mixed_as_python_mixes_them() {
        // From the Python renderer's _mix(), _darken() and _scale_color()
        assert_eq!(mix("#4e79a7", "#ffffff", 0.7).unwrap(), "#c8d6e5");
        assert_eq!(mix("#4e79a7", "#fafafa", 0.7).unwrap(), "#c5d2e2");
        assert_eq!(mix("#f97316", "#0d1117", 0.7).unwrap(), "#4c2e23");
        assert_eq!(mix("#000000", "#ffffff", 0.5).unwrap(), "#636363");
        assert_eq!(darken("#4e79a7", 0.3).unwrap(), "#365474");
        assert_eq!(darken("#ffffff", 0.3).unwrap(), "#b2b2b2");
        let mut sequential = Scale::new(stops(&[
            (0.0, "#86b6ef", "#184f95"),
            (1.0, "#0d366b", "#cde2fb"),
        ]));
        let mut diverging = Scale::new(stops(&[
            (-1.0, "#1c5cab", "#5598e7"),
            (0.0, "#d3d1cb", "#4a4944"),
            (1.0, "#b42318", "#f0706a"),
        ]));
        for (t, light, dark) in [
            (0.0, "#86b6ef", "#184f95"),
            (0.333, "#5d8ac2", "#547fb7"),
            (1.0, "#0d366b", "#cde2fb"),
            (7.0, "#0d366b", "#cde2fb"),
        ] {
            assert_eq!(sequential.at(t).unwrap(), (light.into(), dark.into()));
        }
        for (t, light, dark) in [
            (-1.0, "#1c5cab", "#5598e7"),
            (-0.425, "#88a1c1", "#526a86"),
            (0.005, "#d3d1cb", "#4a4944"),
            // 1.5 and 2.5 hundredths both round to 2, ties going to even
            (0.015, "#d3cec7", "#4d4a45"),
            (0.025, "#d3cec7", "#4d4a45"),
            (0.66, "#c56859", "#b6655d"),
        ] {
            assert_eq!(diverging.at(t).unwrap(), (light.into(), dark.into()));
        }
    }

    #[test]
    fn a_bad_color_is_an_error() {
        Python::with_gil(|_| {
            assert!(darken("#12", 0.3).is_err());
            assert!(mix("#zz0000", "#ffffff", 0.5).is_err());
        });
    }
}
