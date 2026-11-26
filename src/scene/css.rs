// The colors that SVG reads, read for the raster: hex, the named colors, `rgb()` and `hsl()`,
// with their alpha, as CSS Color 4 reads them
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A color as red, green, blue and alpha, from 0 to 1
pub type Rgba = [f32; 4];

/// `color`, as SVG reads it, or an error for one this doesn't read
pub fn parse(color: &str) -> PyResult<Rgba> {
    let s = color.trim().to_ascii_lowercase();
    let rgba = if let Some(hex) = s.strip_prefix('#') {
        hex_color(hex)
    } else if let Some((name, args)) = s.split_once('(') {
        args.strip_suffix(')')
            .and_then(|args| function(name.trim(), args))
    } else if s == "transparent" {
        Some([0.0; 4])
    } else {
        NAMES
            .binary_search_by(|(name, _)| name.cmp(&s.as_str()))
            .ok()
            .map(|k| {
                let v = NAMES[k].1;
                [v >> 16, v >> 8, v].map(|c| (c & 0xff) as f32 / 255.0)
            })
            .map(|[r, g, b]| [r, g, b, 1.0])
    };
    rgba.ok_or_else(|| {
        PyValueError::new_err(format!(
            "can't read the color '{color}' for a raster image; give it as hex, a CSS color \
             name, rgb() or hsl()"
        ))
    })
}

/// The hex digits of `#rgb`, `#rgba`, `#rrggbb` or `#rrggbbaa`
fn hex_color(hex: &str) -> Option<Rgba> {
    if !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let value = |k: usize, n: usize| u8::from_str_radix(&hex[k..k + n], 16).unwrap();
    let v = match hex.len() {
        3 | 4 => [0, 1, 2, 3].map(|k| if k < hex.len() { value(k, 1) * 17 } else { 255 }),
        6 | 8 => [0, 2, 4, 6].map(|k| if k < hex.len() { value(k, 2) } else { 255 }),
        _ => return None,
    };
    Some(v.map(|c| c as f32 / 255.0))
}

/// A number, finite
fn number(s: &str) -> Option<f32> {
    s.parse::<f32>().ok().filter(|v| v.is_finite())
}

/// A number or a percentage, as a share of `whole`
fn share(s: &str, whole: f32) -> Option<f32> {
    match s.strip_suffix('%') {
        Some(p) => number(p).map(|v| v / 100.0),
        None => number(s).map(|v| v / whole),
    }
}

/// An angle, in degrees
fn hue(s: &str) -> Option<f32> {
    for (unit, degrees) in [
        ("deg", 1.0),
        ("grad", 0.9),
        ("rad", 180.0 / std::f32::consts::PI),
        ("turn", 360.0),
    ] {
        if let Some(v) = s.strip_suffix(unit) {
            return number(v).map(|v| v * degrees);
        }
    }
    number(s)
}

/// `rgb()`, `rgba()`, `hsl()` or `hsla()` of `args`, in the legacy syntax (with commas) or the
/// modern one (with spaces, and the alpha after a slash)
fn function(name: &str, args: &str) -> Option<Rgba> {
    let (main, alpha) = match args.split_once('/') {
        Some((main, alpha)) => (main, Some(alpha.trim())),
        None => (args, None),
    };
    let parts: Vec<&str> = if main.contains(',') {
        main.split(',').map(str::trim).collect()
    } else {
        main.split_whitespace().collect()
    };
    let (parts, alpha) = match (parts.len(), alpha) {
        (3, alpha) => (&parts[..], alpha),
        (4, None) => (&parts[..3], Some(parts[3])),
        _ => return None,
    };
    let alpha = match alpha {
        Some(a) => share(a, 1.0)?,
        None => 1.0,
    };
    let [r, g, b] = match name {
        "rgb" | "rgba" => [
            share(parts[0], 255.0)?,
            share(parts[1], 255.0)?,
            share(parts[2], 255.0)?,
        ],
        "hsl" | "hsla" => {
            // Saturation and lightness as percentages, with or without the sign
            let pct = |s: &str| number(s.strip_suffix('%').unwrap_or(s)).map(|v| v / 100.0);
            let (h, s, l) = (hue(parts[0])?, pct(parts[1])?, pct(parts[2])?);
            let (s, l) = (s.clamp(0.0, 1.0), l.clamp(0.0, 1.0));
            // As CSS Color 4 converts it
            let f = |n: f32| {
                let k = (n + h.rem_euclid(360.0) / 30.0) % 12.0;
                l - s * l.min(1.0 - l) * (k - 3.0).min(9.0 - k).clamp(-1.0, 1.0)
            };
            [f(0.0), f(8.0), f(4.0)]
        }
        _ => return None,
    };
    Some([r, g, b, alpha].map(|c| c.clamp(0.0, 1.0)))
}

/// The named colors of CSS, in order, with their values as `0xrrggbb`
const NAMES: [(&str, u32); 148] = [
    ("aliceblue", 0xf0f8ff),
    ("antiquewhite", 0xfaebd7),
    ("aqua", 0x00ffff),
    ("aquamarine", 0x7fffd4),
    ("azure", 0xf0ffff),
    ("beige", 0xf5f5dc),
    ("bisque", 0xffe4c4),
    ("black", 0x000000),
    ("blanchedalmond", 0xffebcd),
    ("blue", 0x0000ff),
    ("blueviolet", 0x8a2be2),
    ("brown", 0xa52a2a),
    ("burlywood", 0xdeb887),
    ("cadetblue", 0x5f9ea0),
    ("chartreuse", 0x7fff00),
    ("chocolate", 0xd2691e),
    ("coral", 0xff7f50),
    ("cornflowerblue", 0x6495ed),
    ("cornsilk", 0xfff8dc),
    ("crimson", 0xdc143c),
    ("cyan", 0x00ffff),
    ("darkblue", 0x00008b),
    ("darkcyan", 0x008b8b),
    ("darkgoldenrod", 0xb8860b),
    ("darkgray", 0xa9a9a9),
    ("darkgreen", 0x006400),
    ("darkgrey", 0xa9a9a9),
    ("darkkhaki", 0xbdb76b),
    ("darkmagenta", 0x8b008b),
    ("darkolivegreen", 0x556b2f),
    ("darkorange", 0xff8c00),
    ("darkorchid", 0x9932cc),
    ("darkred", 0x8b0000),
    ("darksalmon", 0xe9967a),
    ("darkseagreen", 0x8fbc8f),
    ("darkslateblue", 0x483d8b),
    ("darkslategray", 0x2f4f4f),
    ("darkslategrey", 0x2f4f4f),
    ("darkturquoise", 0x00ced1),
    ("darkviolet", 0x9400d3),
    ("deeppink", 0xff1493),
    ("deepskyblue", 0x00bfff),
    ("dimgray", 0x696969),
    ("dimgrey", 0x696969),
    ("dodgerblue", 0x1e90ff),
    ("firebrick", 0xb22222),
    ("floralwhite", 0xfffaf0),
    ("forestgreen", 0x228b22),
    ("fuchsia", 0xff00ff),
    ("gainsboro", 0xdcdcdc),
    ("ghostwhite", 0xf8f8ff),
    ("gold", 0xffd700),
    ("goldenrod", 0xdaa520),
    ("gray", 0x808080),
    ("green", 0x008000),
    ("greenyellow", 0xadff2f),
    ("grey", 0x808080),
    ("honeydew", 0xf0fff0),
    ("hotpink", 0xff69b4),
    ("indianred", 0xcd5c5c),
    ("indigo", 0x4b0082),
    ("ivory", 0xfffff0),
    ("khaki", 0xf0e68c),
    ("lavender", 0xe6e6fa),
    ("lavenderblush", 0xfff0f5),
    ("lawngreen", 0x7cfc00),
    ("lemonchiffon", 0xfffacd),
    ("lightblue", 0xadd8e6),
    ("lightcoral", 0xf08080),
    ("lightcyan", 0xe0ffff),
    ("lightgoldenrodyellow", 0xfafad2),
    ("lightgray", 0xd3d3d3),
    ("lightgreen", 0x90ee90),
    ("lightgrey", 0xd3d3d3),
    ("lightpink", 0xffb6c1),
    ("lightsalmon", 0xffa07a),
    ("lightseagreen", 0x20b2aa),
    ("lightskyblue", 0x87cefa),
    ("lightslategray", 0x778899),
    ("lightslategrey", 0x778899),
    ("lightsteelblue", 0xb0c4de),
    ("lightyellow", 0xffffe0),
    ("lime", 0x00ff00),
    ("limegreen", 0x32cd32),
    ("linen", 0xfaf0e6),
    ("magenta", 0xff00ff),
    ("maroon", 0x800000),
    ("mediumaquamarine", 0x66cdaa),
    ("mediumblue", 0x0000cd),
    ("mediumorchid", 0xba55d3),
    ("mediumpurple", 0x9370db),
    ("mediumseagreen", 0x3cb371),
    ("mediumslateblue", 0x7b68ee),
    ("mediumspringgreen", 0x00fa9a),
    ("mediumturquoise", 0x48d1cc),
    ("mediumvioletred", 0xc71585),
    ("midnightblue", 0x191970),
    ("mintcream", 0xf5fffa),
    ("mistyrose", 0xffe4e1),
    ("moccasin", 0xffe4b5),
    ("navajowhite", 0xffdead),
    ("navy", 0x000080),
    ("oldlace", 0xfdf5e6),
    ("olive", 0x808000),
    ("olivedrab", 0x6b8e23),
    ("orange", 0xffa500),
    ("orangered", 0xff4500),
    ("orchid", 0xda70d6),
    ("palegoldenrod", 0xeee8aa),
    ("palegreen", 0x98fb98),
    ("paleturquoise", 0xafeeee),
    ("palevioletred", 0xdb7093),
    ("papayawhip", 0xffefd5),
    ("peachpuff", 0xffdab9),
    ("peru", 0xcd853f),
    ("pink", 0xffc0cb),
    ("plum", 0xdda0dd),
    ("powderblue", 0xb0e0e6),
    ("purple", 0x800080),
    ("rebeccapurple", 0x663399),
    ("red", 0xff0000),
    ("rosybrown", 0xbc8f8f),
    ("royalblue", 0x4169e1),
    ("saddlebrown", 0x8b4513),
    ("salmon", 0xfa8072),
    ("sandybrown", 0xf4a460),
    ("seagreen", 0x2e8b57),
    ("seashell", 0xfff5ee),
    ("sienna", 0xa0522d),
    ("silver", 0xc0c0c0),
    ("skyblue", 0x87ceeb),
    ("slateblue", 0x6a5acd),
    ("slategray", 0x708090),
    ("slategrey", 0x708090),
    ("snow", 0xfffafa),
    ("springgreen", 0x00ff7f),
    ("steelblue", 0x4682b4),
    ("tan", 0xd2b48c),
    ("teal", 0x008080),
    ("thistle", 0xd8bfd8),
    ("tomato", 0xff6347),
    ("turquoise", 0x40e0d0),
    ("violet", 0xee82ee),
    ("wheat", 0xf5deb3),
    ("white", 0xffffff),
    ("whitesmoke", 0xf5f5f5),
    ("yellow", 0xffff00),
    ("yellowgreen", 0x9acd32),
];

#[cfg(test)]
mod tests {
    use super::*;

    fn bytes(color: &str) -> [u8; 4] {
        parse(color)
            .unwrap()
            .map(|c| (c * 255.0 + 0.5).floor() as u8)
    }

    #[test]
    fn colors_are_read_as_css_reads_them() {
        for (color, want) in [
            ("#3366cc", [0x33, 0x66, 0xcc, 255]),
            ("#36C", [0x33, 0x66, 0xcc, 255]),
            ("#36c8", [0x33, 0x66, 0xcc, 0x88]),
            ("#3366cc80", [0x33, 0x66, 0xcc, 0x80]),
            ("black", [0, 0, 0, 255]),
            (" RebeccaPurple ", [0x66, 0x33, 0x99, 255]),
            ("lightgoldenrodyellow", [0xfa, 0xfa, 0xd2, 255]),
            ("transparent", [0, 0, 0, 0]),
            ("rgb(255, 0, 128)", [255, 0, 128, 255]),
            ("rgba(255,0,128,0.5)", [255, 0, 128, 128]),
            ("rgb(100% 0% 50% / 25%)", [255, 0, 128, 64]),
            ("rgb(300 -5 0)", [255, 0, 0, 255]),
            ("hsl(120, 100%, 50%)", [0, 255, 0, 255]),
            ("hsl(240deg 100% 25%)", [0, 0, 128, 255]),
            ("hsla(0.5turn, 100%, 50%, 0.5)", [0, 255, 255, 128]),
            ("hsl(-60 100 50)", [255, 0, 255, 255]),
            ("hsl(30, 0%, 60%)", [153, 153, 153, 255]),
        ] {
            assert_eq!(bytes(color), want, "{color}");
        }
        assert!(NAMES.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn a_bad_color_is_an_error() {
        Python::with_gil(|_| {
            for color in [
                "",
                "#",
                "#12345",
                "#ggg",
                "#+f+f+f",
                "#aé",
                "notacolor",
                "rgb(1, 2)",
                "rgb(1, 2, 3",
                "rgb(nan, 0, 0)",
                "rgb(1, 2, 3, 4, 5)",
                "cmyk(1, 2, 3)",
                "hsl(1x, 50%, 50%)",
            ] {
                assert!(parse(color).is_err(), "{color}");
            }
        });
    }
}
