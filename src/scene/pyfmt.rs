// Numbers and text written as Python's f-strings write them, so that the SVG made here is the
// SVG that the Python renderer made, byte for byte
use std::fmt;

/// `f"{v:.1f}"`. Rust's `{:.1}` rounds the exact binary value, ties to even, as Python does
pub struct F1(pub f64);

/// `f"{v}"` (`repr()`): the shortest digits that read back as `v`, written out in full from
/// 1e-4 up to 1e16 and in exponent form outside that
pub struct Repr(pub f64);

/// `f"{v:g}"`: six significant digits, without trailing zeros
pub struct General(pub f64);

/// A size in pixels, as an `int` is written (a whole number), or else as a float is
pub struct Px(pub f64);

/// `html.escape(s)`: `&`, `<`, `>`, `"` and `'` as character references
pub struct Escaped<'a>(pub &'a str);

impl fmt::Display for F1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_nan() {
            f.write_str("nan")
        } else {
            write!(f, "{:.1}", self.0)
        }
    }
}

/// Non-finite values and zeros as Python writes them, or `None` for any other value
fn special(v: f64, zero: &'static str, neg_zero: &'static str) -> Option<&'static str> {
    if v.is_nan() {
        Some("nan")
    } else if v.is_infinite() {
        Some(if v > 0.0 { "inf" } else { "-inf" })
    } else if v == 0.0 {
        Some(if v.is_sign_negative() { neg_zero } else { zero })
    } else {
        None
    }
}

/// `v`'s sign, digits and decimal exponent, from Rust's `{:e}` (with `precision`, rounded to
/// that many digits after the first, or else the shortest digits that read back as `v`)
fn digits(v: f64, precision: Option<usize>) -> (bool, String, i32) {
    let text = match precision {
        Some(p) => format!("{:.*e}", p, v.abs()),
        None => format!("{:e}", v.abs()),
    };
    let (mantissa, exp) = text.split_once('e').expect("{:e} writes an exponent");
    let digits = mantissa.chars().filter(|c| *c != '.').collect();
    (
        v < 0.0,
        digits,
        exp.parse().expect("{:e} writes an integer exponent"),
    )
}

/// Python's exponent form: the digits with a point after the first, then `e`, a sign and at
/// least two digits
fn exponent_form(f: &mut fmt::Formatter<'_>, digits: &str, exp: i32) -> fmt::Result {
    f.write_str(&digits[..1])?;
    if digits.len() > 1 {
        write!(f, ".{}", &digits[1..])?;
    }
    write!(f, "e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
}

impl fmt::Display for Repr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(text) = special(self.0, "0.0", "-0.0") {
            return f.write_str(text);
        }
        let (negative, digits, exp) = digits(self.0, None);
        if negative {
            f.write_str("-")?;
        }
        if !(-4..16).contains(&exp) {
            return exponent_form(f, &digits, exp);
        }
        if exp < 0 {
            return write!(f, "0.{}{}", "0".repeat((-exp - 1) as usize), digits);
        }
        let whole = exp as usize + 1;
        if digits.len() > whole {
            write!(f, "{}.{}", &digits[..whole], &digits[whole..])
        } else {
            write!(f, "{}{}.0", digits, "0".repeat(whole - digits.len()))
        }
    }
}

impl fmt::Display for General {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(text) = special(self.0, "0", "-0") {
            return f.write_str(text);
        }
        // The exponent once rounded to six digits picks the form, as in Python
        let (negative, digits, exp) = digits(self.0, Some(5));
        if !(-4..6).contains(&exp) {
            if negative {
                f.write_str("-")?;
            }
            return exponent_form(f, digits.trim_end_matches('0'), exp);
        }
        let text = format!("{:.*}", (5 - exp) as usize, self.0);
        let text = if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.')
        } else {
            &text
        };
        f.write_str(text)
    }
}

impl fmt::Display for Px {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.fract() == 0.0 && self.0.abs() < 1e16 {
            write!(f, "{}", self.0 as i64)
        } else {
            Repr(self.0).fmt(f)
        }
    }
}

impl fmt::Display for Escaped<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut rest = self.0;
        while let Some(at) = rest.find(['&', '<', '>', '"', '\'']) {
            f.write_str(&rest[..at])?;
            f.write_str(match rest.as_bytes()[at] {
                b'&' => "&amp;",
                b'<' => "&lt;",
                b'>' => "&gt;",
                b'"' => "&quot;",
                _ => "&#x27;",
            })?;
            rest = &rest[at + 1..];
        }
        f.write_str(rest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // (bits, repr, :g, .1f), as Python 3 writes them
    const CASES: &[(u64, &str, &str, &str)] = &[
        (0x0000000000000000, "0.0", "0", "0.0"),
        (0x8000000000000000, "-0.0", "-0", "-0.0"),
        (0x3ff0000000000000, "1.0", "1", "1.0"),
        (0x3fb999999999999a, "0.1", "0.1", "0.1"),
        (0x3fc999999999999a, "0.2", "0.2", "0.2"),
        (0x3ff8f5c28f5c28f6, "1.56", "1.56", "1.6"),
        (0x4000cccccccccccd, "2.1", "2.1", "2.1"),
        (0x3ee4f8b588e368f1, "1e-05", "1e-05", "0.0"),
        (0x3f1a36e2eb1c432d, "0.0001", "0.0001", "0.0"),
        (0x3f202e4b6ce5dc68, "0.00012345", "0.00012345", "0.0"),
        (0x4341c37937e08000, "1e+16", "1e+16", "10000000000000000.0"),
        (
            0x4341c37937e07fff,
            "9999999999999998.0",
            "1e+16",
            "9999999999999998.0",
        ),
        (
            0x437b6951ef585a00,
            "1.2345e+17",
            "1.2345e+17",
            "123450000000000000.0",
        ),
        (0x3e8421f5f40d8376, "1.5e-07", "1.5e-07", "0.0"),
        (0x0000000000000001, "5e-324", "4.94066e-324", "0.0"),
        (0xc004000000000000, "-2.5", "-2.5", "-2.5"),
        (0x3feb333333333333, "0.85", "0.85", "0.8"),
        (0x3fd5555555555555, "0.3333333333333333", "0.333333", "0.3"),
        (
            0xbfe5555555555555,
            "-0.6666666666666666",
            "-0.666667",
            "-0.7",
        ),
        (0x4059000000000000, "100.0", "100", "100.0"),
        (
            0x430c6bf526340000,
            "1000000000000000.0",
            "1e+15",
            "1000000000000000.0",
        ),
        (0x40c81cd6c8b43958, "12345.678", "12345.7", "12345.7"),
        (0x4132d68700000000, "1234567.0", "1.23457e+06", "1234567.0"),
        (0x412e847f00000000, "999999.5", "1e+06", "999999.5"),
        (0x4023ffff583a53b9, "9.999995", "10", "10.0"),
        (0x3ee9e40537474b0a, "1.234565e-05", "1.23456e-05", "0.0"),
        (0xbf1a36e2eb1c432d, "-0.0001", "-0.0001", "-0.0"),
        (0x3f1a36e2d51ec34b, "9.9999995e-05", "0.0001", "0.0"),
        (0x40fe240800000000, "123456.5", "123456", "123456.5"),
        (
            0x4480f0cf064dd592,
            "1e+22",
            "1e+22",
            "10000000000000000000000.0",
        ),
        (0x7ff8000000000000, "nan", "nan", "nan"),
        (0x7ff0000000000000, "inf", "inf", "inf"),
        (0xfff0000000000000, "-inf", "-inf", "-inf"),
    ];

    #[test]
    fn numbers_are_written_as_python_writes_them() {
        for &(bits, repr, general, f1) in CASES {
            let v = f64::from_bits(bits);
            assert_eq!(Repr(v).to_string(), repr, "repr of {v:e}");
            assert_eq!(General(v).to_string(), general, ":g of {v:e}");
            assert_eq!(F1(v).to_string(), f1, ".1f of {v:e}");
        }
        assert_eq!(Repr(f64::MAX).to_string(), "1.7976931348623157e+308");
        assert_eq!(General(f64::MAX).to_string(), "1.79769e+308");
    }

    #[test]
    fn sizes_are_written_as_ints_are() {
        assert_eq!(Px(700.0).to_string(), "700");
        assert_eq!(Px(700.5).to_string(), "700.5");
        assert_eq!(Px(-3.0).to_string(), "-3");
    }

    #[test]
    fn text_is_escaped_as_html_escape_does() {
        assert_eq!(Escaped("plain").to_string(), "plain");
        assert_eq!(
            Escaped("a & b < c > d \"e\" 'f' é").to_string(),
            "a &amp; b &lt; c &gt; d &quot;e&quot; &#x27;f&#x27; é"
        );
        assert_eq!(Escaped("&&").to_string(), "&amp;&amp;");
    }
}
