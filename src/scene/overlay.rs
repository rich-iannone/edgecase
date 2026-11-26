// What a raster draws over the edges and nodes, as the SVG draws it: the labels and the legend,
// from their SVG, by resvg, which lays the text out in the system's fonts. A font comes with
// edgecase, so that the text shows where the system has none of the plot's fonts
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use resvg::tiny_skia::{Pixmap, Transform};
use resvg::usvg::{self, fontdb};
use std::sync::{Arc, OnceLock};

/// Liberation Sans (SIL Open Font License 1.1, see `fonts/LICENSE`), with Arial's metrics
const FALLBACK_FONT: &[u8] = include_bytes!("fonts/LiberationSans-Regular.ttf");
const FALLBACK_FAMILY: &str = "Liberation Sans";

/// The system's fonts and the fallback, read once
fn fonts() -> Arc<fontdb::Database> {
    static FONTS: OnceLock<Arc<fontdb::Database>> = OnceLock::new();
    FONTS
        .get_or_init(|| {
            let mut db = fontdb::Database::new();
            db.load_system_fonts();
            db.load_font_data(FALLBACK_FONT.to_vec());
            // The generic families, and what text falls back to when none of its own fonts
            // is there (usvg's last resort is serif)
            db.set_sans_serif_family(FALLBACK_FAMILY);
            db.set_serif_family(FALLBACK_FAMILY);
            Arc::new(db)
        })
        .clone()
}

/// `svg` drawn into an image of `width` × `height` pixels, which its own size is stretched to:
/// premultiplied RGBA, from 0 to 255, row by row
pub fn draw(svg: &str, width: u32, height: u32) -> PyResult<Vec<u8>> {
    let options = usvg::Options {
        fontdb: fonts(),
        font_family: FALLBACK_FAMILY.to_string(),
        ..Default::default()
    };
    let tree = usvg::Tree::from_str(svg, &options)
        .map_err(|e| PyValueError::new_err(format!("can't read the overlay's SVG: {e}")))?;
    let mut pixmap = Pixmap::new(width, height)
        .ok_or_else(|| PyValueError::new_err("an overlay needs an image of at least a pixel"))?;
    let size = tree.size();
    let transform =
        Transform::from_scale(width as f32 / size.width(), height as f32 / size.height());
    resvg::render(&tree, transform, &mut pixmap.as_mut());
    Ok(pixmap.take())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alpha(pixels: &[u8], width: u32, x: u32, y: u32) -> u8 {
        pixels[((y * width + x) * 4 + 3) as usize]
    }

    #[test]
    fn the_svg_is_stretched_to_the_image() {
        let svg = "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"20\" height=\"10\" \
                   viewBox=\"0 0 20 10\"><rect x=\"10\" width=\"10\" height=\"10\" \
                   fill=\"#ff0000\" fill-opacity=\"0.5\"/></svg>";
        let pixels = draw(svg, 40, 20).unwrap();
        assert_eq!(pixels.len(), 40 * 20 * 4);
        assert_eq!(alpha(&pixels, 40, 10, 10), 0);
        // Premultiplied
        let at = ((10 * 40 + 30) * 4) as usize;
        assert!((127..=128).contains(&pixels[at]) && pixels[at] == pixels[at + 3]);
        assert_eq!(&pixels[at + 1..at + 3], &[0, 0]);
    }

    #[test]
    fn text_shows_in_any_font() {
        // A family the system hasn't falls back to the bundled font
        for family in ["sans-serif", "No Such Font", "serif"] {
            let svg = format!(
                "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"60\" height=\"20\" \
                 style=\"font-family: {family};\"><text x=\"2\" y=\"15\" font-size=\"14\">\
                 Hello</text></svg>"
            );
            let pixels = draw(&svg, 60, 20).unwrap();
            let inked = pixels.chunks_exact(4).filter(|p| p[3] > 128).count();
            assert!(inked > 40, "{family}: {inked}");
        }
    }

    #[test]
    fn bad_svg_is_an_error() {
        assert!(draw("<svg", 10, 10).is_err());
    }
}
