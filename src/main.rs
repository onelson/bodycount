use opencv::core::no_array;
use opencv::imgcodecs::IMREAD_COLOR;
use opencv::{core, highgui, imgcodecs, imgproc, prelude::*, Result};

fn main() -> Result<()> {
    let mask = no_array()?;

    let template = imgcodecs::imread("you-died-bw.png", IMREAD_COLOR)?;
    let template_size = template.size()?;
    let full_frame =
        imgcodecs::imread(&std::env::args().nth(1).expect("input image"), IMREAD_COLOR)?;

    let height = 160;
    let width = 910;
    let x = 490;
    let y = 465;

    let frame = Mat::roi(&full_frame, core::Rect::new(x, y, width, height))?;

    let mut out = Mat::default()?;

    let meth = imgproc::TM_CCOEFF;

    imgproc::match_template(&frame, &template, &mut out, meth, &mask)?;

    let (_min_val, _max_val, min_loc, max_loc) = {
        let mut min_val = 0.;
        let mut max_val = 0.;
        let mut min_loc = core::Point::default();
        let mut max_loc = core::Point::default();
        core::min_max_loc(
            &out,
            &mut min_val,
            &mut max_val,
            &mut min_loc,
            &mut max_loc,
            &mask,
        )?;
        (min_val, max_val, min_loc, max_loc)
    };

    let top_left = if [imgproc::TM_SQDIFF_NORMED, imgproc::TM_SQDIFF_NORMED].contains(&meth) {
        min_loc
    } else {
        max_loc
    };

    println!("{:?} {}/{}", top_left, _min_val, _max_val);

    let rect = core::Rect::new(
        top_left.x,
        top_left.y,
        template_size.width,
        template_size.height,
    );

    let mut out2 = frame.clone();
    imgproc::rectangle(&mut out2, rect, core::Scalar_([0., 255., 0., 0.]), 4, 0, 0)?;

    highgui::imshow("hello opencv!", &out2)?;
    highgui::wait_key(30000)?;
    Ok(())
}
