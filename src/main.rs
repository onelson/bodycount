use opencv::{core, highgui, imgcodecs, imgproc, prelude::*, types, Result};

fn main() -> Result<()> {
    let target = imgcodecs::imread(
        concat!(env!("CARGO_MANIFEST_DIR"), "/", "you-died.png"),
        imgcodecs::IMREAD_COLOR,
    )?;
    let full_frame = imgcodecs::imread(
        &std::env::args().nth(1).expect("input image"),
        imgcodecs::IMREAD_COLOR,
    )?;

    // FIXME: need to accept offsets for the bbox to match within.
    // let height = 160;
    // let width = 910;
    // let x = 490;
    // let y = 465;
    // let frame = Mat::roi(&full_frame, core::Rect::new(x, y, width, height))?;

    let frame = full_frame;

    highgui::imshow("frame", &frame)?;

    let mask = Mat::default();

    let mut hsv1 = Mat::default();
    let mut hsv2 = Mat::default();

    imgproc::cvt_color(&target, &mut hsv1, imgproc::COLOR_BGR2HSV, 0)?;
    imgproc::cvt_color(&frame, &mut hsv2, imgproc::COLOR_BGR2HSV, 0)?;

    let mut hist1 = Mat::default();
    let mut hist2 = Mat::default();

    debug_assert_eq!(hsv1.channels()?, 3);
    debug_assert_ne!(hsv1.size()?.width, 0);
    debug_assert_ne!(hsv1.size()?.height, 0);

    let input1 = types::VectorOfMat::from(vec![hsv1]);

    imgproc::calc_hist(
        &input1,
        &vec![0, 1].into(),
        &mask,
        &mut hist1,
        &vec![50, 60].into(),
        &vec![0., 180., 0., 256.].into(),
        true,
    )
    .expect("calc hist 1");

    let input2 = types::VectorOfMat::from(vec![hsv2]);
    imgproc::calc_hist(
        &input2,
        &vec![0, 1].into(),
        &mask,
        &mut hist2,
        &vec![50, 60].into(),
        &vec![0., 180., 0., 256.].into(),
        false,
    )
    .expect("calc hist 2");

    let mut norm1 = Mat::default();
    let mut norm2 = Mat::default();
    core::normalize(&hist1, &mut norm1, 0., 1., core::NORM_MINMAX, -1, &mask)?;
    core::normalize(&hist2, &mut norm2, 0., 1., core::NORM_MINMAX, -1, &mask)?;

    let distance = imgproc::compare_hist(&norm1, &norm2, imgproc::HISTCMP_INTERSECT)?;

    println!("distance: {}", distance);

    highgui::wait_key(10_000)?;
    Ok(())
}
