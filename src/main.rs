use opencv::{core, highgui, imgcodecs, imgproc, prelude::*, Result};

fn main() -> Result<()> {
    let target = imgcodecs::imread("you-died.png", imgcodecs::IMREAD_COLOR)?;
    let full_frame = imgcodecs::imread(
        &std::env::args().nth(1).expect("input image"),
        imgcodecs::IMREAD_COLOR,
    )?;

    println!(
        "channels: {:?}",
        (target.channels()?, full_frame.channels()?)
    );

    let height = 160;
    let width = 910;
    let x = 490;
    let y = 465;
    let frame = Mat::roi(&full_frame, core::Rect::new(x, y, width, height))?;

    let mut splits1 = opencv::types::VectorOfMat::new();
    let mut splits2 = opencv::types::VectorOfMat::new();

    core::split(&target, &mut splits1).expect("split 1");
    core::split(&frame, &mut splits2).expect("split 2");

    let red1 = &splits1.get(2)?;
    let red2 = &splits2.get(2)?;

    highgui::imshow("red1", red1)?;
    highgui::imshow("red2", red2)?;

    let mask = core::no_array()?;

    let mut h1 = Mat::default()?;
    let mut h2 = Mat::default()?;

    imgproc::calc_hist(
        &splits1,
        &core::Vector::from(vec![2]),
        &mask,
        &mut h1,
        &core::Vector::from(vec![256]),
        &core::Vector::from(vec![0., 256.]),
        false,
    )
    .expect("calc hist 1");

    imgproc::calc_hist(
        &splits2,
        &core::Vector::from(vec![2]),
        &mask,
        &mut h2,
        &core::Vector::from(vec![256]),
        &core::Vector::from(vec![0., 256.]),
        false,
    )
    .expect("calc hist 2");

    let mut norm1 = Mat::default()?;
    let mut norm2 = Mat::default()?;
    core::normalize(&h1, &mut norm1, 0., 1., core::NORM_MINMAX, -1, &mask)?;
    core::normalize(&h2, &mut norm2, 0., 1., core::NORM_MINMAX, -1, &mask)?;

    let distance = imgproc::compare_hist(&norm1, &norm2, imgproc::HISTCMP_INTERSECT)?;

    println!("distance: {}", distance);

    highgui::wait_key(10_000)?;
    Ok(())
}
