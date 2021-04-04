use opencv::core::{no_array, Scalar};
use opencv::features2d::DrawMatchesFlags;
use opencv::imgcodecs::IMREAD_COLOR;
use opencv::types::{VectorOfDMatch, VectorOfi8};
use opencv::{
    core, features2d, features2d::Feature2DTrait, highgui, imgcodecs, prelude::*, Result,
};

fn main() -> Result<()> {
    let mask = no_array()?;

    let target = imgcodecs::imread("you-died.png", IMREAD_COLOR)?;
    let full_frame =
        imgcodecs::imread(&std::env::args().nth(1).expect("input image"), IMREAD_COLOR)?;

    let height = 160;
    let width = 910;
    let x = 490;
    let y = 465;

    let frame = Mat::roi(&full_frame, core::Rect::new(x, y, width, height))?;

    let mut orb = features2d::ORB::default()?;

    let (kp1, des1) = {
        let mut kps = opencv::types::VectorOfKeyPoint::new();
        let mut descriptors = Mat::default()?;
        orb.detect_and_compute(&target, &mask, &mut kps, &mut descriptors, false)?;
        (kps, descriptors)
    };

    let (kp2, des2) = {
        let mut kps = opencv::types::VectorOfKeyPoint::new();
        let mut descriptors = Mat::default()?;
        orb.detect_and_compute(&frame, &mask, &mut kps, &mut descriptors, false)?;
        (kps, descriptors)
    };

    let bf = features2d::BFMatcher::new(core::NORM_HAMMING, true)?;
    let matches = {
        let mut matches = VectorOfDMatch::default();
        bf.train_match(&des1, &des2, &mut matches, &mask)?;
        matches
    };

    println!("matches: {}", matches.len());

    let mut matches = matches.to_vec();

    matches.sort_by_key(|x| x.distance as i64);

    for match_ in &matches {
        println!(
            "{:?}",
            (
                match_.query_idx,
                match_.img_idx,
                match_.train_idx,
                match_.distance
            )
        );
    }

    let mut out = Mat::default()?;
    let matches_mask = VectorOfi8::default();
    features2d::draw_matches(
        &target,
        &kp1,
        &frame,
        &kp2,
        &VectorOfDMatch::from_iter(matches.into_iter().take(3)),
        &mut out,
        Scalar::new(0., 127., 0., 0.),
        Scalar::new(0., 0., 127., 0.),
        &matches_mask,
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS,
    )?;
    highgui::imshow("hello opencv!", &out)?;
    highgui::wait_key(30000)?;
    Ok(())
}
