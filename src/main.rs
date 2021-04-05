//! Port of https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp
//! Check the source cpp file for where to get the NN files.

use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

use opencv::{
    core::{self, Point2f, Scalar, Size},
    dnn, highgui, imgcodecs, imgproc,
    prelude::*,
    types::{VectorOfPoint2f, VectorOfString, VectorOfVectorOfPoint},
};

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

fn main() -> Result<()> {
    let full_frame = imgcodecs::imread(
        &std::env::args().nth(1).expect("input image"),
        imgcodecs::IMREAD_COLOR,
    )?;

    // XXX: need to mess with the crop once we get video moving through
    let height = 480;
    let width = 1100;
    let x = 380;
    let y = 300;
    let frame = Mat::roi(&full_frame, core::Rect::new(x, y, width, height))?;

    // Parameters.
    let conf_threshold = 0.5;
    let nms_threshold = 0.4;

    let imread_rgb = false;

    let det_model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("dnn")
        .join("frozen_east_text_detection.pb");
    let rec_model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("dnn")
        .join("CRNN_VGG_BiLSTM_CTC.onnx");
    let voc_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("dnn")
        .join("alphabet_36.txt");

    // Load networks.
    let mut detector =
        dnn::TextDetectionModel_EAST::from_file(det_model_path.to_str().unwrap(), "")?;
    detector
        .set_confidence_threshold(conf_threshold)?
        .set_nms_threshold(nms_threshold)?;
    let mut recognizer =
        dnn::TextRecognitionModel::from_file(rec_model_path.to_str().unwrap(), "")?;

    // Load vocabulary
    let mut vocabulary = VectorOfString::new();
    let voc_file = BufReader::new(File::open(voc_path)?);
    for voc_line in voc_file.lines() {
        vocabulary.push(&voc_line?);
    }
    recognizer
        .set_vocabulary(&vocabulary)?
        .set_decode_type("CTC-greedy")?;

    // Parameters for Recognition
    let rec_scale = 1. / 127.5;
    let rec_mean = Scalar::from((127.5, 127.5, 127.5));
    let rec_input_size = Size::new(100, 32);
    recognizer.set_input_params(rec_scale, rec_input_size, rec_mean, false, false)?;

    // Parameters for Detection
    let det_scale = 1.;
    let det_input_size = Size::new(320, 320);
    let det_mean = Scalar::from((123.68, 116.78, 103.94));
    let swap_rb = true;
    detector.set_input_params(det_scale, det_input_size, det_mean, swap_rb, false)?;

    let win_name = "EAST: An Efficient and Accurate Scene Text Detector";

    println!("Frame size: {:?}", frame.size()?);
    // Detection
    let mut det_results = VectorOfVectorOfPoint::new();
    detector.detect(&frame, &mut det_results)?;

    if !det_results.is_empty() {
        // Text Recognition
        let rec_input = if !imread_rgb {
            let mut rec_input = Mat::default()?;
            imgproc::cvt_color(&frame, &mut rec_input, imgproc::COLOR_BGR2GRAY, 0)?;
            Some(rec_input)
        } else {
            None
        };

        for quadrangle in &det_results {
            let mut quadrangle_2f = VectorOfPoint2f::new();
            for pt in &quadrangle {
                quadrangle_2f.push(Point2f::new(pt.x as f32, pt.y as f32))
            }
            let cropped = four_points_transform(
                rec_input.as_ref().unwrap_or(&frame),
                quadrangle_2f.as_slice(),
            )?;

            let recognition_result = recognizer.recognize(&cropped)?;
            println!("Recognition result: {}", recognition_result);
        }
    }

    highgui::imshow(win_name, &frame)?;
    highgui::wait_key(10_000)?;

    Ok(())
}

fn four_points_transform(frame: &Mat, vertices: &[Point2f]) -> Result<Mat> {
    let output_size = Size::new(100, 32);
    let target_vertices = [
        Point2f::new(0., (output_size.height - 1) as f32),
        Point2f::new(0., 0.),
        Point2f::new((output_size.width - 1) as f32, 0.),
        Point2f::new(
            (output_size.width - 1) as f32,
            (output_size.height - 1) as f32,
        ),
    ];
    let rotation_matrix =
        imgproc::get_perspective_transform_slice(&vertices, &target_vertices, core::DECOMP_LU)?;
    let mut out = Mat::default()?;
    imgproc::warp_perspective(
        frame,
        &mut out,
        &rotation_matrix,
        output_size,
        imgproc::INTER_LINEAR,
        core::BORDER_CONSTANT,
        Scalar::default(),
    )?;
    Ok(out)
}
