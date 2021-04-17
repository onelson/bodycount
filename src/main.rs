//! Port of https://github.com/opencv/opencv/blob/master/samples/dnn/text_detection.cpp
//! Check the source cpp file for where to get the NN files.

use levenshtein::levenshtein;
use opencv::{
    core::{self, Point2f, Scalar, Size},
    dnn, highgui, imgproc,
    prelude::*,
    types::{VectorOfPoint2f, VectorOfString, VectorOfVectorOfPoint},
    videoio,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};
use std::{
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};
use structopt::StructOpt;

type Result<T, E = Box<dyn Error>> = std::result::Result<T, E>;

#[derive(Debug, StructOpt)]
struct Opt {
    #[structopt(short, long, env)]
    stream_url: String,
    #[structopt(short, long, env, parse(from_os_str))]
    data_file: PathBuf,
    #[structopt(long, env, default_value = "0.0.0.0")]
    http_host: String,
    #[structopt(long, env, default_value = "7878")]
    http_port: u16,
}

/// Find text in the frame.
fn detect_text(
    frame: &Mat,
    detector: &dnn::TextDetectionModel_EAST,
    recognizer: &dnn::TextRecognitionModel,
) -> Result<Option<Vec<String>>> {
    let mut det_results = VectorOfVectorOfPoint::new();
    detector.detect(&frame, &mut det_results)?;

    if !det_results.is_empty() {
        let mut matches = vec![];
        // Text Recognition
        let rec_input = {
            let mut rec_input = Mat::default()?;
            imgproc::cvt_color(&frame, &mut rec_input, imgproc::COLOR_BGR2GRAY, 0)?;
            rec_input
        };

        for quadrangle in &det_results {
            let mut quadrangle_2f = VectorOfPoint2f::new();
            for pt in &quadrangle {
                quadrangle_2f.push(Point2f::new(pt.x as f32, pt.y as f32))
            }
            let cropped = four_points_transform(&rec_input, quadrangle_2f.as_slice())?;

            let recognition_result = recognizer.recognize(&cropped)?;
            matches.push(recognition_result);
        }
        return Ok(Some(matches));
    }

    Ok(None)
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

    // XXX: Seems like we could probably skip the warp since the text should always be level.
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

fn get_dnn() -> Result<(dnn::TextDetectionModel_EAST, dnn::TextRecognitionModel)> {
    let conf_threshold = 0.5;
    let nms_threshold = 0.4;

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
    Ok((detector, recognizer))
}

fn watch_stream(video_url: &str, count_data: SharedCountData) -> Result<()> {
    let (detector, recognizer) = get_dnn()?;

    let mut cam = videoio::VideoCapture::from_file(video_url, videoio::CAP_FFMPEG)?;
    let opened = videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open stream!");
    }
    let mut frame_num = 0;
    let mut frame = Mat::default()?;

    let count = count_data.lock().unwrap().len();
    let mut last_seen: Vec<Option<SystemTime>> = Vec::with_capacity(count);
    for _i in 0..count {
        last_seen.push(None);
    }

    const DEBOUNCE: Duration = Duration::from_secs(2);
    // How often to run text detection.
    const FRAME_RATE: u8 = 5;

    loop {
        frame_num += 1;
        cam.grab()?;
        if frame_num % FRAME_RATE == 0 {
            frame_num = 0;
            cam.retrieve(&mut frame, 0)?;

            // FIXME: offload text matching to another thread
            //  Buffer only the most current frame, and share whatever it is
            //  with the text detection thread.

            highgui::imshow("monitor", &frame)?;
            highgui::wait_key(1)?;

            let size = frame.size()?;
            if size.width > 0 {
                let col = size.width / 8;
                let row = size.height / 8;
                let height = row * 4;
                let width = col * 6;
                let x = row;
                let y = col * 2;
                let frame = Mat::roi(&frame, core::Rect::new(x, y, width, height))?;

                if let Some(matches) = detect_text(&frame, &detector, &recognizer)? {
                    log::trace!("Matches: {:?}", &matches);
                    {
                        let records = count_data.lock().unwrap();

                        for recognizer_match in &matches {
                            for (idx, record) in records.iter().enumerate() {
                                if levenshtein(&record.match_text, &recognizer_match)
                                    <= record.threshold
                                {
                                    last_seen[idx] = Some(SystemTime::now());
                                }
                            }
                        }
                    }
                }
            }
        }

        let now = SystemTime::now();
        let ready_to_inc: Vec<_> = last_seen
            .iter_mut()
            .enumerate()
            .filter_map(|(idx, x)| {
                x.and_then(|when| {
                    if now.duration_since(when).unwrap() >= DEBOUNCE {
                        *x = None; // reset timer
                        Some(idx)
                    } else {
                        None
                    }
                })
            })
            .collect();

        if !ready_to_inc.is_empty() {
            let mut counts = count_data.lock().unwrap();
            for idx in ready_to_inc {
                log::debug!("Inc idx=`{}` ({})", idx, &counts[idx].label);
                counts[idx].value += 1;
            }
        }

        // FIXME: need a way to terminate the loop during shutdown.
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CountRecord {
    threshold: usize,
    label: String,
    match_text: String,
    value: usize,
}

type SharedCountData = Arc<Mutex<Vec<CountRecord>>>;

fn main() -> Result<()> {
    env_logger::init();
    let args: Opt = Opt::from_args();

    let stream_url = args.stream_url;
    let http_host = args.http_host;
    let http_port = args.http_port.to_string();

    let count_data: Vec<CountRecord> =
        serde_json::from_reader(std::fs::File::open(&args.data_file)?)?;

    let count_data = Arc::new(Mutex::new(count_data));
    let http_counts = count_data.clone();
    let cv_counts = count_data.clone();

    let _http_thread = std::thread::spawn(move || {
        let count_data = http_counts;
        log::info!("Starting http server.");
        let server = simple_server::Server::new(move |_req, mut resp| {
            let data = count_data.lock().unwrap();
            Ok(resp
                .header("content-type", "application/json")
                .body(serde_json::to_vec(data.as_slice()).unwrap())?)
        });
        server.listen(&http_host, &http_port);
    });
    let cv_thread = std::thread::spawn(move || {
        log::info!("Starting stream watcher.");
        watch_stream(&stream_url, cv_counts).map_err(|e| e.to_string())
    });

    cv_thread.join().unwrap()?;
    Ok(())
}
