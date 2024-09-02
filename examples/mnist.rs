use std::{
    fs::File,
    io::{self, BufReader, Read, Write},
    time::Instant,
};

use rand::seq::SliceRandom;

use neural::nn::NeuralNetwork;

macro_rules! verify_img_header {
    ($n:expr, $buf:expr) => {
        assert_eq!(
            (($buf[0] as i32) << 24)
                + (($buf[1] as i32) << 16)
                + (($buf[2] as i32) << 8)
                + ($buf[3] as i32),
            0x00000803
        );
        assert_eq!(
            (($buf[4] as i32) << 24)
                + (($buf[5] as i32) << 16)
                + (($buf[6] as i32) << 8)
                + ($buf[7] as i32),
            $n
        );
        assert_eq!(
            (($buf[8] as i32) << 24)
                + (($buf[9] as i32) << 16)
                + (($buf[10] as i32) << 8)
                + ($buf[11] as i32),
            28
        );
        assert_eq!(
            (($buf[12] as i32) << 24)
                + (($buf[13] as i32) << 16)
                + (($buf[14] as i32) << 8)
                + ($buf[15] as i32),
            28
        );
    };
}

macro_rules! verify_labels_header {
    ($n:expr, $buf:expr) => {
        assert_eq!(
            (($buf[0] as i32) << 24)
                + (($buf[1] as i32) << 16)
                + (($buf[2] as i32) << 8)
                + ($buf[3] as i32),
            0x00000801
        );
        assert_eq!(
            (($buf[4] as i32) << 24)
                + (($buf[5] as i32) << 16)
                + (($buf[6] as i32) << 8)
                + ($buf[7] as i32),
            $n
        );
    };
}

#[derive(Copy, Clone)]
struct Image {
    pub data: [f32; 784],
    pub label: [f32; 10],
}

#[inline(always)]
fn secs_to_human(secs: u64) -> String {
    let mut secs = secs;
    let mut s = String::new();
    if secs > 60 * 60 {
        let hours = secs / (60 * 60);
        secs -= hours * 60 * 60;
        s.push_str(&format!("{}h", hours));
    }
    if secs > 60 {
        let mins = secs / 60;
        secs -= mins * 60;
        s.push_str(&format!("{}m", mins));
    }
    s.push_str(&format!("{}s", secs));
    s
}

fn parse_training_images() -> Vec<Image> {
    let mut images = Vec::new();
    let img_f = File::open("./data/mnist/train-images-idx3-ubyte").unwrap();
    let label_f = File::open("./data/mnist/train-labels-idx1-ubyte").unwrap();
    let mut img_reader = BufReader::new(img_f);
    let mut label_reader = BufReader::new(label_f);

    let mut img_header_buf: [u8; 16] = [0; 16];
    img_reader.read_exact(&mut img_header_buf).unwrap();
    verify_img_header!(60000, img_header_buf);

    let mut label_header_buf: [u8; 8] = [0; 8];
    label_reader.read_exact(&mut label_header_buf).unwrap();
    verify_labels_header!(60000, label_header_buf);

    for _ in 0..60000 {
        let mut img_buf: [u8; 784] = [0; 784];
        let mut label_buf: [u8; 1] = [0; 1];
        img_reader.read_exact(&mut img_buf).unwrap();
        label_reader.read_exact(&mut label_buf).unwrap();
        assert!(label_buf[0] < 10);
        let mut label = [0.0; 10];
        label[label_buf[0] as usize] = 1.0;
        images.push(Image {
            data: img_buf
                .iter()
                .map(|v| *v as f32 / 255.0)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
            label,
        });
    }

    images
}

fn parse_test_images() -> Vec<Image> {
    let mut images = Vec::new();
    let img_f = File::open("./data/mnist/t10k-images-idx3-ubyte").unwrap();
    let label_f = File::open("./data/mnist/t10k-labels-idx1-ubyte").unwrap();
    let mut img_reader = BufReader::new(img_f);
    let mut label_reader = BufReader::new(label_f);

    let mut img_header_buf: [u8; 16] = [0; 16];
    img_reader.read_exact(&mut img_header_buf).unwrap();
    verify_img_header!(10000, img_header_buf);

    let mut label_header_buf: [u8; 8] = [0; 8];
    label_reader.read_exact(&mut label_header_buf).unwrap();
    verify_labels_header!(10000, label_header_buf);

    for _ in 0..10000 {
        let mut img_buf: [u8; 784] = [0; 784];
        let mut label_buf: [u8; 1] = [0; 1];
        img_reader.read_exact(&mut img_buf).unwrap();
        label_reader.read_exact(&mut label_buf).unwrap();
        assert!(label_buf[0] < 10);
        let mut label = [0.0; 10];
        label[label_buf[0] as usize] = 1.0;
        images.push(Image {
            data: img_buf
                .iter()
                .map(|v| *v as f32 / 255.0)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap(),
            label,
        });
    }

    images
}

fn main() {
    let training = parse_training_images();
    let tests = parse_test_images();

    let mut nn = NeuralNetwork::new(784, vec![128, 128, 128], 10);
    nn.set_learning_rate(0.1);

    let mut before = 100.0;
    if true {
        println!("Testing...");

        let mut correct = 0;
        let mut total = 0;
        for test_img in &tests {
            let pred = nn.feedforward(test_img.data.to_vec());
            let label_index = 'out: {
                for (i, l) in test_img.label.iter().enumerate() {
                    if *l == 1.0 {
                        break 'out i;
                    }
                }
                unreachable!();
            };
            let max = {
                let mut m = 0.0;
                let mut idx = 0;
                for (i, p) in pred.iter().enumerate() {
                    if *p > m {
                        idx = i;
                        m = *p;
                    }
                }
                idx
            };
            if label_index == max {
                correct += 1;
            }
            total += 1;
        }
        before = correct as f64 / total as f64 * 100.0;
        println!("Accuracy: {:.20}%", before);
    }

    let start = Instant::now();
    let mut last = 0;
    // const TRAINING_ITERATIONS: usize = 100000;
    const TRAINING_ITERATIONS: usize = 100000;
    print!("Training... Elapsed time: 0s [0/{TRAINING_ITERATIONS} 0.00%]");
    io::stdout().flush().unwrap();
    let mut rng = rand::thread_rng();
    for index in 0..TRAINING_ITERATIONS {
        let training_img = training.choose(&mut rng).unwrap();
        nn.train(&training_img.data, &training_img.label);
        let elapsed_secs = start.elapsed().as_secs();
        if elapsed_secs - last > 0 {
            print!(
                "\r\x1B[0JTraining... Elapsed time: {} [{index}/{TRAINING_ITERATIONS} {:.2}%]",
                secs_to_human(elapsed_secs),
                index as f32 / TRAINING_ITERATIONS as f32 * 100.0
            );
            io::stdout().flush().unwrap();
            last = elapsed_secs;
        }
    }

    println!(
        "\r\x1B[0JTrained on {TRAINING_ITERATIONS} images in {}",
        secs_to_human(start.elapsed().as_secs())
    );

    println!("Testing...");

    let mut correct = 0;
    let mut total = 0;
    for test_img in &tests {
        let pred = nn.feedforward(test_img.data.to_vec());
        let label_index = 'out: {
            for (i, l) in test_img.label.iter().enumerate() {
                if *l == 1.0 {
                    break 'out i;
                }
            }
            unreachable!();
        };
        let max = {
            let mut m = 0.0;
            let mut idx = 0;
            for (i, p) in pred.iter().enumerate() {
                if *p > m {
                    idx = i;
                    m = *p;
                }
            }
            idx
        };
        if label_index == max {
            correct += 1;
        }
        total += 1;
    }

    let after = correct as f64 / total as f64 * 100.0;
    println!("Accuracy: {:.20}% +{}%", after, before - after);
}
