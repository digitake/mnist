extern crate rand;
use rand::Rng;

use std::io::{self, Write, BufWriter, Read};
use std::fs::File;
use rust_mnist::{print_image, Mnist};

fn normalize(image: &[u8]) -> Vec<f64> {
    // Normalize the image.
    image
        .iter()
        .map(|pixel| 2.0 * f64::from(*pixel) / 255.0 - 1.0)
        .collect()
}


fn largest(arr: &[f64; 10]) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}


fn softmax(arr: &[f64; 10]) -> [f64; 10] {
    let exp: Vec<f64> = arr.iter().map(|x| x.exp()).collect();
    let sum_exp: f64 = exp.iter().sum();
    let mut softmax_arr: [f64; 10] = [0.0; 10];
    for index in 0..softmax_arr.len() {
        softmax_arr[index] = exp[index] / sum_exp;
    }
    softmax_arr
}


fn read_weight_from_file() -> [[f64; 785]; 10] {
    let mut file = File::open("weights.txt").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    let mut weights: [[f64; 785]; 10] = [[0.0; 785]; 10];
    for (i, line) in contents.lines().enumerate() {
        for (j, number) in line.split_whitespace().enumerate() {
            weights[i][j] = number.parse().unwrap();
        }
    }
    weights
}


fn generate_output(image: &[u8], weights: [[f64; 785]; 10]) -> [f64; 10] {
    // Normalize the image.
    let image = normalize(image);

    // Calculate the outputs.
    let mut outputs = dot_product(&image, weights);
    outputs = softmax(&outputs);

    outputs
}


fn dot_product(image: &[f64], weights: [[f64; 785]; 10]) -> [f64; 10] {
    let mut outputs: [f64; 10] = [0.0; 10];
    for output_index in 0..outputs.len() {
        for (pixel_index, pixel) in image.iter().enumerate() {
            outputs[output_index] += pixel * weights[output_index][pixel_index];
            outputs[output_index] += weights[output_index][784];
        }
    }
    outputs
}


fn main() {
    // Load the dataset into an "Mnist" object. 
    // If on windows, replace the forward slashes with backslashes.
    let mnist = Mnist::new("data/");

    // Print one image (the one at index 5) for verification.
    let mut rng = rand::thread_rng();
    let n: usize = rng.gen_range(0..mnist.test_data.len());
    print_image(&mnist.test_data[n], mnist.test_labels[n]);

    // Read weights from file.
    let mut weights = read_weight_from_file();

    let outputs = generate_output(&mnist.test_data[n], weights);
    let n = largest(&outputs);
    println!("predict = {:?}", n);

}