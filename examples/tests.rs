use neural::matrix::{init, Matrix};
use rand::Rng;

fn mat_mult_mat() {
    {
        let rows = 3;
        let cols = 3;
        let a_dat = vec![1.5, 1.0, 2.5, 2.5, 2.0, 3.5, 3.5, 3.0, 4.5];
        let b_dat = vec![2.0; rows * cols];
        let mut c_dat = Vec::new();
        for i in 0..a_dat.len() {
            c_dat.push(a_dat[i] * b_dat[i]);
        }
        let mut a = Matrix::from_slice_cm(&a_dat, rows, cols);
        let b = Matrix::from_slice_cm(&b_dat, rows, cols);
        a.multiply_matrix(&b);
        assert_eq!(a.to_vec(), c_dat);
    }
    {
        let rows = 3;
        let cols = 2;
        let a_dat = vec![1.5, 1.0, 2.5, 2.5, 2.0, 3.5];
        let b_dat = vec![2.0; rows * cols];
        let mut c_dat = Vec::new();
        for i in 0..a_dat.len() {
            c_dat.push(a_dat[i] * b_dat[i]);
        }
        let mut a = Matrix::from_slice_cm(&a_dat, rows, cols);
        let b = Matrix::from_slice_cm(&b_dat, rows, cols);
        a.multiply_matrix(&b);
        assert_eq!(a.to_vec(), c_dat);
    }
    println!("\x1b[0;32mpassed\x1b[0m");
}

fn mat_mult_scal() {
    {
        let rows = 3;
        let cols = 3;
        let scal = 0.023;
        let a_dat = vec![1.5, 1.0, 2.5, 2.5, 2.0, 3.5, 3.5, 3.0, 4.5];

        let mut c_dat = Vec::new();
        for i in 0..a_dat.len() {
            c_dat.push(a_dat[i] * scal);
        }
        let mut a = Matrix::from_slice_cm(&a_dat, rows, cols);
        a.multiply_scalar(scal);
        assert_eq!(a.to_vec(), c_dat);
    }
    {
        let rows = 300;
        let cols = 1;
        let scal = 0.023;
        let a_dat = vec![30.0; rows * cols];
        let mut c_dat = Vec::new();
        for i in 0..a_dat.len() {
            c_dat.push(a_dat[i] * scal);
        }
        let mut a = Matrix::from_slice_cm(&a_dat, rows, cols);
        a.multiply_scalar(scal);
        assert_eq!(a.to_vec(), c_dat);
    }
    println!("\x1b[0;32mpassed\x1b[0m");
}

fn mat_sub_mat() {
    {
        let rows = 3;
        let cols = 3;
        let a_dat = vec![1.5, 1.0, 2.5, 2.5, 2.0, 3.5, 3.5, 3.0, 4.5];
        let b_dat = vec![1.0; rows * cols];
        let mut c_dat = Vec::new();
        for i in 0..a_dat.len() {
            c_dat.push(a_dat[i] - b_dat[i]);
        }
        let a = Matrix::from_slice_cm(&a_dat, rows, cols);
        let b = Matrix::from_slice_cm(&b_dat, rows, cols);
        assert_eq!(a.subtract_matrix(&b).to_vec(), c_dat);
    }
    {
        let rows = 3;
        let cols = 2;
        let a_dat = vec![1.5, 1.0, 2.5, 2.5, 2.0, 3.5];
        let b_dat = vec![1.0; rows * cols];
        let mut c_dat = Vec::new();
        for i in 0..a_dat.len() {
            c_dat.push(a_dat[i] - b_dat[i]);
        }
        let a = Matrix::from_slice_cm(&a_dat, rows, cols);
        let b = Matrix::from_slice_cm(&b_dat, rows, cols);
        assert_eq!(a.subtract_matrix(&b).to_vec(), c_dat);
    }
    println!("\x1b[0;32mpassed\x1b[0m");
}

fn mat_add_mat() {
    {
        let rows = 3;
        let cols = 3;
        let a_dat = vec![1.5, 1.0, 2.5, 2.5, 2.0, 3.5, 3.5, 3.0, 4.5];
        let b_dat = vec![0.0069; rows * cols];
        let mut c_dat = Vec::new();
        for i in 0..a_dat.len() {
            c_dat.push(a_dat[i] + b_dat[i]);
        }
        let mut a = Matrix::from_slice_cm(&a_dat, rows, cols);
        let b = Matrix::from_slice_cm(&b_dat, rows, cols);
        a.add_matrix(&b);
        assert_eq!(a.to_vec(), c_dat);
    }
    {
        let rows = 3;
        let cols = 2;
        let a_dat = vec![1.5, 1.0, 2.5, 2.5, 2.0, 3.5];
        let b_dat = vec![0.0069; rows * cols];
        let mut c_dat = Vec::new();
        for i in 0..a_dat.len() {
            c_dat.push(a_dat[i] + b_dat[i]);
        }
        let mut a = Matrix::from_slice_cm(&a_dat, rows, cols);
        let b = Matrix::from_slice_cm(&b_dat, rows, cols);
        a.add_matrix(&b);
        assert_eq!(a.to_vec(), c_dat);
    }
    {
        let rows = 10;
        let cols = 128;
        let a_dat = vec![1.22; rows * cols];
        let b_dat = vec![0.0069; rows * cols];
        let mut c_dat = Vec::new();
        for i in 0..a_dat.len() {
            c_dat.push(a_dat[i] + b_dat[i]);
        }
        let mut a = Matrix::from_slice_cm(&a_dat, rows, cols);
        let b = Matrix::from_slice_cm(&b_dat, rows, cols);
        a.add_matrix(&b);
        assert_eq!(a.to_vec(), c_dat);
    }
    println!("\x1b[0;32mpassed\x1b[0m");
}

fn mat_add_scal() {
    {
        let rows = 3;
        let cols = 3;
        let scal = 0.023;
        let a_dat = vec![1.5, 1.0, 2.5, 2.5, 2.0, 3.5, 3.5, 3.0, 4.5];

        let mut c_dat = Vec::new();
        for i in 0..a_dat.len() {
            c_dat.push(a_dat[i] + scal);
        }
        let mut a = Matrix::from_slice_cm(&a_dat, rows, cols);
        a.add_scalar(scal);
        assert_eq!(a.to_vec(), c_dat);
    }
    {
        let rows = 300;
        let cols = 1;
        let scal = 0.023;
        let a_dat = vec![30.0; rows * cols];
        let mut c_dat = Vec::new();
        for i in 0..a_dat.len() {
            c_dat.push(a_dat[i] + scal);
        }
        let mut a = Matrix::from_slice_cm(&a_dat, rows, cols);
        a.add_scalar(scal);
        assert_eq!(a.to_vec(), c_dat);
    }
    println!("\x1b[0;32mpassed\x1b[0m");
}

fn transpose() {
    {
        let rows = 3;
        let cols = 3;
        let a_dat = vec![1.5, 1.0, 2.5, 2.5, 2.0, 3.5, 3.5, 3.0, 4.5];
        let exp = vec![1.5, 2.5, 3.5, 1.0, 2.0, 3.0, 2.5, 3.5, 4.5];
        let a = Matrix::from_slice_cm(&a_dat, rows, cols);
        let mut b = Matrix::new(cols, rows);
        a.transpose_into(&mut b);
        assert_eq!(exp, b.to_vec());
    }
    {
        let rows = 3;
        let cols = 2;
        let a_dat = vec![1.5, 1.0, 2.5, 2.5, 2.0, 3.5];
        let exp = vec![1.5, 2.5, 1.0, 2.0, 2.5, 3.5];
        let a = Matrix::from_slice_cm(&a_dat, rows, cols);
        let mut b = Matrix::new(cols, rows);
        a.transpose_into(&mut b);
        assert_eq!(exp, b.to_vec());
    }
    println!("\x1b[0;32mpassed\x1b[0m");
}

#[derive(Debug, Clone)]
struct CPUMatrix {
    pub data: Vec<Vec<f32>>,
    rows: usize,
    columns: usize,
}

impl CPUMatrix {
    pub fn new(rows: usize, columns: usize) -> Self {
        Self {
            data: vec![vec![0.0; columns]; rows],
            rows,
            columns,
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.iter().flatten().map(|v| *v).collect::<Vec<f32>>()
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    pub fn product(&self, m: &Self) -> Self {
        assert_eq!(self.columns, m.size().0);
        let mut res = Self::new(self.rows, m.size().1);
        for i in 0..res.rows {
            for j in 0..res.columns {
                let mut sum = 0.0;
                let mut k = 0;
                while k < self.columns {
                    unsafe {
                        sum += self.data.get_unchecked(i).get_unchecked(k)
                            * m.data.get_unchecked(k).get_unchecked(j);
                    }
                    k += 1;
                }
                unsafe {
                    *res.data.get_unchecked_mut(i).get_unchecked_mut(j) = sum;
                }
            }
        }
        res
    }

    pub fn transpose(&self) -> Self {
        let mut res = Self::new(self.columns, self.rows);
        for i in 0..self.rows {
            for j in 0..self.columns {
                res.data[j][i] = self.data[i][j];
            }
        }
        res
    }

    pub fn randomize(&mut self) {
        let mut rng = rand::thread_rng();
        for i in 0..self.rows {
            for j in 0..self.columns {
                self.data[i][j] = rng.gen_range(0.0..=1.0);
            }
        }
    }
}

fn gemm() {
    {
        let rows = 3;
        let cols = 3;
        let mut ca = CPUMatrix::new(rows, cols);
        ca.data[0][0] = 1.0;
        ca.data[0][1] = 2.0;
        ca.data[0][2] = 1.0;
        ca.data[1][0] = 6.0;
        ca.data[1][1] = 1.0;
        ca.data[1][2] = 6.0;
        ca.data[2][0] = 2.0;
        ca.data[2][1] = 3.0;
        ca.data[2][2] = 4.0;
        let mut cb = CPUMatrix::new(cols, rows);
        cb.data[0][0] = 2.0;
        cb.data[0][1] = 5.0;
        cb.data[0][2] = 15.0;
        cb.data[1][0] = 6.0;
        cb.data[1][1] = 7.0;
        cb.data[1][2] = 17.0;
        cb.data[2][0] = 1.0;
        cb.data[2][1] = 8.0;
        cb.data[2][2] = 18.0;
        let exp = ca.product(&cb).transpose();
        let ca_t = ca.transpose();
        let cb_t = cb.transpose();
        let a = Matrix::from_slice_cm(&ca_t.to_vec(), ca_t.rows, ca_t.columns);
        let b = Matrix::from_slice_cm(&cb_t.to_vec(), cb_t.rows, cb_t.columns);
        let c = a.product(&b);
        assert_eq!(exp.to_vec(), c.to_vec());
    }
    {
        let rows = 128;
        let cols = 128;
        let mut ca = CPUMatrix::new(rows, cols);
        ca.randomize();
        let mut cb = CPUMatrix::new(cols, rows);
        cb.randomize();
        let exp = ca.product(&cb).transpose();
        let ca_t = ca.transpose();
        let cb_t = cb.transpose();
        let a = Matrix::from_slice_cm(&ca_t.to_vec(), ca_t.rows, ca_t.columns);
        let b = Matrix::from_slice_cm(&cb_t.to_vec(), cb_t.rows, cb_t.columns);
        let c = a.product(&b);
        let real = exp.to_vec().iter().map(|v| v.round()).collect::<Vec<f32>>();
        let got = c.to_vec().iter().map(|v| v.round()).collect::<Vec<f32>>();
        assert_eq!(real, got);
    }
    println!("\x1b[0;32mpassed\x1b[0m");
}

fn sigmoid() {
    {
        let rows = 32;
        let cols = 32;
        let mut ca = CPUMatrix::new(rows, cols);
        ca.randomize();
        let ca_t = ca.transpose().to_vec();
        let mut a = Matrix::from_slice_cm(&ca_t.clone(), rows, cols);
        a.sigmoid();
        let real = ca_t
            .iter()
            .map(|v| 1.0 / (1.0 + (-v).exp()))
            .collect::<Vec<f32>>();
        let got = a.to_vec();
        for i in 0..real.len() {
            if real[i].max(got[i]) - real[i].min(got[i]) > 0.01 {
                panic!("Not the same");
            }
        }
    }
    println!("\x1b[0;32mpassed\x1b[0m");
}

fn main() {
    init();

    print!("Testing multiply_matrix...");
    mat_mult_mat();
    print!("Testing multiply_scalar...");
    mat_mult_scal();
    print!("Testing subtract_matrix...");
    mat_sub_mat();
    print!("Testing add_matrix...");
    mat_add_mat();
    print!("Testing add_scalar...");
    mat_add_scal();
    print!("Testing transpose_into...");
    transpose();
    print!("Testing product...");
    gemm();
    print!("Testing sigmoid...");
    sigmoid();
}
