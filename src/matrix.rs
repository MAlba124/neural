use std::{collections::HashMap, sync::Mutex};

use ocl::{Buffer, MemFlags, ProQue};
use rand::Rng;

#[derive(Debug, Clone)]
#[repr(align(64))]
pub struct Matrix {
    // Row major flat array
    pub data: Vec<f32>,
    rows: usize,
    columns: usize,
}

lazy_static::lazy_static! {
    static ref PRO_QUE: ProQue = ProQue::builder()
        .src(include_str!("../kernels/matrix.opencl"))
        .build()
        .unwrap();
    static ref BUFFER_POOL: BufferPool = BufferPool::new();
}

struct BufferPool {
    buffers: Mutex<HashMap<usize, Vec<Buffer<f32>>>>,
}

impl BufferPool {
    pub fn new() -> Self {
        Self {
            buffers: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_buffer(&self, n: usize) -> Buffer<f32> {
        let mut buffers = self.buffers.lock().unwrap();
        let entry = buffers.entry(n).or_insert(Vec::new());
        if entry.is_empty() {
            entry.push(unsafe {
                Buffer::new(PRO_QUE.context(), MemFlags::READ_WRITE, n, None).unwrap()
            });
        }

        entry.pop().unwrap()
    }

    pub fn give_buffer(&self, buffer: Buffer<f32>) {
        let mut buffers = self.buffers.lock().unwrap();
        let _ = buffers
            .entry(buffer.len())
            .or_insert(Vec::new())
            .push(buffer);
    }
}

impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Self {
        Self {
            data: vec![0.0; columns * rows],
            rows,
            columns,
        }
    }

    #[inline(always)]
    pub fn from_slice(v: &[f32]) -> Self {
        let mut m = Self::new(v.len(), 1);
        m.data[..m.rows].copy_from_slice(&v[..m.rows]);
        m
    }

    #[inline(always)]
    pub fn to_vec(&self) -> Vec<f32> {
        self.data.to_vec()
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    #[inline(always)]
    pub fn multiply_scalar(&mut self, n: f32) {
        self.data.iter_mut().for_each(|v| *v *= n);
        // SLOW:
        // let mut a_buf = BUFFER_POOL.get_buffer(self.data.len());
        // a_buf.set_default_queue(PRO_QUE.queue().clone());
        // a_buf.write(&self.data).enq().unwrap();

        // let kernel = PRO_QUE.kernel_builder("mult_scalar")
        //     .arg(&a_buf)
        //     .arg(n)
        //     // .global_work_size(self.data.len())
        //     .global_work_size(1)
        //     .build()
        //     .unwrap();

        // unsafe {
        //     kernel.enq().unwrap();
        // }

        // a_buf.set_default_queue(PRO_QUE.queue().clone());
        // a_buf.read(&mut self.data).enq().unwrap();

        // BUFFER_POOL.give_buffer(a_buf);
    }

    #[inline(always)]
    pub fn add_scalar(&mut self, n: f32) {
        self.data.iter_mut().for_each(|v| *v += n);
    }

    #[inline(always)]
    pub fn subtract_matrix(&self, b: &Self) -> Matrix {
        #[cfg(debug_assertions)]
        assert_eq!(self.size(), b.size());

        let mut res = Matrix::new(self.rows, self.columns);
        for i in 0..self.data.len() {
            res.data[i] = self.data[i] - b.data[i];
        }
        res
    }

    #[inline(always)]
    pub fn multiply_matrix(&mut self, b: &Self) {
        #[cfg(debug_assertions)]
        assert_eq!(self.size(), b.size());

        for i in 0..self.data.len() {
            self.data[i] *= b.data[i];
        }
    }

    #[inline(always)]
    pub fn multiply_matrix_ret(&self, b: &Self) -> Self {
        let mut a = self.clone();
        a.multiply_matrix(b);
        a
    }

    #[inline(always)]
    pub fn add_matrix(&mut self, b: &Self) {
        #[cfg(debug_assertions)]
        assert_eq!(b.size(), self.size());

        for i in 0..self.data.len() {
            self.data[i] += b.data[i];
        }
    }

    // TODO: Do tilewise for better cache friendliness
    #[inline(always)]
    pub fn product_into(&self, b: &Self, res: &mut Self) {
        #[cfg(debug_assertions)]
        assert_eq!(self.columns, b.size().0);
        #[cfg(debug_assertions)]
        assert_eq!(res.size(), (self.rows, b.size().1));

        // let b_transposed = b.transpose();

        // let a = self.data.as_ptr();
        // let btm = b_transposed.data.as_ptr();
        // let bm = b.data.as_ptr();
        // let r = res.data.as_mut_ptr();

        let res_rows = res.rows as isize;
        let res_cols = res.columns as isize;
        let self_cols = self.columns as isize;
        // let b_cols = b.columns as isize;
        // let b_t_cols = b_transposed.columns as isize;

        // for i in 0..res_rows {
        //     for j in 0..res_cols {
        //         let mut sum = 0.0;
        //         for k in 0..self_cols {
        //             unsafe {
        //                 // sum += *a.offset(k + self_cols * i) * *btm.offset(k + b_t_cols * j);
        //                 sum += *a.offset(k + self_cols * i) * *bm.offset(j + b_cols * k);
        //             }
        //         }
        //         unsafe {
        //             *r.offset(j + res_cols * i) = sum;
        //         }
        //     }
        // }

        let mut a_buf = BUFFER_POOL.get_buffer(self.data.len());
        a_buf.set_default_queue(PRO_QUE.queue().clone());
        a_buf.write(&self.data).enq().unwrap();
        let mut b_buf = BUFFER_POOL.get_buffer(b.data.len());
        b_buf.set_default_queue(PRO_QUE.queue().clone());
        b_buf.write(&b.data).enq().unwrap();
        let mut c_buf = BUFFER_POOL.get_buffer(res.data.len());
        c_buf.set_default_queue(PRO_QUE.queue().clone());
        c_buf.write(&res.data).enq().unwrap();

        let kernel = PRO_QUE
            .kernel_builder("gemm")
            .arg(res_rows as i32)
            .arg(res_cols as i32)
            .arg(self_cols as i32)
            .arg(&a_buf)
            .arg(&b_buf)
            .arg(&c_buf)
            .global_work_size([res_cols, res_rows])
            .build()
            .unwrap();

        unsafe {
            kernel.enq().unwrap();
        }

        c_buf.set_default_queue(PRO_QUE.queue().clone());
        c_buf.read(&mut res.data).enq().unwrap();

        BUFFER_POOL.give_buffer(a_buf);
        BUFFER_POOL.give_buffer(b_buf);
        BUFFER_POOL.give_buffer(c_buf);
    }

    #[inline(always)]
    pub fn product(&self, b: &Self) -> Self {
        let mut res = Self::new(self.rows, b.size().1);
        self.product_into(b, &mut res);
        res
    }

    #[inline(always)]
    pub fn transpose(&self) -> Self {
        let mut res = Self::new(self.columns, self.rows);
        let rd = res.data.as_mut_ptr();
        let sd = self.data.as_ptr();
        let self_rows = self.rows as isize;
        let self_cols = self.columns as isize;
        let res_cols = res.columns as isize;
        for i in 0..self_rows {
            for j in 0..self_cols {
                unsafe {
                    *rd.offset(i + res_cols * j) = *sd.offset(j + self_cols * i);
                }
            }
        }
        res
    }

    #[inline(always)]
    pub fn transpose_into(&self, res: &mut Matrix) {
        #[cfg(debug_assertions)]
        assert_eq!((self.size().1, self.size().0), res.size());

        let rd = res.data.as_mut_ptr();
        let sd = self.data.as_ptr();
        let self_rows = self.rows as isize;
        let self_cols = self.columns as isize;
        let res_cols = res.columns as isize;
        for i in 0..self_rows {
            for j in 0..self_cols {
                unsafe {
                    *rd.offset(i + res_cols * j) = *sd.offset(j + self_cols * i);
                }
            }
        }
    }

    pub fn randomize(&mut self) {
        let mut rng = rand::thread_rng();
        for i in 0..self.data.len() {
            self.data[i] = rng.gen_range(0.0..=1.0);
        }
    }

    #[inline(always)]
    pub fn map(&mut self, func: &dyn Fn(f32) -> f32) {
        for i in 0..self.data.len() {
            self.data[i] = func(self.data[i]);
        }
    }

    #[inline(always)]
    pub fn map_into(&mut self, func: &dyn Fn(f32) -> f32, b: &mut Matrix) {
        #[cfg(debug_assertions)]
        assert_eq!(self.size(), b.size());

        for i in 0..self.data.len() {
            b.data[i] = func(self.data[i]);
        }
    }

    #[inline(always)]
    pub fn map_ret(&self, func: &dyn Fn(f32) -> f32) -> Self {
        let mut a = self.clone();
        a.map(func);
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! indx {
        ($a:expr, $cc:expr, $r:expr, $c:expr) => {
            $a[$c + $cc * $r]
        };
    }

    #[test]
    fn matrix_product() {
        let mut a = Matrix::new(2, 3);
        indx!(a.data, a.columns, 0, 0) = 3.0;
        indx!(a.data, a.columns, 0, 1) = 6.0;
        indx!(a.data, a.columns, 0, 2) = 7.0;
        indx!(a.data, a.columns, 1, 0) = 13.0;
        indx!(a.data, a.columns, 1, 1) = 16.0;
        indx!(a.data, a.columns, 1, 2) = 17.0;
        let mut b = Matrix::new(3, 2);
        indx!(b.data, b.columns, 0, 0) = 30.0;
        indx!(b.data, b.columns, 0, 1) = 60.0;
        indx!(b.data, b.columns, 1, 0) = 130.0;
        indx!(b.data, b.columns, 1, 1) = 160.0;
        indx!(b.data, b.columns, 2, 0) = 130.0;
        indx!(b.data, b.columns, 2, 1) = 160.0;
        let mut res = Matrix::new(a.size().0, b.size().1);
        a.product_into(&b, &mut res);
        let mut exp = Matrix::new(2, 2);
        indx!(exp.data, exp.columns, 0, 0) = 1780.0;
        indx!(exp.data, exp.columns, 0, 1) = 2260.0;
        indx!(exp.data, exp.columns, 1, 0) = 4680.0;
        indx!(exp.data, exp.columns, 1, 1) = 6060.0;
        assert_eq!(res.data, exp.data);
    }

    #[test]
    fn matrix_map() {
        let mut a = Matrix::new(2, 2);
        indx!(a.data, a.columns, 0, 0) = 1.0;
        indx!(a.data, a.columns, 0, 1) = 2.0;
        indx!(a.data, a.columns, 1, 0) = 3.0;
        indx!(a.data, a.columns, 1, 1) = 4.0;
        a.map(&|v| v * 2.0);
        let mut exp = Matrix::new(2, 2);
        indx!(exp.data, exp.columns, 0, 0) = 2.0;
        indx!(exp.data, exp.columns, 0, 1) = 4.0;
        indx!(exp.data, exp.columns, 1, 0) = 6.0;
        indx!(exp.data, exp.columns, 1, 1) = 8.0;
        assert_eq!(a.data, exp.data);
    }

    #[test]
    fn matrix_from_vec() {
        let a = Matrix::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let mut exp = Matrix::new(4, 1);
        indx!(exp.data, exp.columns, 0, 0) = 1.0;
        indx!(exp.data, exp.columns, 1, 0) = 2.0;
        indx!(exp.data, exp.columns, 2, 0) = 3.0;
        indx!(exp.data, exp.columns, 3, 0) = 4.0;
        assert_eq!(a.data, exp.data);
    }

    #[test]
    fn matrix_to_vec() {
        let mut a = Matrix::new(2, 3);
        indx!(a.data, a.columns, 0, 0) = 3.0;
        indx!(a.data, a.columns, 0, 1) = 6.0;
        indx!(a.data, a.columns, 0, 2) = 7.0;
        indx!(a.data, a.columns, 1, 0) = 13.0;
        indx!(a.data, a.columns, 1, 1) = 16.0;
        indx!(a.data, a.columns, 1, 2) = 17.0;
        assert_eq!(a.to_vec(), vec![3.0, 6.0, 7.0, 13.0, 16.0, 17.0]);
    }
}
