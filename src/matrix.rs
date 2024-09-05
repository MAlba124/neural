use std::{ptr::null, sync::Arc};

use cudarc::{
    cublas::{sys::lib, CudaBlas},
    curand::CudaRng,
    driver::{
        CudaDevice, CudaSlice, DevicePtr, DevicePtrMut, DeviceSlice, LaunchAsync, LaunchConfig,
    },
    nvrtc::compile_ptx,
};

lazy_static::lazy_static! {
    static ref CUDA_DEV: Arc<CudaDevice> = CudaDevice::new(0).unwrap();
    static ref CUBLAS: CudaBlas = CudaBlas::new(Arc::clone(&CUDA_DEV)).unwrap();
}

pub fn init() {
    println!("[matrix::init] Compiling kernel `mat_add_scalar`...");
    let add_scal_kernel = compile_ptx(include_str!("../kernels/mat_add_scalar.cu")).unwrap();
    println!("[matrix::init] Loading kernel `mat_add_scalar`...");
    CUDA_DEV
        .load_ptx(add_scal_kernel, "mat_add_scalar", &["mat_add_scalar"])
        .unwrap();

    println!("[matrix::init] Compiling kernel `mat_sub_mat`...");
    let sub_mat_kernel = compile_ptx(include_str!("../kernels/mat_sub_mat.cu")).unwrap();
    println!("[matrix::init] Loading kernel `mat_sub_mat`...");
    CUDA_DEV
        .load_ptx(sub_mat_kernel, "mat_sub_mat", &["mat_sub_mat"])
        .unwrap();

    println!("[matrix::init] Compiling kernel `sigmoid`...");
    let sigmoid_kernel = compile_ptx(include_str!("../kernels/sigmoid.cu")).unwrap();
    println!("[matrix::init] Loading kernel `sigmoid`...");
    CUDA_DEV
        .load_ptx(sigmoid_kernel, "sigmoid", &["sigmoid"])
        .unwrap();

    println!("[matrix::init] Compiling kernel `dsigmoid`...");
    let dsigmoid_kernel = compile_ptx(include_str!("../kernels/dsigmoid.cu")).unwrap();
    println!("[matrix::init] Loading kernel `dsigmoid`...");
    CUDA_DEV
        .load_ptx(dsigmoid_kernel, "dsigmoid", &["dsigmoid"])
        .unwrap();
}

#[derive(Debug, Clone)]
#[repr(align(64))]
pub struct Matrix {
    pub cudata: CudaSlice<f32>,
    rows: usize,
    columns: usize,
}

impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Self {
        let cudata = CUDA_DEV.alloc_zeros(columns * rows).unwrap();

        CUDA_DEV.synchronize().unwrap();

        Self {
            cudata,
            rows,
            columns,
        }
    }

    pub fn from_slice(v: &[f32]) -> Self {
        let cudata = CUDA_DEV.htod_copy(v.to_vec()).unwrap();

        CUDA_DEV.synchronize().unwrap();

        Self {
            cudata,
            rows: v.len(),
            columns: 1,
        }
    }

    /// From slice column major
    pub fn from_slice_cm(v: &[f32], rows: usize, columns: usize) -> Self {
        assert_eq!(v.len(), rows * columns);

        let cudata = CUDA_DEV.htod_copy(v.to_vec()).unwrap();

        CUDA_DEV.synchronize().unwrap();

        Self {
            cudata,
            rows: rows,
            columns: columns,
        }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        CUDA_DEV.dtoh_sync_copy(&self.cudata).unwrap()
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    pub fn multiply_scalar(&mut self, n: f32) {
        unsafe {
            lib()
                .cublasSscal_v2(
                    CUBLAS.handle().clone(),
                    self.cudata.len() as i32,
                    (&n) as *const f32 as *const _,
                    *self.cudata.device_ptr_mut() as *mut _,
                    1,
                )
                .result()
                .unwrap();
        }

        CUDA_DEV.synchronize().unwrap();
    }

    pub fn add_scalar(&mut self, n: f32) {
        let f = CUDA_DEV
            .get_func("mat_add_scalar", "mat_add_scalar")
            .unwrap();

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (self.cudata.len() as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            f.launch(cfg, (&self.cudata, n)).unwrap();
        }

        CUDA_DEV.synchronize().unwrap();
    }

    pub fn subtract_matrix(&self, b: &Self) -> Matrix {
        #[cfg(debug_assertions)]
        assert_eq!(self.size(), b.size());

        let f = CUDA_DEV.get_func("mat_sub_mat", "mat_sub_mat").unwrap();

        let mut r = Matrix::new(self.rows, self.columns);

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (self.cudata.len() as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            f.launch(cfg, (&self.cudata, &b.cudata, &mut r.cudata))
                .unwrap();
        }

        CUDA_DEV.synchronize().unwrap();

        r
    }

    pub fn multiply_matrix(&mut self, b: &Self) {
        #[cfg(debug_assertions)]
        assert_eq!(self.size(), b.size());

        unsafe {
            lib()
                .cublasSdgmm(
                    CUBLAS.handle().clone(),
                    cudarc::cublas::sys::cublasSideMode_t::CUBLAS_SIDE_LEFT,
                    self.rows as i32,
                    self.columns as i32,
                    *self.cudata.device_ptr() as *const _,
                    self.rows as i32,
                    *b.cudata.device_ptr() as *const _,
                    self.rows as i32,
                    *self.cudata.device_ptr_mut() as *mut _,
                    self.rows as i32,
                )
                .result()
                .unwrap();
        }

        CUDA_DEV.synchronize().unwrap();
    }

    pub fn multiply_matrix_ret(&self, b: &Self) -> Self {
        let mut a = self.clone();
        a.multiply_matrix(b);
        a
    }

    pub fn add_matrix(&mut self, b: &Self) {
        #[cfg(debug_assertions)]
        assert_eq!(b.size(), self.size());

        unsafe {
            lib()
                .cublasSaxpy_v2(
                    CUBLAS.handle().clone(),
                    self.cudata.len() as i32,
                    &1.0f32 as *const f32,
                    *b.cudata.device_ptr() as *const _,
                    1,
                    *self.cudata.device_ptr_mut() as *mut _,
                    1,
                )
                .result()
                .unwrap();
        }

        CUDA_DEV.synchronize().unwrap();
    }

    pub fn product_into(&self, b: &Self, res: &mut Self) {
        #[cfg(debug_assertions)]
        assert_eq!(self.columns, b.size().0);
        #[cfg(debug_assertions)]
        assert_eq!(res.size(), (self.rows, b.size().1));

        unsafe {
            lib()
                .cublasSgemm_v2(
                    CUBLAS.handle().clone(),
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    self.rows as i32,
                    b.columns as i32,
                    self.columns as i32,
                    &1.0f32 as *const f32,
                    *self.cudata.device_ptr() as *const _,
                    self.rows as i32,
                    *b.cudata.device_ptr() as *const _,
                    b.rows as i32,
                    &1.0f32 as *const f32,
                    *res.cudata.device_ptr() as *mut _,
                    res.rows as i32,
                )
                .result()
                .unwrap();
        }

        CUDA_DEV.synchronize().unwrap();
    }

    pub fn product(&self, b: &Self) -> Self {
        let mut res = Self::new(self.rows, b.size().1);
        self.product_into(b, &mut res);
        res
    }

    pub fn transpose_into(&self, res: &mut Matrix) {
        #[cfg(debug_assertions)]
        assert_eq!((self.size().1, self.size().0), res.size());

        unsafe {
            lib()
                .cublasSgeam(
                    CUBLAS.handle().clone(),
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                    cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                    self.columns as i32,
                    self.rows as i32,
                    (&1.0f32) as *const f32,
                    *self.cudata.device_ptr() as *const _,
                    self.rows as i32,
                    (&0.0f32) as *const f32,
                    null(),
                    res.rows as i32,
                    *res.cudata.device_ptr_mut() as *mut _,
                    res.rows as i32,
                )
                .result()
                .unwrap();
        }

        CUDA_DEV.synchronize().unwrap();
    }

    pub fn randomize(&mut self) {
        let rng = CudaRng::new(0, Arc::clone(&CUDA_DEV)).unwrap();
        rng.fill_with_uniform(&mut self.cudata).unwrap();

        CUDA_DEV.synchronize().unwrap();
    }

    pub fn sigmoid(&mut self) {
        let f = CUDA_DEV.get_func("sigmoid", "sigmoid").unwrap();

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (self.cudata.len() as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            f.launch(cfg, (&mut self.cudata,)).unwrap();
        }

        CUDA_DEV.synchronize().unwrap();
    }

    pub fn dsigmoid(&mut self, b: &mut Matrix) {
        let f = CUDA_DEV.get_func("dsigmoid", "dsigmoid").unwrap();

        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (self.cudata.len() as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            f.launch(cfg, (&self.cudata, &mut b.cudata)).unwrap();
        }

        CUDA_DEV.synchronize().unwrap();
    }
}
