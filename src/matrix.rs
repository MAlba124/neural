use rand::Rng;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<Vec<f32>>,
    rows: usize,
    columns: usize,
}

impl Matrix {
    pub fn new(rows: usize, columns: usize) -> Self {
        Self {
            data: vec![vec![0.0; columns]; rows],
            rows,
            columns,
        }
    }

    pub fn from_vec(v: Vec<f32>) -> Self {
        let mut m = Self::new(v.len(), 1);
        for i in 0..m.rows {
            m.data[i][0] = v[i];
        }
        m
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.iter().flatten().map(|v| *v).collect::<Vec<f32>>()
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    pub fn multiply_scalar(&mut self, n: f32) {
        for i in 0..self.rows {
            for j in 0..self.columns {
                self.data[i][j] *= n;
            }
        }
    }

    pub fn add_scalar(&mut self, n: f32) {
        for i in 0..self.rows {
            for j in 0..self.columns {
                self.data[i][j] += n;
            }
        }
    }

    pub fn subtract_matrix(&self, b: &Self) -> Matrix {
        let mut res = Matrix::new(self.rows, self.columns);
        for i in 0..self.rows {
            for j in 0..self.columns {
                res.data[i][j] = self.data[i][j] - b.data[i][j];
            }
        }
        res
    }

    pub fn multiply_matrix(&mut self, m: &Self) {
        assert_eq!(self.size(), m.size());
        for i in 0..self.rows {
            for j in 0..self.columns {
                self.data[i][j] *= m.data[i][j];
            }
        }
    }

    pub fn multiply_matrix_ret(&self, m: &Self) -> Self {
        let mut a = self.clone();
        a.multiply_matrix(m);
        a
    }

    pub fn add_matrix(&mut self, m: &Self) {
        assert_eq!(m.size(), self.size());
        for i in 0..self.rows {
            for j in 0..self.columns {
                self.data[i][j] += m.data[i][j];
            }
        }
    }

    pub fn product(&self, m: &Self ) -> Self {
        assert_eq!(self.columns, m.size().0);
        let mut res = Self::new(self.rows, m.size().1);
        for i in 0..res.rows {
            for j in 0..res.columns {
                let mut sum = 0.0;
                let mut k = 0;
                while k < self.columns {
                        unsafe {
                            sum += self.data.get_unchecked(i).get_unchecked(k) * m.data.get_unchecked(k).get_unchecked(j);
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
        for i in 0..self.rows {
            for j in 0..self.columns {
                self.data[i][j] = rand::thread_rng().gen_range(0.0..=1.0);
            }
        }
    }

    pub fn map(&mut self, func: &dyn Fn(f32) -> f32) {
        for i in 0..self.rows {
            for j in 0..self.columns {
                self.data[i][j] = func(self.data[i][j]);
            }
        }
    }

    pub fn map_ret(&self, func: &dyn Fn(f32) -> f32) -> Self {
        let mut a = self.clone();
        a.map(func);
        a
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_product() {
        let mut a = Matrix::new(2, 3);
        a.data[0][0] = 3.0; a.data[0][1] = 6.0; a.data[0][2] = 7.0;
        a.data[1][0] = 13.0; a.data[1][1] = 16.0; a.data[1][2] = 17.0;
        let mut b = Matrix::new(3, 2);
        b.data[0][0] = 30.0; b.data[0][1] = 60.0;
        b.data[1][0] = 130.0; b.data[1][1] = 160.0;
        b.data[2][0] = 130.0; b.data[2][1] = 160.0;
        let res = a.product(&b);
        let mut exp = Matrix::new(2, 2);
        exp.data[0][0] = 1780.0; exp.data[0][1] = 2260.0;
        exp.data[1][0] = 4680.0; exp.data[1][1] = 6060.0;
        assert_eq!(res.data, exp.data);
    }

    #[test]
    fn matrix_map() {
        let mut a = Matrix::new(2, 2);
        a.data[0][0] = 1.0; a.data[0][1] = 2.0;
        a.data[1][0] = 3.0; a.data[1][1] = 4.0;
        a.map(&|v| v * 2.0);
        let mut exp = Matrix::new(2, 2);
        exp.data[0][0] = 2.0; exp.data[0][1] = 4.0;
        exp.data[1][0] = 6.0; exp.data[1][1] = 8.0;
        assert_eq!(a.data, exp.data);
    }

    #[test]
    fn matrix_from_vec() {
        let a = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let mut exp = Matrix::new(4, 1);
        exp.data[0][0] = 1.0;
        exp.data[1][0] = 2.0;
        exp.data[2][0] = 3.0;
        exp.data[3][0] = 4.0;
        assert_eq!(a.data, exp.data);
    }

    #[test]
    fn matrix_to_vec() {
        let mut a = Matrix::new(2, 3);
        a.data[0][0] = 3.0; a.data[0][1] = 6.0; a.data[0][2] = 7.0;
        a.data[1][0] = 13.0; a.data[1][1] = 16.0; a.data[1][2] = 17.0;
        assert_eq!(a.to_vec(), vec![3.0, 6.0, 7.0, 13.0, 16.0, 17.0]);
    }
}
