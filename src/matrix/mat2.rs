use crate::vec::Vec2;

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Mat2<T>([[T; 2]; 2]);

impl<T> Mat2<T> {
    #[inline]
    pub const fn from_array(value: [[T; 2]; 2]) -> Self {
        Self(value)
    }

    #[inline]
    pub fn from_vec(value: [Vec2<T>; 2]) -> Self {
        let [a, b] = value;

        Self([[a.x, a.y], [b.x, b.y]])
    }

    #[inline]
    pub fn to_arrays(self) -> [[T; 2]; 2] {
        self.0
    }

    #[inline]
    pub fn to_vecs(self) -> [Vec2<T>; 2] {
        let [a, b] = self.0;
        [Vec2::from(a), Vec2::from(b)]
    }

    #[inline]
    #[must_use]
    pub fn transposed(self) -> Self {
        let [a, b] = self.0;

        let [aa, ab] = a;
        let [ba, bb] = b;
        Self([[aa, ba], [ab, bb]])
    }

    #[inline]
    pub const fn transpose(&mut self) {
        let [a, b] = &mut self.0;

        let [_, ab] = a;
        let [ba, _] = b;

        core::mem::swap(ab, ba);
    }
}

impl<T: Clone> Mat2<T> {
    #[inline]
    pub fn splat(t: T) -> Self {
        Self([[t.clone(), t.clone()], [t.clone(), t]])
    }
}

impl<T: core::fmt::Debug> core::fmt::Debug for Mat2<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let [a, b] = &self.0;
        f.write_fmt(format_args!("Mat2(\n{a:?}\n{b:?}\n)"))
    }
}

impl<T> core::ops::Index<usize> for Mat2<T> {
    type Output = [T; 2];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> core::ops::IndexMut<usize> for Mat2<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T> From<[[T; 2]; 2]> for Mat2<T> {
    #[inline]
    fn from(value: [[T; 2]; 2]) -> Self {
        Self::from_array(value)
    }
}

impl<T> From<[Vec2<T>; 2]> for Mat2<T> {
    #[inline]
    fn from(value: [Vec2<T>; 2]) -> Self {
        Self::from_vec(value)
    }
}

mod add {
    use super::Mat2;
    use core::ops::Add;

    impl<T> Mat2<T>
    where
        T: Add<Output = T> + Copy,
    {
        #[inline]
        fn add_inner(&self, rhs: &Self) -> Self {
            let [a, b] = self.0;
            let [x, y] = rhs.0;

            let [aa, ab] = a;
            let [ba, bb] = b;

            let [xx, xy] = x;
            let [yx, yy] = y;

            Self([[aa + xx, ab + xy], [ba + yx, bb + yy]])
        }
    }

    impl<T> Add for Mat2<T>
    where
        T: Add<Output = T> + Copy,
    {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            self.add_inner(&rhs)
        }
    }

    impl<T> Add for &Mat2<T>
    where
        T: Add<Output = T> + Copy,
    {
        type Output = Mat2<T>;

        fn add(self, rhs: Self) -> Self::Output {
            self.add_inner(rhs)
        }
    }

    impl<T> Add<&Self> for Mat2<T>
    where
        T: Add<Output = T> + Copy,
    {
        type Output = Self;

        fn add(self, rhs: &Self) -> Self::Output {
            self.add_inner(rhs)
        }
    }

    impl<T> Add<Mat2<T>> for &Mat2<T>
    where
        T: Add<Output = T> + Copy,
    {
        type Output = Mat2<T>;

        fn add(self, rhs: Mat2<T>) -> Self::Output {
            self.add_inner(&rhs)
        }
    }
}

mod sub {
    use super::Mat2;
    use core::ops::Sub;

    impl<T> Mat2<T>
    where
        T: Sub<Output = T> + Copy,
    {
        #[inline]
        fn sub_inner(&self, rhs: &Self) -> Self {
            let [a, b] = self.0;
            let [x, y] = rhs.0;

            let [aa, ab] = a;
            let [ba, bb] = b;

            let [xx, xy] = x;
            let [yx, yy] = y;

            Self([[aa - xx, ab - xy], [ba - yx, bb - yy]])
        }
    }

    impl<T> Sub for Mat2<T>
    where
        T: Sub<Output = T> + Copy,
    {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            self.sub_inner(&rhs)
        }
    }

    impl<T> Sub for &Mat2<T>
    where
        T: Sub<Output = T> + Copy,
    {
        type Output = Mat2<T>;

        fn sub(self, rhs: Self) -> Self::Output {
            self.sub_inner(rhs)
        }
    }

    impl<T> Sub<&Self> for Mat2<T>
    where
        T: Sub<Output = T> + Copy,
    {
        type Output = Self;

        fn sub(self, rhs: &Self) -> Self::Output {
            self.sub_inner(rhs)
        }
    }

    impl<T> Sub<Mat2<T>> for &Mat2<T>
    where
        T: Sub<Output = T> + Copy,
    {
        type Output = Mat2<T>;

        fn sub(self, rhs: Mat2<T>) -> Self::Output {
            self.sub_inner(&rhs)
        }
    }
}

mod mul {

    use super::Mat2;
    use core::ops::Add;
    use core::ops::Mul;

    impl<T> Mat2<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        #[inline]
        fn mul_inner(&self, rhs: &Self) -> Self {
            let [a, b] = self.0;
            let [x, y] = rhs.0;

            let [aa, ab] = a;
            let [ba, bb] = b;

            let [xx, xy] = x;
            let [yx, yy] = y;

            Self([
                [aa * xx + ab * yx, aa * xy + ab * yy],
                [ba * xx + bb * yx, ba * xy + bb * yy],
            ])
        }
    }

    impl<T> Mul for Mat2<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            self.mul_inner(&rhs)
        }
    }

    impl<T> Mul for &Mat2<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Mat2<T>;

        fn mul(self, rhs: Self) -> Self::Output {
            self.mul_inner(rhs)
        }
    }

    impl<T> Mul<&Self> for Mat2<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Self;

        fn mul(self, rhs: &Self) -> Self::Output {
            self.mul_inner(rhs)
        }
    }

    impl<T> Mul<Mat2<T>> for &Mat2<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Mat2<T>;

        fn mul(self, rhs: Mat2<T>) -> Self::Output {
            self.mul_inner(&rhs)
        }
    }
}

mod scalar {
    use super::Mat2;
    use core::ops::Mul;

    impl<T> Mat2<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        #[inline]
        fn scalar_inner(&self, rhs: &T) -> Self {
            let [a, b] = self.0;
            let [aa, ab] = a;
            let [ba, bb] = b;

            Self([[aa * rhs, ab * rhs], [ba * rhs, bb * rhs]])
        }
    }

    impl<T> Mul<T> for Mat2<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Self;

        fn mul(self, rhs: T) -> Self::Output {
            self.scalar_inner(&rhs)
        }
    }

    impl<T> Mul<T> for &Mat2<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Mat2<T>;

        fn mul(self, rhs: T) -> Self::Output {
            self.scalar_inner(&rhs)
        }
    }

    impl<T> Mul<&T> for Mat2<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Self;

        fn mul(self, rhs: &T) -> Self::Output {
            self.scalar_inner(rhs)
        }
    }

    impl<T> Mul<&T> for &Mat2<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Mat2<T>;

        fn mul(self, rhs: &T) -> Self::Output {
            self.scalar_inner(rhs)
        }
    }

    impl Mul<Mat2<Self>> for f32 {
        type Output = Mat2<Self>;

        fn mul(self, rhs: Mat2<Self>) -> Self::Output {
            rhs.scalar_inner(&self)
        }
    }

    impl Mul<Mat2<Self>> for u32 {
        type Output = Mat2<Self>;

        fn mul(self, rhs: Mat2<Self>) -> Self::Output {
            rhs.scalar_inner(&self)
        }
    }
}

mod vector {
    use crate::vec::Vec2;

    use super::Mat2;
    use core::ops::Add;
    use core::ops::Mul;

    impl<T> Mat2<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        #[inline]
        fn vector_mul(&self, rhs: &Vec2<T>) -> Vec2<T> {
            let [a, b] = self.0;

            let [aa, ab] = a;
            let [ba, bb] = b;

            let [x, y] = rhs.as_array();

            Vec2 {
                x: aa * x + ab * y,
                y: ba * x + bb * y,
            }
        }
    }

    impl<T> Mul<Vec2<T>> for Mat2<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Vec2<T>;

        fn mul(self, rhs: Vec2<T>) -> Self::Output {
            self.vector_mul(&rhs)
        }
    }

    impl<T> Mul<Vec2<T>> for &Mat2<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Vec2<T>;

        fn mul(self, rhs: Vec2<T>) -> Self::Output {
            self.vector_mul(&rhs)
        }
    }

    impl<T> Mul<&Vec2<T>> for Mat2<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Vec2<T>;

        fn mul(self, rhs: &Vec2<T>) -> Self::Output {
            self.vector_mul(rhs)
        }
    }

    impl<T> Mul<&Vec2<T>> for &Mat2<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Vec2<T>;

        fn mul(self, rhs: &Vec2<T>) -> Self::Output {
            self.vector_mul(rhs)
        }
    }
}

impl Mat2<f32> {
    #[inline]
    #[must_use]
    pub const fn identity() -> Self {
        Self([[1., 0.], [0., 1.]])
    }

    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self([[0., 0.], [0., 0.]])
    }

    #[inline]
    #[must_use]
    pub const fn one() -> Self {
        Self([[1., 1.], [1., 1.]])
    }

    #[inline]
    #[must_use]
    #[allow(clippy::similar_names)]
    pub const fn as_bytes(&self) -> [u8; core::mem::size_of::<Self>()] {
        let [a, b] = self.0;

        let [aa, ab] = a;
        let [ba, bb] = b;

        let [aa1, aa2, aa3, aa4] = aa.to_ne_bytes();
        let [ab1, ab2, ab3, ab4] = ab.to_ne_bytes();

        let [ba1, ba2, ba3, ba4] = ba.to_ne_bytes();
        let [bb1, bb2, bb3, bb4] = bb.to_ne_bytes();

        #[rustfmt::skip]
        [
            aa1, aa2, aa3, aa4, ab1, ab2, ab3, ab4,
            ba1, ba2, ba3, ba4, bb1, bb2, bb3, bb4,
        ]
    }
}

// specialization
impl Mat2<f32> {
    #[must_use]
    pub fn multiply(&self, rhs: &Self) -> Self {
        let [a, b] = self.0;
        let [x, y] = rhs.0;

        let [aa, ab] = a;
        let [ba, bb] = b;

        let [xx, xy] = x;
        let [yx, yy] = y;

        Self([
            [aa.mul_add(xx, ab * yx), aa.mul_add(xy, ab * yy)],
            [ba.mul_add(xx, bb * yx), ba.mul_add(xy, bb * yy)],
        ])
    }

    #[must_use]
    pub fn scalar(&self, rhs: f32) -> Self {
        let [a, b] = self.0;
        let [aa, ab] = a;
        let [ba, bb] = b;

        Self([[aa * rhs, ab * rhs], [ba * rhs, bb * rhs]])
    }
}

#[cfg(test)]
mod tests {

    use crate::vec::Vec2;

    use super::Mat2;

    #[test]
    fn test_mat2_add() {
        let mat1 = Mat2::<f32>::zero();
        let mat2 = Mat2::<f32>::one();

        let out = mat1 + mat2;
        assert_eq!(out, Mat2::one());
    }

    #[test]
    fn test_mat2_sub() {
        let mat1 = Mat2::<f32>::one();
        let mat2 = Mat2::<f32>::one();

        let out = mat1 - mat2;
        assert_eq!(out, Mat2::zero());
    }

    #[test]
    fn test_mat2_mul() {
        let mat1 = Mat2::<f32>::identity();
        let mat2 = Mat2::<f32>::one();

        let out = mat1 * mat2;
        assert_eq!(out, Mat2::one());
    }

    #[test]
    fn test_mat2_transpose() {
        let mat = Mat2::<f32>::from([[0., 1.], [0., 0.]]);

        let mut transposed = mat.transposed();
        assert_eq!(transposed, Mat2::<f32>::from([[0., 0.], [1., 0.]]));

        transposed.transpose();
        assert_eq!(transposed, mat);
    }

    #[test]
    fn test_mat2_vector() {
        let mat1 = Mat2::<f32>::from_array([[5.0, 2.0], [1.0, 0.0]]);
        let vec1 = Vec2::new(2.0, 5.0);

        let out = mat1 * vec1;

        assert_eq!(out, Vec2::new(20.0, 2.0));
    }
}
