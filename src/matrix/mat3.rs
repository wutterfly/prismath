use crate::vec::Vec3;

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Mat3<T>([[T; 3]; 3]);

impl<T> Mat3<T> {
    #[inline]
    pub const fn from_array(value: [[T; 3]; 3]) -> Self {
        Self(value)
    }

    #[inline]
    pub fn from_vec(value: [Vec3<T>; 3]) -> Self {
        let [a, b, c] = value;

        let Vec3 {
            x: aa,
            y: ab,
            z: ac,
        } = a;

        let Vec3 {
            x: ba,
            y: bb,
            z: bc,
        } = b;

        let Vec3 {
            x: ca,
            y: cb,
            z: cc,
        } = c;

        Self([[aa, ab, ac], [ba, bb, bc], [ca, cb, cc]])
    }

    #[inline]
    pub fn to_arrays(self) -> [[T; 3]; 3] {
        self.0
    }

    #[inline]
    pub fn to_vecs(self) -> [Vec3<T>; 3] {
        let [a, b, c] = self.0;
        [Vec3::from(a), Vec3::from(b), Vec3::from(c)]
    }

    #[inline]
    #[must_use]
    #[rustfmt::skip]
    pub  fn transposed(self) -> Self {
        let [a, b, c] = self.0;

        let [aa, ab, ac] = a;
        let [ba, bb, bc] = b;
        let [ca, cb, cc] = c;

        Self([
            [aa, ba, ca],
            [ab, bb, cb],
            [ac, bc, cc],
        ])
    }

    #[inline]
    pub const fn transpose(&mut self) {
        let [a, b, c] = &mut self.0;

        let [_, ab, ac] = a;
        let [ba, _, bc] = b;
        let [ca, cb, _] = c;

        core::mem::swap(ab, ba);
        core::mem::swap(ac, ca);
        core::mem::swap(bc, cb);
    }
}

impl<T: Clone> Mat3<T> {
    #[inline]
    pub fn splat(t: T) -> Self {
        Self([
            [t.clone(), t.clone(), t.clone()],
            [t.clone(), t.clone(), t.clone()],
            [t.clone(), t.clone(), t],
        ])
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Mat3<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [a, b, c] = &self.0;
        f.write_fmt(format_args!("Mat3(\n{a:?}\n{b:?}\n{c:?}\n)"))
    }
}

impl<T> core::ops::Index<usize> for Mat3<T> {
    type Output = [T; 3];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> core::ops::IndexMut<usize> for Mat3<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T> From<[[T; 3]; 3]> for Mat3<T> {
    #[inline]
    fn from(value: [[T; 3]; 3]) -> Self {
        Self::from_array(value)
    }
}

impl<T> From<[Vec3<T>; 3]> for Mat3<T> {
    #[inline]
    fn from(value: [Vec3<T>; 3]) -> Self {
        Self::from_vec(value)
    }
}

mod add {
    use super::Mat3;
    use core::ops::Add;

    impl<T> Mat3<T>
    where
        T: Add<Output = T> + Copy,
    {
        #[inline]
        fn add_inner(&self, rhs: &Self) -> Self {
            let [a, b, c] = self.0;
            let [x, y, z] = rhs.0;

            let [aa, ab, ac] = a;
            let [ba, bb, bc] = b;
            let [ca, cb, cc] = c;

            let [xx, xy, xz] = x;
            let [yx, yy, yz] = y;
            let [zx, zy, zz] = z;

            Self([
                [aa + xx, ab + xy, ac + xz],
                [ba + yx, bb + yy, bc + yz],
                [ca + zx, cb + zy, cc + zz],
            ])
        }
    }

    impl<T> Add for Mat3<T>
    where
        T: Add<Output = T> + Copy,
    {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            self.add_inner(&rhs)
        }
    }

    impl<T> Add for &Mat3<T>
    where
        T: Add<Output = T> + Copy,
    {
        type Output = Mat3<T>;

        fn add(self, rhs: Self) -> Self::Output {
            self.add_inner(rhs)
        }
    }

    impl<T> Add<&Self> for Mat3<T>
    where
        T: Add<Output = T> + Copy,
    {
        type Output = Self;

        fn add(self, rhs: &Self) -> Self::Output {
            self.add_inner(rhs)
        }
    }

    impl<T> Add<Mat3<T>> for &Mat3<T>
    where
        T: Add<Output = T> + Copy,
    {
        type Output = Mat3<T>;

        fn add(self, rhs: Mat3<T>) -> Self::Output {
            self.add_inner(&rhs)
        }
    }
}

mod sub {
    use super::Mat3;
    use core::ops::Sub;

    impl<T> Mat3<T>
    where
        T: Sub<Output = T> + Copy,
    {
        #[inline]
        fn sub_inner(&self, rhs: &Self) -> Self {
            let [a, b, c] = self.0;
            let [x, y, z] = rhs.0;

            let [aa, ab, ac] = a;
            let [ba, bb, bc] = b;
            let [ca, cb, cc] = c;

            let [xx, xy, xz] = x;
            let [yx, yy, yz] = y;
            let [zx, zy, zz] = z;

            Self([
                [aa - xx, ab - xy, ac - xz],
                [ba - yx, bb - yy, bc - yz],
                [ca - zx, cb - zy, cc - zz],
            ])
        }
    }

    impl<T> Sub for Mat3<T>
    where
        T: Sub<Output = T> + Copy,
    {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            self.sub_inner(&rhs)
        }
    }

    impl<T> Sub for &Mat3<T>
    where
        T: Sub<Output = T> + Copy,
    {
        type Output = Mat3<T>;

        fn sub(self, rhs: Self) -> Self::Output {
            self.sub_inner(rhs)
        }
    }

    impl<T> Sub<&Self> for Mat3<T>
    where
        T: Sub<Output = T> + Copy,
    {
        type Output = Self;

        fn sub(self, rhs: &Self) -> Self::Output {
            self.sub_inner(rhs)
        }
    }

    impl<T> Sub<Mat3<T>> for &Mat3<T>
    where
        T: Sub<Output = T> + Copy,
    {
        type Output = Mat3<T>;

        fn sub(self, rhs: Mat3<T>) -> Self::Output {
            self.sub_inner(&rhs)
        }
    }
}

mod mul {

    use super::Mat3;
    use core::ops::Add;
    use core::ops::Mul;

    impl<T> Mat3<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        #[inline]
        fn mul_inner(&self, rhs: &Self) -> Self {
            let [a, b, c] = self.0;
            let [x, y, z] = rhs.0;

            let [aa, ab, ac] = a;
            let [ba, bb, bc] = b;
            let [ca, cb, cc] = c;

            let [xx, xy, xz] = x;
            let [yx, yy, yz] = y;
            let [zx, zy, zz] = z;

            Self([
                [
                    aa * xx + ab * yx + ac * zx,
                    aa * xy + ab * yy + ac * zy,
                    aa * xz + ab * yz + ac * zz,
                ],
                [
                    ba * xx + bb * yx + bc * zx,
                    ba * xy + bb * yy + bc * zy,
                    ba * xz + bb * yz + bc * zz,
                ],
                [
                    ca * xx + cb * yx + cc * zx,
                    ca * xy + cb * yy + cc * zy,
                    ca * xz + cb * yz + cc * zz,
                ],
            ])
        }
    }

    impl<T> Mul for Mat3<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            self.mul_inner(&rhs)
        }
    }

    impl<T> Mul for &Mat3<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Mat3<T>;

        fn mul(self, rhs: Self) -> Self::Output {
            self.mul_inner(rhs)
        }
    }

    impl<T> Mul<&Self> for Mat3<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Self;

        fn mul(self, rhs: &Self) -> Self::Output {
            self.mul_inner(rhs)
        }
    }

    impl<T> Mul<Mat3<T>> for &Mat3<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Mat3<T>;

        fn mul(self, rhs: Mat3<T>) -> Self::Output {
            self.mul_inner(&rhs)
        }
    }
}

mod scalar {
    use super::Mat3;
    use core::ops::Mul;

    impl<T> Mat3<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        #[inline]
        fn scalar_inner(&self, rhs: &T) -> Self {
            let [a, b, c] = self.0;
            let [aa, ab, ac] = a;
            let [ba, bb, bc] = b;
            let [ca, cb, cc] = c;

            Self([
                [aa * rhs, ab * rhs, ac * rhs],
                [ba * rhs, bb * rhs, bc * rhs],
                [ca * rhs, cb * rhs, cc * rhs],
            ])
        }
    }

    impl<T> Mul<T> for Mat3<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Self;

        fn mul(self, rhs: T) -> Self::Output {
            self.scalar_inner(&rhs)
        }
    }

    impl<T> Mul<T> for &Mat3<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Mat3<T>;

        fn mul(self, rhs: T) -> Self::Output {
            self.scalar_inner(&rhs)
        }
    }

    impl<T> Mul<&T> for Mat3<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Self;

        fn mul(self, rhs: &T) -> Self::Output {
            self.scalar_inner(rhs)
        }
    }

    impl<T> Mul<&T> for &Mat3<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Mat3<T>;

        fn mul(self, rhs: &T) -> Self::Output {
            self.scalar_inner(rhs)
        }
    }

    impl Mul<Mat3<Self>> for f32 {
        type Output = Mat3<Self>;

        fn mul(self, rhs: Mat3<Self>) -> Self::Output {
            rhs.scalar_inner(&self)
        }
    }

    impl Mul<Mat3<Self>> for u32 {
        type Output = Mat3<Self>;

        fn mul(self, rhs: Mat3<Self>) -> Self::Output {
            rhs.scalar_inner(&self)
        }
    }
}

mod vector {
    use crate::vec::Vec3;

    use super::Mat3;
    use core::ops::Add;
    use core::ops::Mul;

    impl<T> Mat3<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        #[inline]
        fn vector_mul(&self, rhs: &Vec3<T>) -> Vec3<T> {
            let [a, b, c] = self.0;

            let [aa, ab, ac] = a;
            let [ba, bb, bc] = b;
            let [ca, cb, cc] = c;

            let [x, y, z] = rhs.as_array();

            Vec3 {
                x: aa * x + ab * y + ac * z,
                y: ba * x + bb * y + bc * z,
                z: ca * x + cb * y + cc * z,
            }
        }
    }

    impl<T> Mul<Vec3<T>> for Mat3<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Vec3<T>;

        fn mul(self, rhs: Vec3<T>) -> Self::Output {
            self.vector_mul(&rhs)
        }
    }

    impl<T> Mul<Vec3<T>> for &Mat3<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Vec3<T>;

        fn mul(self, rhs: Vec3<T>) -> Self::Output {
            self.vector_mul(&rhs)
        }
    }

    impl<T> Mul<&Vec3<T>> for Mat3<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Vec3<T>;

        fn mul(self, rhs: &Vec3<T>) -> Self::Output {
            self.vector_mul(rhs)
        }
    }

    impl<T> Mul<&Vec3<T>> for &Mat3<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Vec3<T>;

        fn mul(self, rhs: &Vec3<T>) -> Self::Output {
            self.vector_mul(rhs)
        }
    }
}

impl Mat3<f32> {
    #[inline]
    #[must_use]
    pub const fn identity() -> Self {
        Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    }

    #[inline]
    #[must_use]
    pub const fn one() -> Self {
        Self([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    }

    #[inline]
    #[must_use]
    #[allow(clippy::similar_names)]
    pub const fn as_bytes(&self) -> [u8; std::mem::size_of::<Self>()] {
        let [a, b, c] = self.0;

        let [aa, ab, ac] = a;
        let [ba, bb, bc] = b;
        let [ca, cb, cc] = c;

        let [aa1, aa2, aa3, aa4] = aa.to_ne_bytes();
        let [ab1, ab2, ab3, ab4] = ab.to_ne_bytes();
        let [ac1, ac2, ac3, ac4] = ac.to_ne_bytes();

        let [ba1, ba2, ba3, ba4] = ba.to_ne_bytes();
        let [bb1, bb2, bb3, bb4] = bb.to_ne_bytes();
        let [bc1, bc2, bc3, bc4] = bc.to_ne_bytes();

        let [ca1, ca2, ca3, ca4] = ca.to_ne_bytes();
        let [cb1, cb2, cb3, cb4] = cb.to_ne_bytes();
        let [cc1, cc2, cc3, cc4] = cc.to_ne_bytes();

        #[rustfmt::skip]
        [
            aa1, aa2, aa3, aa4, ab1, ab2, ab3, ab4, ac1, ac2, ac3, ac4,
            ba1, ba2, ba3, ba4, bb1, bb2, bb3, bb4, bc1, bc2, bc3, bc4,
            ca1, ca2, ca3, ca4, cb1, cb2, cb3, cb4, cc1, cc2, cc3, cc4,
        ]
    }
}

// specialization
impl Mat3<f32> {
    #[must_use]
    pub fn multiply(&self, rhs: &Self) -> Self {
        let [a, b, c] = self.0;
        let [x, y, z] = rhs.0;

        let [aa, ab, ac] = a;
        let [ba, bb, bc] = b;
        let [ca, cb, cc] = c;

        let [xx, xy, xz] = x;
        let [yx, yy, yz] = y;
        let [zx, zy, zz] = z;

        Self([
            [
                ac.mul_add(zx, aa.mul_add(xx, ab * yx)),
                ac.mul_add(zy, aa.mul_add(xy, ab * yy)),
                ac.mul_add(zz, aa.mul_add(xz, ab * yz)),
            ],
            [
                bc.mul_add(zx, ba.mul_add(xx, bb * yx)),
                bc.mul_add(zy, ba.mul_add(xy, bb * yy)),
                bc.mul_add(zz, ba.mul_add(xz, bb * yz)),
            ],
            [
                cc.mul_add(zx, ca.mul_add(xx, cb * yx)),
                cc.mul_add(zy, ca.mul_add(xy, cb * yy)),
                cc.mul_add(zz, ca.mul_add(xz, cb * yz)),
            ],
        ])
    }

    #[must_use]
    pub fn scalar(&self, rhs: f32) -> Self {
        let [a, b, c] = self.0;
        let [aa, ab, ac] = a;
        let [ba, bb, bc] = b;
        let [ca, cb, cc] = c;

        Self([
            [aa * rhs, ab * rhs, ac * rhs],
            [ba * rhs, bb * rhs, bc * rhs],
            [ca * rhs, cb * rhs, cc * rhs],
        ])
    }
}

#[cfg(test)]
mod tests {

    use crate::vec::Vec3;

    use super::Mat3;

    #[test]
    fn test_mat3_add() {
        let mat1 = Mat3::<f32>::zero();
        let mat2 = Mat3::<f32>::one();

        let mat3 = mat1 + mat2;
        assert_eq!(mat3, Mat3::one());
    }

    #[test]
    fn test_mat3_sub() {
        let mat1 = Mat3::<f32>::one();
        let mat2 = Mat3::<f32>::one();

        let mat3 = mat1 - mat2;
        assert_eq!(mat3, Mat3::zero());
    }

    #[test]
    fn test_mat3_mul() {
        let mat1 = Mat3::<f32>::identity();
        let mat2 = Mat3::<f32>::one();

        let mat3 = mat1 * mat2;
        assert_eq!(mat3, Mat3::one());
    }

    #[test]
    fn test_mat3_transpose() {
        let mat = Mat3::<f32>::from([[0.0, 1.0, 2.0], [0.0, 0.0, 3.0], [5.0, -1.0, 0.0]]);

        let mut transposed = mat.transposed();
        assert_eq!(
            transposed,
            Mat3::<f32>::from([[0.0, 0.0, 5.0], [1.0, 0.0, -1.0], [2.0, 3.0, 0.0]])
        );

        transposed.transpose();
        assert_eq!(transposed, mat);
    }

    #[test]
    fn test_mat3_vector() {
        let mat1 = Mat3::<f32>::from_array([[5.0, 2.0, -3.0], [1.0, 0.0, 2.0], [-2.0, 1.0, 4.0]]);
        let vec1 = Vec3::new(2.0, 5.0, 8.0);

        let out = mat1 * vec1;

        assert_eq!(out, Vec3::new(-4.0, 18.0, 33.0));
    }
}
