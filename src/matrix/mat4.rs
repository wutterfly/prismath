use crate::vec::{Vec3, Vec4};

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Mat4<T>([[T; 4]; 4]);

impl<T> Mat4<T> {
    #[inline]
    pub const fn from_array(value: [[T; 4]; 4]) -> Self {
        Self(value)
    }

    #[inline]
    pub fn from_vec(value: [Vec4<T>; 4]) -> Self {
        let [a, b, c, d] = value;
        Self([
            [a.x, a.y, a.z, a.w],
            [b.x, b.y, b.z, b.w],
            [c.x, c.y, c.z, c.w],
            [d.x, d.y, d.z, d.w],
        ])
    }

    #[inline]
    pub fn to_arrays(self) -> [[T; 4]; 4] {
        self.0
    }

    #[inline]
    pub fn to_vecs(self) -> [Vec4<T>; 4] {
        let [a, b, c, d] = self.0;
        [Vec4::from(a), Vec4::from(b), Vec4::from(c), Vec4::from(d)]
    }

    #[inline]
    #[must_use]
    #[rustfmt::skip]
    pub  fn transposed(self) -> Self {
        let [a, b, c, d] = self.0;

        let [aa, ab, ac, ad] = a;
        let [ba, bb, bc, bd] = b;
        let [ca, cb, cc, cd] = c;
        let [da, db, dc, dd] = d;

        Self([
            [aa, ba, ca, da],
            [ab, bb, cb, db],
            [ac, bc, cc, dc],
            [ad, bd, cd, dd],
        ])
    }

    #[inline]
    pub const fn transpose(&mut self) {
        let [a, b, c, d] = &mut self.0;

        let [_, ab, ac, ad] = a;
        let [ba, _, bc, bd] = b;
        let [ca, cb, _, cd] = c;
        let [da, db, dc, _] = d;

        core::mem::swap(ab, ba);
        core::mem::swap(ac, ca);
        core::mem::swap(bc, cb);
        core::mem::swap(ad, da);
        core::mem::swap(bd, db);
        core::mem::swap(cd, dc);
    }
}

impl<T: Clone> Mat4<T> {
    #[inline]
    pub fn splat(t: T) -> Self {
        Self([
            [t.clone(), t.clone(), t.clone(), t.clone()],
            [t.clone(), t.clone(), t.clone(), t.clone()],
            [t.clone(), t.clone(), t.clone(), t.clone()],
            [t.clone(), t.clone(), t.clone(), t],
        ])
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Mat4<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [a, b, c, d] = &self.0;
        f.write_fmt(format_args!("Mat4(\n{a:?}\n{b:?}\n{c:?}\n{d:?})"))
    }
}

impl<T> core::ops::Index<usize> for Mat4<T> {
    type Output = [T; 4];

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> core::ops::IndexMut<usize> for Mat4<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T> From<[[T; 4]; 4]> for Mat4<T> {
    #[inline]
    fn from(value: [[T; 4]; 4]) -> Self {
        Self::from_array(value)
    }
}

impl<T> From<[Vec4<T>; 4]> for Mat4<T> {
    #[inline]
    fn from(value: [Vec4<T>; 4]) -> Self {
        Self::from_vec(value)
    }
}

mod add {
    use super::Mat4;
    use core::ops::Add;

    impl<T> Mat4<T>
    where
        T: Add<Output = T> + Copy,
    {
        #[inline]
        fn add_inner(&self, rhs: &Self) -> Self {
            let [a, b, c, d] = self.0;
            let [x, y, z, w] = rhs.0;

            let [aa, ab, ac, ad] = a;
            let [ba, bb, bc, bd] = b;
            let [ca, cb, cc, cd] = c;
            let [da, db, dc, dd] = d;

            let [xx, xy, xz, xw] = x;
            let [yx, yy, yz, yw] = y;
            let [zx, zy, zz, zw] = z;
            let [wx, wy, wz, ww] = w;

            Self([
                [aa + xx, ab + xy, ac + xz, ad + xw],
                [ba + yx, bb + yy, bc + yz, bd + yw],
                [ca + zx, cb + zy, cc + zz, cd + zw],
                [da + wx, db + wy, dc + wz, dd + ww],
            ])
        }
    }

    impl<T> Add for Mat4<T>
    where
        T: Add<Output = T> + Copy,
    {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            self.add_inner(&rhs)
        }
    }

    impl<T> Add for &Mat4<T>
    where
        T: Add<Output = T> + Copy,
    {
        type Output = Mat4<T>;

        fn add(self, rhs: Self) -> Self::Output {
            self.add_inner(rhs)
        }
    }

    impl<T> Add<&Self> for Mat4<T>
    where
        T: Add<Output = T> + Copy,
    {
        type Output = Self;

        fn add(self, rhs: &Self) -> Self::Output {
            self.add_inner(rhs)
        }
    }

    impl<T> Add<Mat4<T>> for &Mat4<T>
    where
        T: Add<Output = T> + Copy,
    {
        type Output = Mat4<T>;

        fn add(self, rhs: Mat4<T>) -> Self::Output {
            self.add_inner(&rhs)
        }
    }
}

mod sub {
    use super::Mat4;
    use core::ops::Sub;

    impl<T> Mat4<T>
    where
        T: Sub<Output = T> + Copy,
    {
        #[inline]
        fn sub_inner(&self, rhs: &Self) -> Self {
            let [a, b, c, d] = self.0;
            let [x, y, z, w] = rhs.0;

            let [aa, ab, ac, ad] = a;
            let [ba, bb, bc, bd] = b;
            let [ca, cb, cc, cd] = c;
            let [da, db, dc, dd] = d;

            let [xx, xy, xz, xw] = x;
            let [yx, yy, yz, yw] = y;
            let [zx, zy, zz, zw] = z;
            let [wx, wy, wz, ww] = w;

            Self([
                [aa - xx, ab - xy, ac - xz, ad - xw],
                [ba - yx, bb - yy, bc - yz, bd - yw],
                [ca - zx, cb - zy, cc - zz, cd - zw],
                [da - wx, db - wy, dc - wz, dd - ww],
            ])
        }
    }

    impl<T> Sub for Mat4<T>
    where
        T: Sub<Output = T> + Copy,
    {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            self.sub_inner(&rhs)
        }
    }

    impl<T> Sub for &Mat4<T>
    where
        T: Sub<Output = T> + Copy,
    {
        type Output = Mat4<T>;

        fn sub(self, rhs: Self) -> Self::Output {
            self.sub_inner(rhs)
        }
    }

    impl<T> Sub<&Self> for Mat4<T>
    where
        T: Sub<Output = T> + Copy,
    {
        type Output = Self;

        fn sub(self, rhs: &Self) -> Self::Output {
            self.sub_inner(rhs)
        }
    }

    impl<T> Sub<Mat4<T>> for &Mat4<T>
    where
        T: Sub<Output = T> + Copy,
    {
        type Output = Mat4<T>;

        fn sub(self, rhs: Mat4<T>) -> Self::Output {
            self.sub_inner(&rhs)
        }
    }
}

mod mul {

    use super::Mat4;
    use core::ops::Add;
    use core::ops::Mul;

    impl<T> Mat4<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        #[inline]
        fn mul_inner(&self, rhs: &Self) -> Self {
            let [a, b, c, d] = self.0;
            let [x, y, z, w] = rhs.0;

            let [aa, ab, ac, ad] = a;
            let [ba, bb, bc, bd] = b;
            let [ca, cb, cc, cd] = c;
            let [da, db, dc, dd] = d;

            let [xx, xy, xz, xw] = x;
            let [yx, yy, yz, yw] = y;
            let [zx, zy, zz, zw] = z;
            let [wx, wy, wz, ww] = w;

            Self([
                [
                    aa * xx + ab * yx + ac * zx + ad * wx,
                    aa * xy + ab * yy + ac * zy + ad * wy,
                    aa * xz + ab * yz + ac * zz + ad * wz,
                    aa * xw + ab * yw + ac * zw + ad * ww,
                ],
                [
                    ba * xx + bb * yx + bc * zx + bd * wx,
                    ba * xy + bb * yy + bc * zy + bd * wy,
                    ba * xz + bb * yz + bc * zz + bd * wz,
                    ba * xw + bb * yw + bc * zw * bd * ww,
                ],
                [
                    ca * xx + cb * yx + cc * zx + cd * wx,
                    ca * xy + cb * yy + cc * zy + cd * wy,
                    ca * xz + cb * yz + cc * zz + cd * wz,
                    ca * xw + cb * yw + cc * zw + cd * ww,
                ],
                [
                    da * xx + db * xy + dc * zx + dd * wx,
                    da * xy + db * yy + dc * zy + dd * wy,
                    da * xz + db * zy + dc * zz + dd * wz,
                    da * xw + db * wy + dc * zw + dd * ww,
                ],
            ])
        }
    }

    impl<T> Mul for Mat4<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            self.mul_inner(&rhs)
        }
    }

    impl<T> Mul for &Mat4<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Mat4<T>;

        fn mul(self, rhs: Self) -> Self::Output {
            self.mul_inner(rhs)
        }
    }

    impl<T> Mul<&Self> for Mat4<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Self;

        fn mul(self, rhs: &Self) -> Self::Output {
            self.mul_inner(rhs)
        }
    }

    impl<T> Mul<Mat4<T>> for &Mat4<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Mat4<T>;

        fn mul(self, rhs: Mat4<T>) -> Self::Output {
            self.mul_inner(&rhs)
        }
    }
}

mod scalar {
    use super::Mat4;
    use core::ops::Mul;

    impl<T> Mat4<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        #[inline]
        fn scalar_inner(&self, rhs: &T) -> Self {
            let [a, b, c, d] = self.0;

            let [aa, ab, ac, ad] = a;
            let [ba, bb, bc, bd] = b;
            let [ca, cb, cc, cd] = c;
            let [da, db, dc, dd] = d;

            Self([
                [aa * rhs, ab * rhs, ac * rhs, ad * rhs],
                [ba * rhs, bb * rhs, bc * rhs, bd * rhs],
                [ca * rhs, cb * rhs, cc * rhs, cd * rhs],
                [da * rhs, db * rhs, dc * rhs, dd * rhs],
            ])
        }
    }

    impl<T> Mul<T> for Mat4<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Self;

        fn mul(self, rhs: T) -> Self::Output {
            self.scalar_inner(&rhs)
        }
    }

    impl<T> Mul<T> for &Mat4<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Mat4<T>;

        fn mul(self, rhs: T) -> Self::Output {
            self.scalar_inner(&rhs)
        }
    }

    impl<T> Mul<&T> for Mat4<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Self;

        fn mul(self, rhs: &T) -> Self::Output {
            self.scalar_inner(rhs)
        }
    }

    impl<T> Mul<&T> for &Mat4<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Mat4<T>;

        fn mul(self, rhs: &T) -> Self::Output {
            self.scalar_inner(rhs)
        }
    }

    impl Mul<Mat4<Self>> for f32 {
        type Output = Mat4<Self>;

        fn mul(self, rhs: Mat4<Self>) -> Self::Output {
            rhs.scalar_inner(&self)
        }
    }

    impl Mul<Mat4<Self>> for u32 {
        type Output = Mat4<Self>;

        fn mul(self, rhs: Mat4<Self>) -> Self::Output {
            rhs.scalar_inner(&self)
        }
    }
}

mod vector {
    use crate::vec::Vec4;

    use super::Mat4;
    use core::ops::Add;
    use core::ops::Mul;

    impl<T> Mat4<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        #[inline]
        fn vector_mul(&self, rhs: &Vec4<T>) -> Vec4<T> {
            let [a, b, c, d] = self.0;

            let [aa, ab, ac, ad] = a;
            let [ba, bb, bc, bd] = b;
            let [ca, cb, cc, cd] = c;
            let [da, db, dc, dd] = d;

            let [x, y, z, w] = rhs.as_array();

            Vec4 {
                x: aa * x + ab * y + ac * z + ad * w,
                y: ba * x + bb * y + bc * z + bd * w,
                z: ca * x + cb * y + cc * z + cd * w,
                w: da * x + db * y + dc * z + dd * w,
            }
        }
    }

    impl<T> Mul<Vec4<T>> for Mat4<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Vec4<T>;

        fn mul(self, rhs: Vec4<T>) -> Self::Output {
            self.vector_mul(&rhs)
        }
    }

    impl<T> Mul<Vec4<T>> for &Mat4<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Vec4<T>;

        fn mul(self, rhs: Vec4<T>) -> Self::Output {
            self.vector_mul(&rhs)
        }
    }

    impl<T> Mul<&Vec4<T>> for Mat4<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Vec4<T>;

        fn mul(self, rhs: &Vec4<T>) -> Self::Output {
            self.vector_mul(rhs)
        }
    }

    impl<T> Mul<&Vec4<T>> for &Mat4<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy,
    {
        type Output = Vec4<T>;

        fn mul(self, rhs: &Vec4<T>) -> Self::Output {
            self.vector_mul(rhs)
        }
    }
}

impl Mat4<f32> {
    #[inline]
    #[must_use]
    pub const fn identity() -> Self {
        Self([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
    }

    #[inline]
    #[must_use]
    pub const fn one() -> Self {
        Self([
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ])
    }

    #[inline]
    #[must_use]
    pub const fn translation(position: Vec3<f32>) -> Self {
        Self([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [position.x, position.y, position.z, 1.0],
        ])
    }

    #[inline]
    #[must_use]
    pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let lr = 1.0 / (left - right);
        let bt = 1.0 / (bottom - top);
        let nf = 1.0 / (near - far);

        #[rustfmt::skip]
        Self([
            [-2.0 * lr, 0.0, 0.0, 0.0],
            [0.0, -2.0 * bt, 0.0, 0.0],
            [0.0, 0.0, 2.0 * nf, 0.0],
            [(left + right) * lr, (top + bottom) * bt, (far + near) * nf, 1.0,],
        ])
    }

    #[inline]
    #[must_use]
    #[rustfmt::skip]
    pub fn perspective(fov: f32, aspect: f32, near: f32, far: f32) -> Self {
        let half_fov = (0.5 * fov).tan();

        Self([
            [1.0 / (aspect * half_fov), 0.0, 0.0, 0.0],
            [0.0, 1.0 / half_fov, 0.0, 0.0],
            [0.0, 0.0, -((far + near) / (far - near)), -((2.0 * far * near) / (far * near))],
            [0.0, 0.0, -1.0, 0.0],
        ])
    }

    #[inline]
    #[must_use]
    pub fn look_at(pos: Vec3<f32>, target: Vec3<f32>, up: Vec3<f32>) -> Self {
        let z_axis = (target - pos).normalized();

        let x_axis = z_axis.cross(&up).normalized();
        let y_axis = x_axis.cross(&z_axis);

        Self([
            [x_axis.x, y_axis.x, -z_axis.x, 0.0],
            [x_axis.y, y_axis.y, -z_axis.y, 0.0],
            [x_axis.z, y_axis.z, -z_axis.z, 0.0],
            [-x_axis.dot(&pos), -y_axis.dot(&pos), z_axis.dot(&pos), 1.0],
        ])
    }

    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        let [a, b, c, d] = self.0;

        let [aa, ab, ac, ad] = a;
        let [ba, bb, bc, bd] = b;
        let [ca, cb, cc, cd] = c;
        let [da, db, dc, dd] = d;

        let t0 = cc * dd;
        let t1 = dc * cd;
        let t2 = bc * dd;
        let t3 = dc * bd;
        let t4 = bc * cd;
        let t5 = cc * bd;
        let t6 = ac * dd;
        let t7 = dc * ad;
        let t8 = ac * cd;
        let t9 = cc * ad;
        let t10 = ac * bd;
        let t11 = bc * ad;
        let t12 = ca * db;
        let t13 = da * cb;
        let t14 = ba * db;
        let t15 = da * bb;
        let t16 = ba * cb;
        let t17 = ca * bb;
        let t18 = aa * db;
        let t19 = da * ab;
        let t20 = aa * cb;
        let t21 = ca * ab;
        let t22 = aa * bb;
        let t23 = ba * ab;

        let mut o = Self::zero();

        o[0][0] = t4.mul_add(db, t0.mul_add(bb, t3 * cb)) - t5.mul_add(db, t1.mul_add(bb, t2 * cb));
        o[0][1] = t9.mul_add(db, t1.mul_add(ab, t6 * cb)) - t8.mul_add(db, t0.mul_add(ab, t7 * cb));
        o[0][2] =
            t10.mul_add(db, t2.mul_add(ab, t7 * bb)) - t11.mul_add(db, t3.mul_add(ab, t6 * bb));
        o[0][3] =
            t11.mul_add(cb, t5.mul_add(ab, t8 * bb)) - t10.mul_add(cb, t4.mul_add(ab, t9 * bb));

        let d = 1.0
            / da.mul_add(
                o[0][3],
                ca.mul_add(o[0][2], aa.mul_add(o[0][0], ba * o[0][1])),
            );

        o[0][0] *= d;
        o[0][1] *= d;
        o[0][2] *= d;
        o[0][3] *= d;

        o[1][0] =
            d * (t5.mul_add(da, t1.mul_add(ba, t2 * ca)) - t4.mul_add(da, t0.mul_add(ba, t3 * ca)));
        o[1][1] =
            d * (t8.mul_add(da, t0.mul_add(aa, t7 * ca)) - t9.mul_add(da, t1.mul_add(aa, t6 * ca)));
        o[1][2] = d
            * (t11.mul_add(da, t3.mul_add(aa, t6 * ba)) - t10.mul_add(da, t2.mul_add(aa, t7 * ba)));
        o[1][3] = d
            * (t10.mul_add(ca, t4.mul_add(aa, t9 * ba)) - t11.mul_add(ca, t5.mul_add(aa, t8 * ba)));
        o[2][0] = d
            * (t16.mul_add(dd, t12.mul_add(bd, t15 * cd))
                - t17.mul_add(dd, t13.mul_add(bd, t14 * cd)));
        o[2][1] = d
            * (t21.mul_add(dd, t13.mul_add(ad, t18 * cd))
                - t20.mul_add(dd, t12.mul_add(ad, t19 * cd)));
        o[2][2] = d
            * (t22.mul_add(dd, t14.mul_add(ad, t19 * bd))
                - t23.mul_add(dd, t15.mul_add(ad, t18 * bd)));
        o[2][3] = d
            * (t23.mul_add(cd, t17.mul_add(ad, t20 * bd))
                - t22.mul_add(cd, t16.mul_add(ad, t21 * bd)));
        o[3][0] = d
            * (t13.mul_add(bc, t14.mul_add(cc, t17 * dc))
                - t15.mul_add(cc, t16.mul_add(dc, t12 * bc)));
        o[3][1] = d
            * (t19.mul_add(cc, t20.mul_add(dc, t12 * ac))
                - t13.mul_add(ac, t18.mul_add(cc, t21 * dc)));
        o[3][2] = d
            * (t15.mul_add(ac, t18.mul_add(bc, t23 * dc))
                - t19.mul_add(bc, t22.mul_add(dc, t14 * ac)));
        o[3][3] = d
            * (t21.mul_add(bc, t22.mul_add(cc, t16 * ac))
                - t17.mul_add(ac, t20.mul_add(bc, t23 * cc)));

        o
    }

    #[inline]
    #[must_use]
    pub fn euler_xyz(x_rad: f32, y_rad: f32, z_rad: f32) -> Self {
        #[inline]
        #[must_use]
        fn euler_x(radians: f32) -> Mat4<f32> {
            let c = radians.cos();
            let s = radians.sin();

            Mat4([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, c, s, 0.0],
                [0.0, -s, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
        }

        #[inline]
        #[must_use]
        fn euler_y(radians: f32) -> Mat4<f32> {
            let c = radians.cos();
            let s = radians.sin();

            Mat4([
                [c, 0.0, -s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
        }

        #[inline]
        #[must_use]
        fn euler_z(radians: f32) -> Mat4<f32> {
            let c = radians.cos();
            let s = radians.sin();

            Mat4([
                [c, s, 0.0, 0.0],
                [-s, c, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ])
        }

        let rx = euler_x(x_rad);
        let ry = euler_y(y_rad);
        let rz = euler_z(z_rad);

        (rx * ry) * rz
    }

    #[inline]
    #[must_use]
    pub fn forward(&self) -> Vec3<f32> {
        Vec3 {
            x: -self[0][2],
            y: -self[1][2],
            z: -self[2][2],
        }
        .normalized()
    }

    #[inline]
    #[must_use]
    pub fn backward(&self) -> Vec3<f32> {
        Vec3 {
            x: self[0][2],
            y: self[1][2],
            z: self[2][2],
        }
        .normalized()
    }

    #[inline]
    #[must_use]
    pub fn up(&self) -> Vec3<f32> {
        Vec3 {
            x: self[0][1],
            y: self[1][1],
            z: self[2][1],
        }
        .normalized()
    }

    #[inline]
    #[must_use]
    pub fn down(&self) -> Vec3<f32> {
        Vec3 {
            x: -self[0][1],
            y: -self[1][1],
            z: -self[2][1],
        }
        .normalized()
    }

    #[inline]
    #[must_use]
    pub fn left(&self) -> Vec3<f32> {
        Vec3 {
            x: self[0][0],
            y: self[1][0],
            z: self[2][0],
        }
        .normalized()
    }

    #[inline]
    #[must_use]
    pub fn right(&self) -> Vec3<f32> {
        Vec3 {
            x: -self[0][0],
            y: -self[1][0],
            z: -self[2][0],
        }
        .normalized()
    }

    #[inline]
    #[must_use]
    #[allow(clippy::similar_names)]
    pub const fn as_bytes(&self) -> [u8; std::mem::size_of::<Self>()] {
        let [a, b, c, d] = self.0;

        let [aa, ab, ac, ad] = a;
        let [ba, bb, bc, bd] = b;
        let [ca, cb, cc, cd] = c;
        let [da, db, dc, dd] = d;

        let [aa1, aa2, aa3, aa4] = aa.to_ne_bytes();
        let [ab1, ab2, ab3, ab4] = ab.to_ne_bytes();
        let [ac1, ac2, ac3, ac4] = ac.to_ne_bytes();
        let [ad1, ad2, ad3, ad4] = ad.to_ne_bytes();

        let [ba1, ba2, ba3, ba4] = ba.to_ne_bytes();
        let [bb1, bb2, bb3, bb4] = bb.to_ne_bytes();
        let [bc1, bc2, bc3, bc4] = bc.to_ne_bytes();
        let [bd1, bd2, bd3, bd4] = bd.to_ne_bytes();

        let [ca1, ca2, ca3, ca4] = ca.to_ne_bytes();
        let [cb1, cb2, cb3, cb4] = cb.to_ne_bytes();
        let [cc1, cc2, cc3, cc4] = cc.to_ne_bytes();
        let [cd1, cd2, cd3, cd4] = cd.to_ne_bytes();

        let [da1, da2, da3, da4] = da.to_ne_bytes();
        let [db1, db2, db3, db4] = db.to_ne_bytes();
        let [dc1, dc2, dc3, dc4] = dc.to_ne_bytes();
        let [dd1, dd2, dd3, dd4] = dd.to_ne_bytes();

        #[rustfmt::skip]
        [
            aa1, aa2, aa3, aa4, ab1, ab2, ab3, ab4, ac1, ac2, ac3, ac4, ad1, ad2, ad3, ad4,
            ba1, ba2, ba3, ba4, bb1, bb2, bb3, bb4, bc1, bc2, bc3, bc4, bd1, bd2, bd3, bd4,
            ca1, ca2, ca3, ca4, cb1, cb2, cb3, cb4, cc1, cc2, cc3, cc4, cd1, cd2, cd3, cd4,
            da1, da2, da3, da4, db1, db2, db3, db4, dc1, dc2, dc3, dc4, dd1, dd2, dd3, dd4,
        ]
    }
}

// specialization
impl Mat4<f32> {
    #[must_use]
    pub fn multiply(&self, rhs: &Self) -> Self {
        let [a, b, c, d] = self.0;
        let [x, y, z, w] = rhs.0;

        let [aa, ab, ac, ad] = a;
        let [ba, bb, bc, bd] = b;
        let [ca, cb, cc, cd] = c;
        let [da, db, dc, dd] = d;

        let [xx, xy, xz, xw] = x;
        let [yx, yy, yz, yw] = y;
        let [zx, zy, zz, zw] = z;
        let [wx, wy, wz, ww] = w;

        Self([
            [
                ad.mul_add(wx, ac.mul_add(zx, aa.mul_add(xx, ab * yx))),
                ad.mul_add(wy, ac.mul_add(zy, aa.mul_add(xy, ab * yy))),
                ad.mul_add(wz, ac.mul_add(zz, aa.mul_add(xz, ab * yz))),
                ad.mul_add(ww, ac.mul_add(zw, aa.mul_add(xw, ab * yw))),
            ],
            [
                bd.mul_add(wx, bc.mul_add(zx, ba.mul_add(xx, bb * yx))),
                bd.mul_add(wy, bc.mul_add(zy, ba.mul_add(xy, bb * yy))),
                bd.mul_add(wz, bc.mul_add(zz, ba.mul_add(xz, bb * yz))),
                (bc * zw * bd).mul_add(ww, ba.mul_add(xw, bb * yw)),
            ],
            [
                cd.mul_add(wx, cc.mul_add(zx, ca.mul_add(xx, cb * yx))),
                cd.mul_add(wy, cc.mul_add(zy, ca.mul_add(xy, cb * yy))),
                cd.mul_add(wz, cc.mul_add(zz, ca.mul_add(xz, cb * yz))),
                cd.mul_add(ww, cc.mul_add(zw, ca.mul_add(xw, cb * yw))),
            ],
            [
                dd.mul_add(wx, dc.mul_add(zx, da.mul_add(xx, db * xy))),
                dd.mul_add(wy, dc.mul_add(zy, da.mul_add(xy, db * yy))),
                dd.mul_add(wz, dc.mul_add(zz, da.mul_add(xz, db * zy))),
                dd.mul_add(ww, dc.mul_add(zw, da.mul_add(xw, db * wy))),
            ],
        ])
    }

    #[must_use]
    pub fn scalar(&self, rhs: f32) -> Self {
        let [a, b, c, d] = self.0;

        let [aa, ab, ac, ad] = a;
        let [ba, bb, bc, bd] = b;
        let [ca, cb, cc, cd] = c;
        let [da, db, dc, dd] = d;

        Self([
            [aa * rhs, ab * rhs, ac * rhs, ad * rhs],
            [ba * rhs, bb * rhs, bc * rhs, bd * rhs],
            [ca * rhs, cb * rhs, cc * rhs, cd * rhs],
            [da * rhs, db * rhs, dc * rhs, dd * rhs],
        ])
    }
}

#[cfg(test)]
mod tests {

    use crate::vec::Vec4;

    use super::Mat4;

    #[test]
    fn test_mat4_add() {
        let mat1 = Mat4::<f32>::zero();
        let mat2 = Mat4::<f32>::one();

        let mat3 = mat1 + mat2;
        assert_eq!(mat3, Mat4::one());
    }

    #[test]
    fn test_mat4_sub() {
        let mat1 = Mat4::<f32>::one();
        let mat2 = Mat4::<f32>::one();

        let mat3 = mat1 - mat2;
        assert_eq!(mat3, Mat4::zero());
    }

    #[test]
    fn test_mat4_mul() {
        let mat1 = Mat4::<f32>::identity();
        let mat2 = Mat4::<f32>::one();

        let mat3 = mat1 * mat2;
        assert_eq!(mat3, Mat4::one());
    }

    #[test]
    fn test_mat4_transpose() {
        let mat = Mat4::<f32>::from([
            [0.0, 1.0, 2.0, -2.0],
            [0.0, 0.0, 3.0, 0.0],
            [5.0, -1.0, 0.0, 5.0],
            [1.1, 2.2, 3.3, 4.4],
        ]);

        let mut transposed = mat.transposed();
        assert_eq!(
            transposed,
            Mat4::<f32>::from([
                [0.0, 0.0, 5.0, 1.1],
                [1.0, 0.0, -1.0, 2.2],
                [2.0, 3.0, 0.0, 3.3],
                [-2.0, 0.0, 5.0, 4.4],
            ])
        );

        transposed.transpose();
        assert_eq!(transposed, mat);
    }

    #[test]
    fn test_mat4_scalar() {
        let mat = Mat4::from_array([
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, -2.0, -3.0, -4.0],
            [4.0, 3.0, 2.0, 1.0],
            [-4.0, -3.0, -2.0, -1.0],
        ]);

        let out = mat * 2.0;

        assert_eq!(
            out,
            Mat4::from_array([
                [2.0, 4.0, 6.0, 8.0],
                [-2.0, -4.0, -6.0, -8.0],
                [8.0, 6.0, 4.0, 2.0],
                [-8.0, -6.0, -4.0, -2.0],
            ])
        );
    }

    #[test]
    fn test_mat4_vector() {
        let mat1 = Mat4::<f32>::from_array([
            [1.0, 4.0, 0.0, -3.0],
            [4.0, 2.0, 0.0, 5.0],
            [1.0, 5.0, -2.0, 6.0],
            [-1.0, 4.0, -4.0, -7.0],
        ]);
        let vec1 = Vec4::new(-3.0, 3.0, 4.0, 5.0);

        let out = mat1 * vec1;
        assert_eq!(out, Vec4::new(-6.0, 19.0, 34.0, -36.0));
    }

    #[test]
    fn test_mat4_inverse() {
        let mat = Mat4::<f32>::from_array([
            [1.0, -2.0, 5.0, -9.0],
            [27.0, 12.0, -3.0, -7.0],
            [7.0, 8.0, 10.0, 0.0],
            [9.0, 0.0, 1.0, 9.0],
        ]);

        let out = mat.inverse() * 100.0;

        let result = Mat4::<f32>::from_array([
            [4.282_951, 2.113_843_4, -2.100_027_6, 5.927_051_5],
            [-11.052_777, 1.802_984_2, 7.032_329_6, -9.650_455],
            [5.844_156, -2.922_078, 5.844_156, 3.571_428_3],
            [-4.932_301_5, -1.789_168_2, 1.450_676_9, 4.787_234],
        ]);

        assert_eq!(out, result);
    }
}
