use core::ops::{Deref, DerefMut};

use crate::{matrix::Mat4, vec::Vec3};

use super::vec::Vec4;

const SLERP_THRESHHOLD: f32 = 0.9995;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Quat<T>(Vec4<T>);

impl<T> Deref for Quat<T> {
    type Target = Vec4<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Quat<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

mod mul {
    use crate::vec::Vec4;

    use super::Quat;
    use core::ops::Add;
    use core::ops::Mul;
    use core::ops::Sub;

    impl<T> Quat<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Copy,
    {
        #[inline]
        fn mul_inner(&self, rhs: &Self) -> Self {
            Self(Vec4 {
                x: self.x * rhs.w + self.y * rhs.z - self.z * rhs.y + self.w * rhs.x,
                y: self.x * rhs.z + self.y * rhs.w + self.z * rhs.x + self.w * rhs.y,
                z: self.x * rhs.y - self.y * rhs.x + self.z * rhs.w + self.w * rhs.z,
                w: self.x * rhs.x - self.y * rhs.y - self.z * rhs.z + self.w * rhs.w,
            })
        }
    }

    impl<T> Mul for Quat<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Copy,
    {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            self.mul_inner(&rhs)
        }
    }

    impl<T> Mul for &Quat<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Copy,
    {
        type Output = Quat<T>;

        fn mul(self, rhs: Self) -> Self::Output {
            self.mul_inner(rhs)
        }
    }

    impl<T> Mul<Quat<T>> for &Quat<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Copy,
    {
        type Output = Quat<T>;

        fn mul(self, rhs: Quat<T>) -> Self::Output {
            self.mul_inner(&rhs)
        }
    }

    impl<T> Mul<&Self> for Quat<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Sub<Output = T> + Copy,
    {
        type Output = Self;

        fn mul(self, rhs: &Self) -> Self::Output {
            self.mul_inner(rhs)
        }
    }
}

impl Quat<f32> {
    #[inline]
    #[must_use]
    pub const fn identity() -> Self {
        Self(Vec4::new(0.0, 0.0, 0.0, 1.0))
    }

    #[inline]
    #[must_use]
    pub fn normal(&self) -> f32 {
        self.0.lenght()
    }

    #[inline]
    #[must_use]
    pub fn normalized(&self) -> Self {
        Self(self.0.normalized())
    }

    #[inline]
    #[must_use]
    pub fn conjugate(&self) -> Self {
        Self(Vec4 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: self.w,
        })
    }

    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Self {
        self.conjugate().normalized()
    }

    #[inline]
    #[must_use]
    pub fn dot(&self, rhs: &Self) -> f32 {
        self.0.dot(&rhs.0)
    }

    #[inline]
    #[must_use]
    pub fn to_mat4(&self) -> Mat4<f32> {
        let n = self.normalized();

        Mat4::from_array([
            [
                (2.0 * n.z).mul_add(-n.z, (2.0 * n.y).mul_add(-n.y, 1.0)),
                (2.0 * n.x).mul_add(n.y, -(2.0 * n.z * n.w)),
                (2.0 * n.x).mul_add(n.z, 2.0 * n.y * n.w),
                0.0,
            ],
            [
                (2.0 * n.x).mul_add(n.y, 2.0 * n.z * n.w),
                (2.0 * n.z).mul_add(-n.z, (2.0 * n.x).mul_add(-n.x, 1.0)),
                (2.0 * n.y).mul_add(n.z, -(2.0 * n.x * n.w)),
                0.0,
            ],
            [
                (2.0 * n.x).mul_add(n.z, -(2.0 * n.y * n.w)),
                (2.0 * n.y).mul_add(n.z, 2.0 * n.x * n.w),
                (2.0 * n.y).mul_add(-n.y, (2.0 * n.x).mul_add(-n.x, 1.0)),
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0],
        ])
    }

    #[inline]
    #[must_use]
    pub fn rotation_matrix(&self, center: &Vec3<f32>) -> Mat4<f32> {
        let q = self.0;

        let aa =
            q.w.mul_add(q.w, q.z.mul_add(-q.z, q.x.mul_add(q.x, -(q.y * q.y))));
        let ab = 2.0 * q.x.mul_add(q.y, q.z * q.w);
        let ac = 2.0 * q.x.mul_add(q.z, -(q.y * q.w));
        let ad = center
            .z
            .mul_add(-ac, center.y.mul_add(-ab, center.x.mul_add(-aa, center.x)));

        let ba = 2.0 * q.x.mul_add(q.y, -(q.z * q.w));
        let bb =
            q.w.mul_add(q.w, q.z.mul_add(-q.z, q.y.mul_add(q.y, -(q.x * q.x))));
        let bc = 2.0 * q.y.mul_add(q.z, q.x * q.w);
        let bd = center
            .z
            .mul_add(-bc, center.y.mul_add(-bb, center.x.mul_add(-ba, center.y)));

        let ca = 2.0 * q.x.mul_add(q.z, q.y * q.w);
        let cb = 2.0 * q.y.mul_add(q.z, -(q.x * q.w));
        let cc =
            q.w.mul_add(q.w, q.z.mul_add(q.z, q.y.mul_add(-q.y, -(q.x * q.x))));
        let cd = center
            .z
            .mul_add(-cc, center.y.mul_add(-cb, center.x.mul_add(-ca, center.z)));

        Mat4::from_array([
            [aa, ab, ac, ad],
            [ba, bb, bc, bd],
            [ca, cb, cc, cd],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: Vec3<f32>, angle: f32) -> Self {
        let half_angle = 0.5 * angle;
        let s = half_angle.sin();
        let c = half_angle.cos();

        Self(Vec4 {
            x: s * axis.x,
            y: s * axis.y,
            z: s * axis.z,
            w: c,
        })
    }

    #[inline]
    #[must_use]
    pub fn slerp(&self, rhs: &Self, percent: f32) -> Self {
        let v0 = self.normalized();
        let mut v1 = rhs.normalized();

        let mut dot = v0.dot(&v1);

        if dot < 0.0 {
            v1.x *= -1.;
            v1.y *= -1.;
            v1.z *= -1.;
            v1.w *= -1.;
            dot *= -1.;
        }

        if dot > SLERP_THRESHHOLD {
            return Self(v0.0 + ((v1.0 - v0.0) * percent)).normalized();
        }

        let theta_0 = dot.acos();
        let theta = theta_0 * percent;

        let sin_theta = theta.sin();
        let sin_theta_0 = theta_0.sin();

        let s0 = theta.cos() - dot * sin_theta / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;

        Self((v0.0 * s0) + (v1.0 * s1))
    }
}
