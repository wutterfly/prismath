use core::f32::consts::PI;

const SLERP_THRESHHOLD: f32 = 0.9995;

use crate::{
    matrix::{Mat3, Mat4},
    vec::{Vec3, Vec4},
};

// From:
// https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Quat<T> {
    w: T,
    x: T,
    y: T,
    z: T,
}

impl<T: Copy> Quat<T> {
    #[inline]
    #[must_use]
    pub const fn new(w: T, xyz: (T, T, T)) -> Self {
        Self {
            w,
            x: xyz.0,
            y: xyz.1,
            z: xyz.2,
        }
    }

    #[inline]
    #[must_use]
    const fn parts(self) -> (T, T, T, T) {
        (self.w, self.x, self.y, self.z)
    }
}

mod mul {

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
            let (r0, r1, r2, r3) = self.parts();
            let (s0, s1, s2, s3) = rhs.parts();

            let q0 = r0 * s0 - r1 * s1 - r2 * s2 - r3 * s3;
            let q1 = r0 * s1 + r1 * s0 - r2 * s3 + r3 * s2;
            let q2 = r0 * s2 + r1 * s3 + r2 * s0 - r3 * s1;
            let q3 = r0 * s3 - r1 * s2 + r2 * s1 + r3 * s0;

            Self::new(q0, (q1, q2, q3))
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
        Self::new(1.0, (0.0, 0.0, 0.0))
    }

    #[must_use]
    /// Takes
    ///     - an axis
    ///     - an angle in degree
    ///
    /// Returns a Quaternion.
    pub fn from_axis_angle(axis: Vec3<f32>, angle: f32) -> Self {
        let (x, y, z) = axis.parts();

        let angle = f32::to_radians(angle);

        let q0 = f32::cos(angle * 0.5);
        let q1 = x * f32::sin(angle * 0.5);
        let q2 = y * f32::sin(angle * 0.5);
        let q3 = z * f32::sin(angle * 0.5);

        Self::new(q0, (q1, q2, q3))
    }

    #[must_use]
    /// Takes an Quaternion.
    ///
    /// Returns
    ///     - an axis
    ///     - an angle in degree
    pub fn to_axis_angle(self) -> (Vec3<f32>, f32) {
        let (q0, q1, q2, q3) = self.parts();

        // check for identity
        #[allow(clippy::float_cmp)]
        if q0 == 1.0 {
            return (Vec3::new(1.0, 0.0, 0.0), 0.0);
        }

        let angle = 2.0 * f32::acos(q0);
        let x = q1 / f32::sin(angle * 0.5);
        let y = q2 / f32::sin(angle * 0.5);
        let z = q3 / f32::sin(angle * 0.5);

        (Vec3::new(x.abs(), y.abs(), z.abs()), angle.to_degrees())
    }

    #[must_use]
    /// Takes an Quaternion.
    ///
    /// Returns a rotation matrix as Mat3;
    pub fn to_rotation_matrix3(self) -> Mat3<f32> {
        let (q0, q1, q2, q3) = self.parts();

        let q0_2 = q0 * q0;
        let q1_2 = q1 * q1;
        let q2_2 = q2 * q2;
        let q3_2 = q3 * q3;

        let q0q1 = q0 * q1;
        let q0q2 = q0 * q2;
        let q0q3 = q0 * q3;

        let q1q2 = q1 * q2;
        let q1q3 = q1 * q3;
        let q2q3 = q2 * q3;

        let r1 = [
            q0_2 + q1_2 - q2_2 - q3_2,
            2.0f32.mul_add(q1q2, -(2. * q0q3)),
            2.0f32.mul_add(q1q3, 2. * q0q2),
        ];
        let r2 = [
            2.0f32.mul_add(q1q2, q0q3),
            2.0f32.mul_add(-q3_2, 2.0f32.mul_add(-q1_2, 1.)),
            2.0f32.mul_add(q2q3, -(2. * q0q1)),
        ];
        let r3 = [
            2.0f32.mul_add(q1q3, -(2. * q0q2)),
            2.0f32.mul_add(q2q3, 2. * q0q1),
            2.0f32.mul_add(-q2_2, 2.0f32.mul_add(-q1_2, 1.)),
        ];

        Mat3::from_array([r1, r2, r3])
    }

    #[inline]
    #[must_use]
    /// Takes an Quaternion.
    ///
    /// Returns a rotation matrix as Mat4;
    pub fn to_rotation_matrix4(self) -> Mat4<f32> {
        let [r1, r2, r3] = Self::to_rotation_matrix3(self).to_arrays();

        let [r11, r12, r13] = r1;
        let [r21, r22, r23] = r2;
        let [r31, r32, r33] = r3;

        let r1 = [r11, r12, r13, 0.0];
        let r2 = [r21, r22, r23, 0.0];
        let r3 = [r31, r32, r33, 0.0];
        let r4 = [0.0, 0.0, 0.0, 0.0];

        Mat4::from_array([r1, r2, r3, r4])
    }

    #[must_use]
    /// Takes an rotation matrix (mat3).
    ///
    /// Returns a Quaternion.
    pub fn from_rotation_matrix3(mat: Mat3<f32>) -> Self {
        let [r1, r2, r3] = mat.to_arrays();
        let [r11, r12, r13] = r1;
        let [r21, r22, r23] = r2;
        let [r31, r32, r33] = r3;

        let q0 = f32::sqrt((1. + r11 + r22 + r33) / 4.);
        let q1 = f32::sqrt((1. + r11 - r22 - r33) / 4.);
        let q2 = f32::sqrt((1. - r11 + r22 - r33) / 4.);
        let q3 = f32::sqrt((1. - r11 - r22 + r33) / 4.);

        let mut max = q0;
        for q in [q0, q1, q2, q3] {
            max = f32::max(q, max);
        }

        #[allow(clippy::float_cmp)]
        match max {
            q if max == q => {
                let q1 = (r32 - r23) / (4. * q0);
                let q2 = (r13 - r31) / (4. * q0);
                let q3 = (r21 - r12) / (4. * q0);

                Self::new(q0, (q1, q2, q3))
            }
            q if max == q => {
                let q0 = (r32 - r23) / (4. * q1);
                let q2 = (r12 - r21) / (4. * q1);
                let q3 = (r13 - r31) / (4. * q1);

                Self::new(q0, (q1, q2, q3))
            }
            q if max == q => {
                let q0 = (r13 - r31) / (4. * q2);
                let q1 = (r12 - r21) / (4. * q2);
                let q3 = (r23 - r32) / (4. * q2);

                Self::new(q0, (q1, q2, q3))
            }
            q if max == q => {
                let q0 = (r21 - r12) / (4. * q3);
                let q1 = (r13 - r32) / (4. * q3);
                let q2 = (r23 - r32) / (4. * q3);

                Self::new(q0, (q1, q2, q3))
            }
            _ => unreachable!(),
        }
    }

    #[inline]
    #[must_use]
    /// Takes an rotation matrix (mat4).
    ///
    /// Returns a Quaternion.
    pub fn from_rotation_matrix4(mat: Mat4<f32>) -> Self {
        let [r1, r2, r3, _] = mat.to_arrays();
        let [r11, r12, r13, _] = r1;
        let [r21, r22, r23, _] = r2;
        let [r31, r32, r33, _] = r3;

        Self::from_rotation_matrix3(Mat3::from_array([
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33],
        ]))
    }

    #[must_use]
    /// Takes three euler angles in degree:
    ///     - roll : x
    ///     - pitch : y
    ///     - yaw : z
    ///
    /// Uses ZYX-ordering.
    ///
    /// Returns a Quaternion.
    pub fn from_euler_angles(roll: f32, pitch: f32, yaw: f32) -> Self {
        let roll = f32::to_radians(roll);
        let pitch = f32::to_radians(pitch);
        let yaw = f32::to_radians(yaw);

        let rh = roll * 0.5;
        let ph = pitch * 0.5;
        let yh = yaw * 0.5;

        let cr = f32::cos(rh);
        let sr = f32::sin(rh);
        let cp = f32::cos(ph);
        let sp = f32::sin(ph);
        let cy = f32::cos(yh);
        let sy = f32::sin(yh);

        let q0 = (cr * cp).mul_add(cy, sr * sp * sy);
        let q1 = (sr * cp).mul_add(cy, -(cr * sp * sy));
        let q2 = (cr * sp).mul_add(cy, sr * cp * sy);
        let q3 = (cr * cp).mul_add(sy, -(sr * sp * cy));

        Self::new(q0, (q1, q2, q3))
    }

    #[must_use]
    /// Takes a Quaternion.
    ///
    /// Returns three euler angles in degree
    ///     - (roll, pitch, yaw)
    pub fn to_euler_angles(self) -> (f32, f32, f32) {
        let (q0, q1, q2, q3) = self.parts();

        let q0_2 = q0 * q0;
        let q1_2 = q1 * q1;
        let q2_2 = q2 * q2;
        let q3_2 = q3 * q3;

        let pitch = f32::asin(2. * q0.mul_add(q2, -(q1 * q3)));

        // check for gimble lock
        // +90 degree
        if pitch == PI * 0.5 {
            let roll = 0.0;
            let yaw = -2. * f32::atan2(q1, q0);
            (roll, pitch.to_degrees(), yaw.to_degrees())
        }
        // -90 degree
        else if pitch == -PI * 0.5 {
            let roll = 0.0;
            let yaw = 2. * f32::atan2(q1, q0);
            (roll, pitch.to_degrees(), yaw.to_degrees())
        }
        // no gimble lock
        else {
            let roll = f32::atan2(2. * q0.mul_add(q1, q2 * q3), q0_2 - q1_2 - q2_2 + q3_2);
            let yaw = f32::atan2(2. * q0.mul_add(q3, q1 * q2), q0_2 + q1_2 - q2_2 - q3_2);

            (roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees())
        }
    }

    #[inline]
    #[must_use]
    pub const fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    #[inline]
    #[must_use]
    pub fn inversed(self) -> Self {
        self.conjugate().normalized()
    }

    #[inline]
    pub fn inverse(&mut self) {
        *self = Self::inversed(*self);
    }

    #[inline]
    #[must_use]
    pub fn lenght(self) -> f32 {
        let (q0, q1, q2, q3) = self.parts();
        f32::sqrt(q3.mul_add(q3, q2.mul_add(q2, q0.mul_add(q0, q1 * q1))))
    }

    #[inline]
    #[must_use]
    pub fn normalized(self) -> Self {
        let l = self.lenght();

        let (q0, q1, q2, q3) = self.parts();

        Self {
            w: q0 / l,
            x: q1 / l,
            y: q2 / l,
            z: q3 / l,
        }
    }

    #[inline]
    pub fn normalize(&mut self) {
        *self = Self::normalized(*self);
    }

    /// Rotate a point by the rotation quaternion.
    ///
    /// This is a active rotation (q*p*q^-1).
    #[must_use]
    pub fn rotate(rotation: Self, point: Vec3<f32>) -> Vec3<f32> {
        let point = Self::new(0., point.parts());

        let active = (rotation.inversed() * point) * rotation;

        let (_, x, y, z) = active.parts();

        Vec3::new(x, y, z)
    }

    #[inline]
    #[must_use]
    pub fn slerp(&self, rhs: &Self, percent: f32) -> Self {
        let v0 = self.normalized();
        let v1 = rhs.normalized();

        let v0 = Vec4::new(v0.x, v0.y, v0.z, v0.w);
        let mut v1 = Vec4::new(v1.x, v1.y, v1.z, v1.w);

        let mut dot = v0.dot(&v1);

        if dot < 0.0 {
            v1.x *= -1.;
            v1.y *= -1.;
            v1.z *= -1.;
            v1.w *= -1.;
            dot *= -1.;
        }

        if dot > SLERP_THRESHHOLD {
            let vec = v0 + ((v1 - v0) * percent).normalized();
            return Self::new(vec.w, (vec.x, vec.y, vec.z));
        }

        let theta_0 = dot.acos();
        let theta = theta_0 * percent;

        let sin_theta = theta.sin();
        let sin_theta_0 = theta_0.sin();

        let s0 = theta.cos() - dot * sin_theta / sin_theta_0;
        let s1 = sin_theta / sin_theta_0;

        let vec = (v0 * s0) + (v1 * s1);
        Self::new(vec.w, (vec.x, vec.y, vec.z))
    }
}

#[cfg(test)]
mod tests {
    use crate::{matrix::Mat3, vec::Vec3};

    use super::Quat;

    #[test]
    fn test_from_to_axis_angle() {
        let axis = Vec3::new(1.0, 0.0, 0.0);
        let angle = 45.0;

        let quad = Quat::from_axis_angle(axis, angle);
        assert_eq!(quad, Quat::new(0.9238795, (0.38268346, 0., 0.)));

        let (ax, an) = quad.to_axis_angle();
        assert!((axis - ax).length() < 0.001);
        assert!((angle - an).abs() < 0.001);

        //

        let axis = Vec3::new(0.0, 1.0, 0.0);
        let angle = 60.0;

        let quad = Quat::from_axis_angle(axis, angle);
        assert_eq!(quad, Quat::new(0.8660254, (0., 0.5, 0.)));

        let (ax, an) = quad.to_axis_angle();
        assert!((axis - ax).length() < 0.001);
        assert!((angle - an).abs() < 0.001);

        //

        let axis = Vec3::new(0.0, 0.0, 1.0);
        let angle = 124.0;

        let quad = Quat::from_axis_angle(axis, angle);
        assert_eq!(quad, Quat::new(0.4694716, (0., 0., 0.88294756)));

        let (ax, an) = quad.to_axis_angle();
        assert!((axis - ax).length() < 0.001);
        assert!((angle - an).abs() < 0.001);
    }

    #[test]
    fn test_from_to_rotation_matrix() {
        let axis = Vec3::new(0.0, 1.0, 0.0);
        let angle = 60.0;

        let quad = Quat::from_axis_angle(axis, angle);

        let mat = quad.to_rotation_matrix3();

        assert_eq!(
            mat,
            Mat3::from_array([
                [0.5, 0.0, 0.8660254],
                [0.0, 1.0, 0.0],
                [-0.8660254, 0.0, 0.5]
            ])
        );

        let res = Quat::from_rotation_matrix3(mat);
        assert_eq!(quad, res);
    }

    #[test]
    fn test_from_to_euler_angle() {
        let quad = Quat::from_euler_angles(60., 0., 0.);
        assert_eq!(quad, Quat::new(0.8660254, (0.5, 0., 0.)));

        let (u, v, w) = Quat::to_euler_angles(quad);
        assert!((u - 60.).abs() < 0.001);
        assert!((v - 0.).abs() < 0.001);
        assert!((w - 0.).abs() < 0.001);

        //

        let quad = Quat::from_euler_angles(0., 45., 0.);
        assert_eq!(quad, Quat::new(0.9238795, (0., 0.38268346, 0.)));

        let (u, v, w) = Quat::to_euler_angles(quad);
        assert!((u - 0.).abs() < 0.001);
        assert!((v - 45.).abs() < 0.001);
        assert!((w - 0.).abs() < 0.001);

        //

        let quad = Quat::from_euler_angles(0., 0., 20.);
        assert_eq!(quad, Quat::new(0.9848077, (0., 0., 0.17364818)));

        let (u, v, w) = Quat::to_euler_angles(quad);
        assert!((u - 0.).abs() < 0.001);
        assert!((v - 0.).abs() < 0.001);
        assert!((w - 20.).abs() < 0.001);

        // ZYX (?)

        let quad = Quat::from_euler_angles(10., 80., 45.);
        assert_eq!(
            quad,
            Quat::new(0.7264786, (-0.18336517, 0.6171484, 0.24027884))
        );

        let (u, v, w) = Quat::to_euler_angles(quad);
        assert!((u - 10.).abs() < 0.001);
        assert!((v - 80.).abs() < 0.001);
        assert!((w - 45.).abs() < 0.001);
    }
}
