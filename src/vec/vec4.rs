use super::{Vec2, Vec3};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T> Vec4<T> {
    #[inline]
    pub const fn new(x: T, y: T, z: T, q: T) -> Self {
        Self { x, y, z, w: q }
    }

    #[inline]
    pub fn parts(self) -> (T, T, T, T) {
        (self.x, self.y, self.z, self.w)
    }

    #[inline]
    pub fn as_array(self) -> [T; 4] {
        [self.x, self.y, self.z, self.w]
    }

    #[inline]
    pub fn from3(vec: Vec3<T>, t: T) -> Self {
        Self {
            x: vec.x,
            y: vec.y,
            z: vec.z,
            w: t,
        }
    }
}

impl<T> core::ops::Index<usize> for Vec4<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Index out of bound: {index}"),
        }
    }
}

impl<T> core::ops::IndexMut<usize> for Vec4<T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Index out of bound: {index}"),
        }
    }
}

impl<T: Copy> Vec4<T> {
    #[inline]
    pub const fn splat(value: T) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
            w: value,
        }
    }
}

impl<T> From<(T, T, T, T)> for Vec4<T> {
    fn from(value: (T, T, T, T)) -> Self {
        Self::new(value.0, value.1, value.2, value.3)
    }
}

impl<T> From<[T; 4]> for Vec4<T> {
    fn from(value: [T; 4]) -> Self {
        let [x, y, z, q] = value;
        Self::new(x, y, z, q)
    }
}

impl<T: Default> From<Vec3<T>> for Vec4<T> {
    fn from(value: Vec3<T>) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
            w: T::default(),
        }
    }
}

impl<T: Default> From<Vec2<T>> for Vec4<T> {
    fn from(value: Vec2<T>) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: T::default(),
            w: T::default(),
        }
    }
}

mod add {
    use super::Vec4;
    use core::ops::Add;

    impl<T> Vec4<T>
    where
        T: Copy + Add<Output = T>,
    {
        #[inline]
        fn add_inner(self, rhs: &Self) -> Self {
            Self {
                x: self.x + rhs.x,
                y: self.y + rhs.y,
                z: self.z + rhs.z,
                w: self.w + rhs.w,
            }
        }
    }

    impl<T> Add for Vec4<T>
    where
        T: Copy + Add<Output = T>,
    {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            self.add_inner(&rhs)
        }
    }

    impl<T> Add for &Vec4<T>
    where
        T: Copy + Add<Output = T>,
    {
        type Output = Vec4<T>;

        fn add(self, rhs: Self) -> Self::Output {
            self.add_inner(rhs)
        }
    }

    impl<T> Add<Vec4<T>> for &Vec4<T>
    where
        T: Copy + Add<Output = T>,
    {
        type Output = Vec4<T>;

        fn add(self, rhs: Vec4<T>) -> Self::Output {
            self.add_inner(&rhs)
        }
    }

    impl<T> Add<&Self> for Vec4<T>
    where
        T: Copy + Add<Output = T>,
    {
        type Output = Self;

        fn add(self, rhs: &Self) -> Self::Output {
            self.add_inner(rhs)
        }
    }
}

mod add_assign {
    use super::Vec4;
    use core::ops::Add;
    use std::ops::AddAssign;

    impl<T> Vec4<T>
    where
        T: Copy + Add<Output = T>,
    {
        #[inline]
        fn add_assing_inner(&mut self, rhs: &Self) {
            self.x = self.x + rhs.x;
            self.y = self.y + rhs.y;
            self.z = self.z + rhs.z;
            self.w = self.w + rhs.w;
        }
    }

    impl<T> AddAssign for Vec4<T>
    where
        T: Copy + Add<Output = T>,
    {
        fn add_assign(&mut self, rhs: Self) {
            Self::add_assing_inner(self, &rhs);
        }
    }

    impl<T> AddAssign<&Self> for Vec4<T>
    where
        T: Copy + Add<Output = T>,
    {
        fn add_assign(&mut self, rhs: &Self) {
            Self::add_assing_inner(self, rhs);
        }
    }
}

mod sub {
    use super::Vec4;
    use core::ops::Sub;

    impl<T> Vec4<T>
    where
        T: Copy + Sub<Output = T>,
    {
        #[inline]
        fn sub_inner(self, rhs: &Self) -> Self {
            Self {
                x: self.x - rhs.x,
                y: self.y - rhs.y,
                z: self.z - rhs.z,
                w: self.w - rhs.w,
            }
        }
    }

    impl<T> Sub for Vec4<T>
    where
        T: Copy + Sub<Output = T>,
    {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            self.sub_inner(&rhs)
        }
    }

    impl<T> Sub for &Vec4<T>
    where
        T: Copy + Sub<Output = T>,
    {
        type Output = Vec4<T>;

        fn sub(self, rhs: Self) -> Self::Output {
            self.sub_inner(rhs)
        }
    }

    impl<T> Sub<Vec4<T>> for &Vec4<T>
    where
        T: Copy + Sub<Output = T>,
    {
        type Output = Vec4<T>;

        fn sub(self, rhs: Vec4<T>) -> Self::Output {
            self.sub_inner(&rhs)
        }
    }

    impl<T> Sub<&Self> for Vec4<T>
    where
        T: Copy + Sub<Output = T>,
    {
        type Output = Self;

        fn sub(self, rhs: &Self) -> Self::Output {
            self.sub_inner(rhs)
        }
    }
}

mod sub_assign {
    use super::Vec4;
    use core::ops::Sub;
    use std::ops::SubAssign;

    impl<T> Vec4<T>
    where
        T: Copy + Sub<Output = T>,
    {
        #[inline]
        fn sub_assing_inner(&mut self, rhs: &Self) {
            self.x = self.x - rhs.x;
            self.y = self.y - rhs.y;
            self.z = self.z - rhs.z;
            self.w = self.w - rhs.w;
        }
    }

    impl<T> SubAssign for Vec4<T>
    where
        T: Copy + Sub<Output = T>,
    {
        fn sub_assign(&mut self, rhs: Self) {
            Self::sub_assing_inner(self, &rhs);
        }
    }

    impl<T> SubAssign<&Self> for Vec4<T>
    where
        T: Copy + Sub<Output = T>,
    {
        fn sub_assign(&mut self, rhs: &Self) {
            Self::sub_assing_inner(self, rhs);
        }
    }
}

mod scalar {
    use super::Vec4;
    use core::ops::Mul;

    impl<T> Vec4<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        #[inline]
        fn scalar(&self, scalar: &T) -> Self {
            Self {
                x: self.x * scalar,
                y: self.y * scalar,
                z: self.z * scalar,
                w: self.w * scalar,
            }
        }
    }

    impl<T> Mul<T> for Vec4<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Self;

        fn mul(self, rhs: T) -> Self::Output {
            self.scalar(&rhs)
        }
    }

    impl<T> Mul<T> for &Vec4<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Vec4<T>;

        fn mul(self, rhs: T) -> Self::Output {
            self.scalar(&rhs)
        }
    }

    impl<T> Mul<&T> for &Vec4<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Vec4<T>;

        fn mul(self, rhs: &T) -> Self::Output {
            self.scalar(rhs)
        }
    }

    impl<T> Mul<&T> for Vec4<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Self;

        fn mul(self, rhs: &T) -> Self::Output {
            self.scalar(rhs)
        }
    }

    impl Mul<Vec4<Self>> for f32 {
        type Output = Vec4<Self>;

        fn mul(self, rhs: Vec4<Self>) -> Self::Output {
            rhs.scalar(&self)
        }
    }

    impl Mul<Vec4<Self>> for u32 {
        type Output = Vec4<Self>;

        fn mul(self, rhs: Vec4<Self>) -> Self::Output {
            rhs.scalar(&self)
        }
    }
}

mod scalar_assign {
    use super::Vec4;
    use core::ops::Mul;
    use std::ops::MulAssign;

    impl<T> Vec4<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        #[inline]
        fn scalar_assign(&mut self, scalar: &T) {
            self.x = self.x * scalar;
            self.y = self.y * scalar;
            self.z = self.z * scalar;
            self.w = self.w * scalar;
        }
    }

    impl<T> MulAssign<T> for Vec4<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        fn mul_assign(&mut self, rhs: T) {
            Self::scalar_assign(self, &rhs);
        }
    }

    impl<T> MulAssign<&T> for Vec4<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        fn mul_assign(&mut self, rhs: &T) {
            Self::scalar_assign(self, rhs);
        }
    }
}

mod neg {
    use super::Vec4;
    use std::ops::Neg;

    impl<T: Neg<Output = T>> Neg for &Vec4<T>
    where
        for<'a> T: Copy + Neg<Output = T>,
    {
        type Output = Vec4<T>;

        #[inline]
        fn neg(self) -> Self::Output {
            Vec4 {
                x: -self.x,
                y: -self.y,
                z: -self.z,
                w: -self.w,
            }
        }
    }

    impl<T: Neg<Output = T>> Neg for Vec4<T>
    where
        T: Neg<Output = T>,
    {
        type Output = Self;

        #[inline]
        fn neg(self) -> Self::Output {
            Vec4 {
                x: -self.x,
                y: -self.y,
                z: -self.z,
                w: -self.w,
            }
        }
    }
}

impl Vec4<f32> {
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    #[inline]
    #[must_use]
    pub const fn one() -> Self {
        Self::new(1.0, 1.0, 1.0, 1.0)
    }

    #[inline]
    #[must_use]
    pub fn dot(&self, rhs: &Self) -> f32 {
        self.w.mul_add(
            rhs.w,
            self.z.mul_add(rhs.z, self.x.mul_add(rhs.x, self.y * rhs.y)),
        )
    }

    #[inline]
    #[must_use]
    pub fn lenght_scale(&self) -> f32 {
        self.w.mul_add(
            self.w,
            self.z
                .mul_add(self.z, self.x.mul_add(self.x, self.y * self.y)),
        )
    }

    #[inline]
    #[must_use]
    pub fn lenght(&self) -> f32 {
        self.lenght_scale().sqrt()
    }

    #[inline]
    #[must_use]
    pub fn normalized(&self) -> Self {
        let len = self.lenght();
        Self {
            x: self.x / len,
            y: self.y / len,
            z: self.z / len,
            w: self.w / len,
        }
    }

    #[inline]
    pub fn normalize(&mut self) {
        *self = self.normalized();
    }

    #[inline]
    #[must_use]
    pub const fn as_bytes(&self) -> [u8; std::mem::size_of::<Self>()] {
        let [x1, x2, x3, x4] = self.x.to_ne_bytes();
        let [y1, y2, y3, y4] = self.y.to_ne_bytes();
        let [z1, z2, z3, z4] = self.z.to_ne_bytes();
        let [w1, w2, w3, w4] = self.w.to_ne_bytes();

        [
            x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4, w1, w2, w3, w4,
        ]
    }
}
