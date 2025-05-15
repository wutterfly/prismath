use super::{Vec2, Vec4};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Vec3<T> {
    #[inline]
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub fn parts(self) -> (T, T, T) {
        (self.x, self.y, self.z)
    }

    pub fn as_array(self) -> [T; 3] {
        [self.x, self.y, self.z]
    }
}

impl<T> core::ops::Index<usize> for Vec3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Index out of bound: {index}"),
        }
    }
}

impl<T> core::ops::IndexMut<usize> for Vec3<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Index out of bound: {index}"),
        }
    }
}

impl<T: Copy> Vec3<T> {
    #[inline]
    pub const fn splat(value: T) -> Self {
        Self {
            x: value,
            y: value,
            z: value,
        }
    }
}

impl<T> From<(T, T, T)> for Vec3<T> {
    #[inline]
    fn from(value: (T, T, T)) -> Self {
        Self::new(value.0, value.1, value.2)
    }
}

impl<T> From<[T; 3]> for Vec3<T> {
    #[inline]
    fn from(value: [T; 3]) -> Self {
        let [x, y, z] = value;
        Self::new(x, y, z)
    }
}

impl<T> From<Vec4<T>> for Vec3<T> {
    #[inline]
    fn from(value: Vec4<T>) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

impl<T: Default> From<Vec2<T>> for Vec3<T> {
    #[inline]
    fn from(value: Vec2<T>) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: T::default(),
        }
    }
}

mod add {
    use super::Vec3;
    use core::ops::Add;

    impl<T> Vec3<T>
    where
        T: Copy + Add<Output = T>,
    {
        #[inline]
        fn add_inner(self, rhs: &Self) -> Self {
            Self {
                x: self.x + rhs.x,
                y: self.y + rhs.y,
                z: self.z + rhs.z,
            }
        }
    }

    impl<T> Add for Vec3<T>
    where
        T: Copy + Add<Output = T>,
    {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            self.add_inner(&rhs)
        }
    }

    impl<T> Add for &Vec3<T>
    where
        T: Copy + Add<Output = T>,
    {
        type Output = Vec3<T>;

        fn add(self, rhs: Self) -> Self::Output {
            self.add_inner(rhs)
        }
    }

    impl<T> Add<Vec3<T>> for &Vec3<T>
    where
        T: Copy + Add<Output = T>,
    {
        type Output = Vec3<T>;

        fn add(self, rhs: Vec3<T>) -> Self::Output {
            self.add_inner(&rhs)
        }
    }

    impl<T> Add<&Self> for Vec3<T>
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
    use super::Vec3;
    use core::ops::Add;
    use std::ops::AddAssign;

    impl<T> Vec3<T>
    where
        T: Copy + Add<Output = T>,
    {
        #[inline]
        fn add_assing_inner(&mut self, rhs: &Self) {
            self.x = self.x + rhs.x;
            self.y = self.y + rhs.y;
            self.z = self.z + rhs.z;
        }
    }

    impl<T> AddAssign for Vec3<T>
    where
        T: Copy + Add<Output = T>,
    {
        fn add_assign(&mut self, rhs: Self) {
            Self::add_assing_inner(self, &rhs);
        }
    }

    impl<T> AddAssign<&Self> for Vec3<T>
    where
        T: Copy + Add<Output = T>,
    {
        fn add_assign(&mut self, rhs: &Self) {
            Self::add_assing_inner(self, rhs);
        }
    }
}

mod sub {
    use super::Vec3;
    use core::ops::Sub;

    impl<T> Vec3<T>
    where
        T: Copy + Sub<Output = T>,
    {
        #[inline]
        fn sub_inner(self, rhs: &Self) -> Self {
            Self {
                x: self.x - rhs.x,
                y: self.y - rhs.y,
                z: self.z - rhs.z,
            }
        }
    }

    impl<T> Sub for Vec3<T>
    where
        T: Copy + Sub<Output = T>,
    {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            self.sub_inner(&rhs)
        }
    }

    impl<T> Sub for &Vec3<T>
    where
        T: Copy + Sub<Output = T>,
    {
        type Output = Vec3<T>;

        fn sub(self, rhs: Self) -> Self::Output {
            self.sub_inner(rhs)
        }
    }

    impl<T> Sub<Vec3<T>> for &Vec3<T>
    where
        T: Copy + Sub<Output = T>,
    {
        type Output = Vec3<T>;

        fn sub(self, rhs: Vec3<T>) -> Self::Output {
            self.sub_inner(&rhs)
        }
    }

    impl<T> Sub<&Self> for Vec3<T>
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
    use super::Vec3;
    use core::ops::Sub;
    use std::ops::SubAssign;

    impl<T> Vec3<T>
    where
        T: Copy + Sub<Output = T>,
    {
        #[inline]
        fn sub_assing_inner(&mut self, rhs: &Self) {
            self.x = self.x - rhs.x;
            self.y = self.y - rhs.y;
            self.z = self.z - rhs.z;
        }
    }

    impl<T> SubAssign for Vec3<T>
    where
        T: Copy + Sub<Output = T>,
    {
        fn sub_assign(&mut self, rhs: Self) {
            Self::sub_assing_inner(self, &rhs);
        }
    }

    impl<T> SubAssign<&Self> for Vec3<T>
    where
        T: Copy + Sub<Output = T>,
    {
        fn sub_assign(&mut self, rhs: &Self) {
            Self::sub_assing_inner(self, rhs);
        }
    }
}

mod scalar {
    use super::Vec3;
    use core::ops::Mul;

    impl<T> Vec3<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        #[inline]
        fn scalar(&self, scalar: &T) -> Self {
            Self {
                x: self.x * scalar,
                y: self.y * scalar,
                z: self.z * scalar,
            }
        }
    }

    impl<T> Mul<T> for Vec3<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Self;

        fn mul(self, rhs: T) -> Self::Output {
            self.scalar(&rhs)
        }
    }

    impl<T> Mul<T> for &Vec3<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Vec3<T>;

        fn mul(self, rhs: T) -> Self::Output {
            self.scalar(&rhs)
        }
    }

    impl<T> Mul<&T> for &Vec3<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Vec3<T>;

        fn mul(self, rhs: &T) -> Self::Output {
            self.scalar(rhs)
        }
    }

    impl<T> Mul<&T> for Vec3<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Self;

        fn mul(self, rhs: &T) -> Self::Output {
            self.scalar(rhs)
        }
    }

    impl Mul<Vec3<Self>> for f32 {
        type Output = Vec3<Self>;

        fn mul(self, rhs: Vec3<Self>) -> Self::Output {
            rhs.scalar(&self)
        }
    }

    impl Mul<Vec3<Self>> for u32 {
        type Output = Vec3<Self>;

        fn mul(self, rhs: Vec3<Self>) -> Self::Output {
            rhs.scalar(&self)
        }
    }
}

mod scalar_assign {
    use super::Vec3;
    use core::ops::Mul;
    use std::ops::MulAssign;

    impl<T> Vec3<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        #[inline]
        fn scalar_assign(&mut self, scalar: &T) {
            self.x = self.x * scalar;
            self.y = self.y * scalar;
            self.z = self.z * scalar;
        }
    }

    impl<T> MulAssign<T> for Vec3<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        fn mul_assign(&mut self, rhs: T) {
            Self::scalar_assign(self, &rhs);
        }
    }

    impl<T> MulAssign<&T> for Vec3<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        fn mul_assign(&mut self, rhs: &T) {
            Self::scalar_assign(self, rhs);
        }
    }
}

mod neg {
    use super::Vec3;
    use std::ops::Neg;

    impl<T: Neg<Output = T>> Neg for &Vec3<T>
    where
        for<'a> T: Copy + Neg<Output = T>,
    {
        type Output = Vec3<T>;

        #[inline]
        fn neg(self) -> Self::Output {
            Vec3 {
                x: -self.x,
                y: -self.y,
                z: -self.z,
            }
        }
    }

    impl<T: Neg<Output = T>> Neg for Vec3<T>
    where
        T: Neg<Output = T>,
    {
        type Output = Self;

        #[inline]
        fn neg(self) -> Self::Output {
            Vec3 {
                x: -self.x,
                y: -self.y,
                z: -self.z,
            }
        }
    }
}

impl Vec3<f32> {
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    #[inline]
    #[must_use]
    pub const fn one() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }

    #[inline]
    #[must_use]
    pub const fn up() -> Self {
        let up = Vec2::up();
        Self {
            x: up.x,
            y: up.y,
            z: 0.0,
        }
    }

    #[inline]
    #[must_use]
    pub const fn down() -> Self {
        let down = Vec2::down();
        Self {
            x: down.x,
            y: down.y,
            z: 0.0,
        }
    }

    #[inline]
    #[must_use]
    pub const fn left() -> Self {
        let left = Vec2::left();
        Self {
            x: left.x,
            y: left.y,
            z: 0.0,
        }
    }

    #[inline]
    #[must_use]
    pub const fn right() -> Self {
        let right = Vec2::up();
        Self {
            x: right.x,
            y: right.y,
            z: 0.0,
        }
    }

    #[inline]
    #[must_use]
    pub const fn forward() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: -1.0,
        }
    }

    #[inline]
    #[must_use]
    pub const fn backward() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    }

    #[inline]
    #[must_use]
    pub fn dot(&self, rhs: &Self) -> f32 {
        //  self.z * rhs.z + self.x * rhs.x + self.y * rhs.y
        self.z.mul_add(rhs.z, self.x.mul_add(rhs.x, self.y * rhs.y))
    }

    #[inline]
    #[must_use]
    pub fn distance(&self, rhs: &Self) -> f32 {
        (self - rhs).lenght()
    }

    #[inline]
    #[must_use]
    pub fn cross(&self, rhs: &Self) -> Self {
        /*
        Self {
            x: self.y * rhs.z - (self.z * rhs.y),
            y: self.z * rhs.x - (self.x * rhs.z),
            z: self.x * rhs.y - (self.y * rhs.x),
        };
        */

        Self {
            x: self.y.mul_add(rhs.z, -(self.z * rhs.y)),
            y: self.z.mul_add(rhs.x, -(self.x * rhs.z)),
            z: self.x.mul_add(rhs.y, -(self.y * rhs.x)),
        }
    }

    #[inline]
    #[must_use]
    pub fn lenght_scale(&self) -> f32 {
        // self.z * self.z + self.x * self.x + self.y * self.y
        self.z
            .mul_add(self.z, self.x.mul_add(self.x, self.y * self.y))
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

        [x1, x2, x3, x4, y1, y2, y3, y4, z1, z2, z3, z4]
    }
}
