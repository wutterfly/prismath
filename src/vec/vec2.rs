use super::Vec3;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

impl<T> Vec2<T> {
    #[inline]
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn parts(self) -> (T, T) {
        (self.x, self.y)
    }

    pub fn as_array(self) -> [T; 2] {
        [self.x, self.y]
    }
}

impl<T> core::ops::Index<usize> for Vec2<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Index out of bound: {index}"),
        }
    }
}

impl<T> core::ops::IndexMut<usize> for Vec2<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Index out of bound: {index}"),
        }
    }
}

impl<T: Copy> Vec2<T> {
    #[inline]
    pub const fn splat(value: T) -> Self {
        Self { x: value, y: value }
    }
}

impl<T> From<(T, T)> for Vec2<T> {
    fn from(value: (T, T)) -> Self {
        Self::new(value.0, value.1)
    }
}

impl<T> From<[T; 2]> for Vec2<T> {
    fn from(value: [T; 2]) -> Self {
        let [x, y] = value;
        Self::new(x, y)
    }
}

impl<T> From<Vec3<T>> for Vec2<T> {
    fn from(value: Vec3<T>) -> Self {
        Self {
            x: value.x,
            y: value.y,
        }
    }
}

mod add {
    use super::Vec2;
    use core::ops::Add;

    impl<T> Vec2<T>
    where
        T: Copy + Add<Output = T>,
    {
        fn add_inner(self, rhs: &Self) -> Self {
            Self {
                x: self.x + rhs.x,
                y: self.y + rhs.y,
            }
        }
    }

    impl<T> Add for Vec2<T>
    where
        T: Copy + Add<Output = T>,
    {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            self.add_inner(&rhs)
        }
    }

    impl<T> Add for &Vec2<T>
    where
        T: Copy + Add<Output = T>,
    {
        type Output = Vec2<T>;

        fn add(self, rhs: Self) -> Self::Output {
            self.add_inner(rhs)
        }
    }

    impl<T> Add<Vec2<T>> for &Vec2<T>
    where
        T: Copy + Add<Output = T>,
    {
        type Output = Vec2<T>;

        fn add(self, rhs: Vec2<T>) -> Self::Output {
            self.add_inner(&rhs)
        }
    }

    impl<T> Add<&Self> for Vec2<T>
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
    use super::Vec2;
    use core::ops::Add;
    use core::ops::AddAssign;

    impl<T> Vec2<T>
    where
        T: Copy + Add<Output = T>,
    {
        #[inline]
        fn add_assing_inner(&mut self, rhs: &Self) {
            self.x = self.x + rhs.x;
            self.y = self.y + rhs.y;
        }
    }

    impl<T> AddAssign for Vec2<T>
    where
        T: Copy + Add<Output = T>,
    {
        fn add_assign(&mut self, rhs: Self) {
            Self::add_assing_inner(self, &rhs);
        }
    }

    impl<T> AddAssign<&Self> for Vec2<T>
    where
        T: Copy + Add<Output = T>,
    {
        fn add_assign(&mut self, rhs: &Self) {
            Self::add_assing_inner(self, rhs);
        }
    }
}

mod sub {
    use super::Vec2;
    use core::ops::Sub;

    impl<T> Vec2<T>
    where
        T: Copy + Sub<Output = T>,
    {
        fn sub_inner(self, rhs: &Self) -> Self {
            Self {
                x: self.x - rhs.x,
                y: self.y - rhs.y,
            }
        }
    }

    impl<T> Sub for Vec2<T>
    where
        T: Copy + Sub<Output = T>,
    {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            self.sub_inner(&rhs)
        }
    }

    impl<T> Sub for &Vec2<T>
    where
        T: Copy + Sub<Output = T>,
    {
        type Output = Vec2<T>;

        fn sub(self, rhs: Self) -> Self::Output {
            self.sub_inner(rhs)
        }
    }

    impl<T> Sub<Vec2<T>> for &Vec2<T>
    where
        T: Copy + Sub<Output = T>,
    {
        type Output = Vec2<T>;

        fn sub(self, rhs: Vec2<T>) -> Self::Output {
            self.sub_inner(&rhs)
        }
    }

    impl<T> Sub<&Self> for Vec2<T>
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
    use super::Vec2;
    use core::ops::Sub;
    use core::ops::SubAssign;

    impl<T> Vec2<T>
    where
        T: Copy + Sub<Output = T>,
    {
        #[inline]
        fn sub_assing_inner(&mut self, rhs: &Self) {
            self.x = self.x - rhs.x;
            self.y = self.y - rhs.y;
        }
    }

    impl<T> SubAssign for Vec2<T>
    where
        T: Copy + Sub<Output = T>,
    {
        fn sub_assign(&mut self, rhs: Self) {
            Self::sub_assing_inner(self, &rhs);
        }
    }

    impl<T> SubAssign<&Self> for Vec2<T>
    where
        T: Copy + Sub<Output = T>,
    {
        fn sub_assign(&mut self, rhs: &Self) {
            Self::sub_assing_inner(self, rhs);
        }
    }
}

mod scalar {
    use super::Vec2;
    use core::ops::Mul;

    impl<T> Vec2<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        #[inline]
        fn scalar(&self, scalar: &T) -> Self {
            Self {
                x: self.x * scalar,
                y: self.y * scalar,
            }
        }
    }

    impl<T> Mul<T> for Vec2<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Self;

        fn mul(self, rhs: T) -> Self::Output {
            self.scalar(&rhs)
        }
    }

    impl<T> Mul<T> for &Vec2<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Vec2<T>;

        fn mul(self, rhs: T) -> Self::Output {
            self.scalar(&rhs)
        }
    }

    impl<T> Mul<&T> for &Vec2<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Vec2<T>;

        fn mul(self, rhs: &T) -> Self::Output {
            self.scalar(rhs)
        }
    }

    impl<T> Mul<&T> for Vec2<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        type Output = Self;

        fn mul(self, rhs: &T) -> Self::Output {
            self.scalar(rhs)
        }
    }

    impl Mul<Vec2<Self>> for f32 {
        type Output = Vec2<Self>;

        fn mul(self, rhs: Vec2<Self>) -> Self::Output {
            rhs.scalar(&self)
        }
    }

    impl Mul<Vec2<Self>> for u32 {
        type Output = Vec2<Self>;

        fn mul(self, rhs: Vec2<Self>) -> Self::Output {
            rhs.scalar(&self)
        }
    }
}

mod scalar_assign {
    use super::Vec2;
    use core::ops::Mul;
    use core::ops::MulAssign;

    impl<T> Vec2<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        #[inline]
        fn scalar_assign(&mut self, scalar: &T) {
            self.x = self.x * scalar;
            self.y = self.y * scalar;
        }
    }

    impl<T> MulAssign<T> for Vec2<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        fn mul_assign(&mut self, rhs: T) {
            Self::scalar_assign(self, &rhs);
        }
    }

    impl<T> MulAssign<&T> for Vec2<T>
    where
        for<'a> T: Copy + Mul<&'a T, Output = T>,
    {
        fn mul_assign(&mut self, rhs: &T) {
            Self::scalar_assign(self, rhs);
        }
    }
}

mod neg {
    use super::Vec2;
    use core::ops::Neg;

    impl<T> Neg for &Vec2<T>
    where
        for<'a> T: Copy + Neg<Output = T>,
    {
        type Output = Vec2<T>;

        #[inline]
        fn neg(self) -> Self::Output {
            Vec2 {
                x: -self.x,
                y: -self.y,
            }
        }
    }

    impl<T> Neg for Vec2<T>
    where
        T: Neg<Output = T>,
    {
        type Output = Self;

        #[inline]
        fn neg(self) -> Self::Output {
            Self {
                x: -self.x,
                y: -self.y,
            }
        }
    }
}

impl Vec2<f32> {
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    #[inline]
    #[must_use]
    pub const fn one() -> Self {
        Self::new(1.0, 1.0)
    }

    #[inline]
    #[must_use]
    pub const fn up() -> Self {
        Self { x: 0., y: 1. }
    }

    #[inline]
    #[must_use]
    pub const fn down() -> Self {
        Self { x: 0., y: -1. }
    }

    #[inline]
    #[must_use]
    pub const fn left() -> Self {
        Self { x: -1., y: 0. }
    }

    #[inline]
    #[must_use]
    pub const fn right() -> Self {
        Self { x: 1., y: 0. }
    }

    #[inline]
    #[must_use]
    pub fn dot(&self, rhs: &Self) -> f32 {
        self.x.mul_add(rhs.x, self.y * rhs.y)
    }

    #[inline]
    #[must_use]
    pub fn distance(&self, rhs: &Self) -> f32 {
        (self - rhs).lenght()
    }

    #[inline]
    #[must_use]
    pub fn lenght_scale(&self) -> f32 {
        self.x.mul_add(self.x, self.y * self.y)
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
        }
    }

    #[inline]
    pub fn normalize(&mut self) {
        *self = self.normalized();
    }

    #[inline]
    #[must_use]
    pub const fn as_bytes(&self) -> [u8; core::mem::size_of::<Self>()] {
        let [x1, x2, x3, x4] = self.x.to_ne_bytes();
        let [y1, y2, y3, y4] = self.y.to_ne_bytes();

        [x1, x2, x3, x4, y1, y2, y3, y4]
    }
}

#[cfg(test)]
mod tests {
    use super::Vec2;

    #[test]
    fn test_vec2_scalar() {
        let vec1: Vec2<f32> = Vec2::new(1.0, 2.0);

        let out = 2.0 * vec1 * 2.0;
        assert_eq!(out, Vec2::new(4.0, 8.0));
    }
}
