pub trait Number {
    fn zero() -> Self
    where
        Self: Sized;

    fn one() -> Self
    where
        Self: Sized;

    fn min() -> Self
    where
        Self: Sized;

    fn max() -> Self
    where
        Self: Sized;
}

macro_rules! number_impl {
    ($ty: tt, $zero: expr, $one: expr, $min: expr, $max: expr) => {
        impl Number for $ty {
            fn zero() -> Self
            where
                Self: Sized,
            {
                $zero
            }

            fn one() -> Self
            where
                Self: Sized,
            {
                $one
            }

            fn min() -> Self
            where
                Self: Sized,
            {
                $min
            }

            fn max() -> Self
            where
                Self: Sized,
            {
                $max
            }
        }
    };
}

number_impl!(u8, 0, 1, u8::MIN, u8::MAX);
number_impl!(u16, 0, 1, u16::MIN, u16::MAX);
number_impl!(u32, 0, 1, u32::MIN, u32::MAX);
number_impl!(u64, 0, 1, u64::MIN, u64::MAX);
number_impl!(u128, 0, 1, u128::MIN, u128::MAX);
number_impl!(usize, 0, 1, usize::MIN, usize::MAX);

number_impl!(i8, 0, 1, i8::MIN, i8::MAX);
number_impl!(i16, 0, 1, i16::MIN, i16::MAX);
number_impl!(i32, 0, 1, i32::MIN, i32::MAX);
number_impl!(i64, 0, 1, i64::MIN, i64::MAX);
number_impl!(i128, 0, 1, i128::MIN, i128::MAX);
number_impl!(isize, 0, 1, isize::MIN, isize::MAX);

number_impl!(f32, 0.0, 1.0, f32::MIN, f32::MAX);
number_impl!(f64, 0.0, 1.0, f64::MIN, f64::MAX);

pub trait PositiveNumber: Number {}

impl PositiveNumber for u8 {}
impl PositiveNumber for u16 {}
impl PositiveNumber for u32 {}
impl PositiveNumber for u64 {}
impl PositiveNumber for u128 {}
impl PositiveNumber for usize {}
