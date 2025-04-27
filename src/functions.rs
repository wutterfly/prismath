use crate::traits;

pub fn round_to_nearest<N>(input: N, multiple: N) -> N
where
    N: std::ops::Add<Output = N>
        + std::ops::Sub<Output = N>
        + std::ops::BitAnd<Output = N>
        + std::ops::Not<Output = N>
        + traits::PositiveNumber
        + Copy,
{
    let multiple = multiple - N::one();
    (input + multiple) & !multiple
}

#[test]
fn test_round_to_nearest() {
    let a = 65;
    let multiple = 64;
    let result = round_to_nearest::<u64>(a, multiple);
    assert_eq!(result, 2 * 64);
}
