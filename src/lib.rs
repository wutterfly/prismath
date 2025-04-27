#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::many_single_char_names)]
// TODO: Remove this, once its stable
#![feature(stmt_expr_attributes)]

mod functions;
mod traits;

pub mod constants;

/// Row major matricies.
pub mod matrix;
pub mod quad;
pub mod transform;
pub mod vec;

pub use functions::round_to_nearest;
