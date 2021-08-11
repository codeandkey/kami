/// Constants used (mostly model architecture)
/// While many of these values are passed in as parameters,
/// they are not really convienent to load from a params struct.
/// Instead we just verify that the loaded params match the values here and fail if they don't.

pub const PORT: u16 = 8124;

pub const FRAME_COUNT: usize = 6;
pub const FRAME_SIZE: usize = 14;
pub const HEADER_SIZE: usize = 18;

pub const TOTAL_FRAMES_SIZE: usize = FRAME_COUNT * 64 * FRAME_SIZE;