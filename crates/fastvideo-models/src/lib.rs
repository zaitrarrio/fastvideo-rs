//! Wan/FastWan model components. Forwards are generic over `TensorBackend`.

pub mod schedulers;
pub mod wan;

pub use schedulers::{DmdSchedule, FlowMatchEulerDiscreteScheduler};
pub use wan::WanVideoArchConfig;
