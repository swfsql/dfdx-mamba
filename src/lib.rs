#![allow(incomplete_features)]
#![cfg_attr(feature = "nightly", feature(generic_const_exprs))]

#[cfg(feature = "nightly")]
pub mod layer;

#[cfg(feature = "nightly")]
pub use layer::{
    stateful::{MambaStateCache, MambaStateCacheConfig},
    MambaBlock, MambaBlockConfig, MambaBlockConstConfig,
};
