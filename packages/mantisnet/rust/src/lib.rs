//! The MantisNet family's native layer: Python bindings over the vendored
//! research crates. All real logic lives in `rust/vendor/*`; this crate only
//! marshals. The bindings compile under the `python` feature (the maturin
//! build turns it on).

#[cfg(feature = "python")]
mod py;
#[cfg(feature = "python")]
mod search_py;
