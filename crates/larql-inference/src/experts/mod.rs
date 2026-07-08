//! Experts: model-adjacent compute the forward pass dispatches into.
//!
//! Two families live here:
//! - **WASM tool experts** (`caller`/`loader`/`registry`/`session`/…): the
//!   model *emits* an op-call, the host parses and dispatches it into a
//!   sandboxed WASM unit (`docs/virtual-experts-dispatch.md`).
//! - **Virtual experts** (`virtual_expert` + `arith`): invisible to the
//!   model — a gate reads forward-pass exhaust, payloads are extracted
//!   through the model's I/O, compute is external and exact, and the answer
//!   is forced back through the sampler
//!   (`docs/specs/virtual-experts/arithmetic-virtual-expert.md`).

pub mod arith;
pub mod caller;
pub mod loader;
pub mod mask;
pub mod parser;
pub mod registry;
pub mod session;
pub mod virtual_expert;

pub use arith::{
    ave_generate_kquant, ArithAnswer, ArithmeticExpert, AveOptions, AveOutcome, AvePath,
    AveTelemetry,
};
pub use caller::{ExpertMetadata, ExpertResult, OpSpec};
pub use loader::load_expert;
pub use mask::OpNameMask;
pub use parser::{parse_op_call, OpCall};
pub use registry::{ExpertHandle, ExpertRegistry, WasmInfo};
pub use session::{DispatchOutcome, DispatchSkip, Dispatcher, ExpertSession, FilteredDispatcher};
pub use virtual_expert::{DriveSchedule, ExtractMiss, Fire, ResidualTap, Verdict, VirtualExpert};
