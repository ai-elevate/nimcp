//! NIMCP Rust Bindings
//!
//! Safe Rust wrapper around the NIMCP C library (v2.6.3).
//! Uses only the public nimcp.h API via FFI.
//!
//! # Example
//! ```no_run
//! use nimcp::{Brain, BrainSize, BrainTask};
//!
//! nimcp::init().unwrap();
//! let mut brain = Brain::new("test", BrainSize::Tiny,
//!     BrainTask::Classification, 4, 2).unwrap();
//! brain.learn(&[1.0, 0.0, 0.5, 0.3], "cat", 0.9).unwrap();
//! let (label, confidence) = brain.predict(&[1.0, 0.0, 0.5, 0.3]).unwrap();
//! nimcp::shutdown();
//! ```

use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_float, c_int, c_uint, c_void};
use std::ptr;

// ============================================================================
// Error Type
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
pub enum NimcpError {
    Generic(String),
    NullArg(String),
    Invalid(String),
    Memory(String),
    Io(String),
    Unknown(i32, String),
}

impl fmt::Display for NimcpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NimcpError::Generic(m) => write!(f, "NIMCP error: {}", m),
            NimcpError::NullArg(m) => write!(f, "NIMCP null arg: {}", m),
            NimcpError::Invalid(m) => write!(f, "NIMCP invalid: {}", m),
            NimcpError::Memory(m) => write!(f, "NIMCP memory: {}", m),
            NimcpError::Io(m) => write!(f, "NIMCP I/O: {}", m),
            NimcpError::Unknown(c, m) => write!(f, "NIMCP error {}: {}", c, m),
        }
    }
}

impl std::error::Error for NimcpError {}

pub type Result<T> = std::result::Result<T, NimcpError>;

fn get_error_string() -> String {
    unsafe {
        let ptr = nimcp_get_error();
        if ptr.is_null() {
            "NIMCP error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

fn check_status(status: c_int) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let msg = get_error_string();
    Err(match status {
        1000 => NimcpError::Generic(msg),
        1003 => NimcpError::NullArg(msg),
        1004 => NimcpError::Invalid(msg),
        2000 => NimcpError::Memory(msg),
        4000 => NimcpError::Io(msg),
        _ => NimcpError::Unknown(status, msg),
    })
}

// ============================================================================
// Enums
// ============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BrainSize {
    Tiny = 0,
    Small = 1,
    Medium = 2,
    Large = 3,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum BrainTask {
    Classification = 0,
    Regression = 1,
    PatternMatching = 2,
    Sequence = 3,
    Association = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum NetworkType {
    Adaptive = 0,
    SNN = 1,
    LNN = 2,
    CNN = 3,
    Hybrid = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SNNTrainMethod {
    STDP = 0,
    RSTDP = 1,
    EProp = 2,
    Surrogate = 3,
    Homeostatic = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LNNTrainMethod {
    Adjoint = 0,
    BPTT = 1,
    RTRL = 2,
    EProp = 3,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum LossType {
    MSE = 0,
    CrossEntropy = 1,
    BinaryCE = 2,
    Huber = 3,
    MAE = 4,
    Focal = 5,
    KLDiv = 6,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum OptimizerType {
    SGD = 0,
    Momentum = 1,
    Adam = 2,
    AdamW = 3,
    RMSprop = 4,
    Adagrad = 5,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum SchedulerType {
    Constant = 0,
    Step = 1,
    Exponential = 2,
    Cosine = 3,
    WarmupCosine = 4,
    ReduceOnPlateau = 5,
    Cyclic = 6,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CallbackEvent {
    StepComplete = 0,
    EpochComplete = 1,
    LossComputed = 2,
    WeightsUpdated = 3,
    LRChanged = 4,
    Convergence = 5,
    Divergence = 6,
    Checkpoint = 7,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CallbackAction {
    Continue = 0,
    Stop = 1,
    Skip = 2,
    Rollback = 3,
    ReduceLR = 4,
    IncreaseLR = 5,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum CognitiveModule {
    None = 0,
    Perception = 1,
    WorkingMemory = 2,
    Executive = 3,
    TheoryOfMind = 4,
    Ethics = 5,
    Attention = 6,
    Emotion = 7,
    Salience = 8,
    Motor = 9,
    Language = 10,
    Metacognition = 11,
    Curiosity = 12,
    Introspection = 13,
    Predictive = 14,
    Consolidation = 15,
    EpisodicMemory = 16,
    SemanticMemory = 17,
    Wellbeing = 18,
    MentalHealth = 19,
    GoalMotivation = 20,
    CognitiveControl = 21,
    CustomStart = 100,
}

// ============================================================================
// FFI Structs
// ============================================================================

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TrainingConfig {
    pub loss_type: c_int,
    pub optimizer_type: c_int,
    pub scheduler_type: c_int,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub momentum: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub scheduler_step_size: u32,
    pub scheduler_gamma: f32,
    pub warmup_steps: u32,
    pub enable_gradient_clipping: bool,
    pub gradient_clip_value: f32,
    pub enable_biological_modulation: bool,
    pub biological_blend: f32,
    pub network_type: c_int,
    pub snn_method: c_int,
    pub snn_eligibility_tau: f32,
    pub snn_reward_tau: f32,
    pub snn_surrogate_beta: f32,
    pub lnn_method: c_int,
    pub lnn_bptt_truncation: u32,
    pub lnn_use_adjoint_checkpointing: bool,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct TrainingResult {
    pub loss: f32,
    pub learning_rate: f32,
    pub step: u32,
    pub early_stopped: bool,
    pub gradient_norm: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CallbackConfig {
    pub enable_auto_checkpoint: bool,
    pub checkpoint_interval: u32,
    pub enable_early_stopping: bool,
    pub patience: u32,
    pub min_delta: f32,
    pub divergence_threshold: f32,
    pub log_interval: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CallbackMetrics {
    pub step: u64,
    pub epoch: u64,
    pub loss: f32,
    pub loss_ema: f32,
    pub learning_rate: f32,
    pub gradient_norm: f32,
    pub step_time_us: u64,
    pub is_converging: bool,
    pub is_diverging: bool,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SnapshotInfo {
    pub name: [u8; 128],
    pub description: [u8; 512],
    pub timestamp: u64,
    pub file_size: u32,
    pub is_compressed: bool,
    pub is_encrypted: bool,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct BrainProbeData {
    pub task_name: [u8; 64],
    pub size: c_int,
    pub task: c_int,
    pub num_neurons: u32,
    pub num_synapses: u32,
    pub num_active_synapses: u32,
    pub total_inferences: u64,
    pub total_learning_steps: u64,
    pub avg_sparsity: f32,
    pub avg_inference_time_us: f32,
    pub current_learning_rate: f32,
    pub accuracy: f32,
    pub memory_bytes: usize,
    pub num_inputs: u32,
    pub num_outputs: u32,
    pub is_cow_clone: bool,
    pub cow_ref_count: u32,
    pub cow_shared_bytes: usize,
    pub cow_private_bytes: usize,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Phasor {
    pub amplitude: f32,
    pub phase: f32,
}

// ============================================================================
// Opaque Handle Types
// ============================================================================

#[repr(C)]
pub struct NimcpBrainHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct NimcpNetworkHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct NimcpEthicsHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct NimcpKnowledgeHandle {
    _private: [u8; 0],
}

#[repr(C)]
pub struct NimcpSnapshotHandle {
    _private: [u8; 0],
}

// ============================================================================
// FFI Callback Type
// ============================================================================

type NativeCallbackFn = extern "C" fn(
    event: c_int,
    metrics: *const CallbackMetrics,
    user_data: *mut c_void,
) -> c_int;

// ============================================================================
// FFI Declarations
// ============================================================================

extern "C" {
    // --- Library lifecycle ---
    fn nimcp_init() -> c_int;
    fn nimcp_shutdown();
    fn nimcp_version() -> *const c_char;
    fn nimcp_version_int() -> c_int;
    fn nimcp_get_error() -> *const c_char;

    // --- Brain core ---
    fn nimcp_brain_create(
        name: *const c_char, size: c_int, task: c_int,
        num_inputs: c_uint, num_outputs: c_uint,
    ) -> *mut NimcpBrainHandle;
    fn nimcp_brain_destroy(brain: *mut NimcpBrainHandle);
    fn nimcp_brain_learn_example(
        brain: *mut NimcpBrainHandle, features: *const c_float,
        num_features: c_uint, label: *const c_char, confidence: c_float,
    ) -> c_int;
    fn nimcp_brain_predict(
        brain: *mut NimcpBrainHandle, features: *const c_float,
        num_features: c_uint, out_label: *mut c_char, out_confidence: *mut c_float,
    ) -> c_int;
    fn nimcp_brain_infer(
        brain: *mut NimcpBrainHandle, features: *const c_float,
        num_features: c_uint, outputs: *mut c_float, num_outputs: c_uint,
    ) -> c_int;
    fn nimcp_brain_save(brain: *mut NimcpBrainHandle, filepath: *const c_char) -> c_int;
    fn nimcp_brain_load(filepath: *const c_char) -> *mut NimcpBrainHandle;
    fn nimcp_brain_create_from_config(filepath: *const c_char) -> *mut NimcpBrainHandle;

    // --- Brain training ---
    fn nimcp_training_config_default() -> TrainingConfig;
    fn nimcp_brain_configure_training(
        brain: *mut NimcpBrainHandle, config: *const TrainingConfig,
    ) -> c_int;
    fn nimcp_brain_train_step(
        brain: *mut NimcpBrainHandle, features: *const c_float,
        num_features: c_uint, targets: *const c_float, num_targets: c_uint,
        result: *mut TrainingResult,
    ) -> c_int;
    fn nimcp_brain_train_batch(
        brain: *mut NimcpBrainHandle, features: *const c_float,
        targets: *const c_float, batch_size: c_uint,
        num_features: c_uint, num_targets: c_uint,
        result: *mut TrainingResult,
    ) -> c_int;
    fn nimcp_brain_get_training_stats(
        brain: *mut NimcpBrainHandle, total_steps: *mut u64,
        total_loss: *mut c_float, current_lr: *mut c_float,
    ) -> c_int;
    fn nimcp_brain_step_scheduler(
        brain: *mut NimcpBrainHandle, validation_metric: c_float,
    ) -> c_float;

    // --- Brain callbacks ---
    fn nimcp_callback_config_default() -> CallbackConfig;
    fn nimcp_brain_enable_callbacks(
        brain: *mut NimcpBrainHandle, config: *const CallbackConfig,
    ) -> c_int;
    fn nimcp_brain_disable_callbacks(brain: *mut NimcpBrainHandle) -> c_int;
    fn nimcp_brain_register_callback(
        brain: *mut NimcpBrainHandle, event: c_int,
        callback: NativeCallbackFn, user_data: *mut c_void,
        name: *const c_char,
    ) -> c_uint;
    fn nimcp_brain_unregister_callback(
        brain: *mut NimcpBrainHandle, callback_id: c_uint,
    ) -> c_int;
    fn nimcp_brain_get_callback_stats(
        brain: *mut NimcpBrainHandle, total_fired: *mut u64,
        avg_time_us: *mut c_float, early_stops: *mut c_uint,
    ) -> c_int;

    // --- Brain resize ---
    fn nimcp_brain_resize(brain: *mut NimcpBrainHandle, count: c_uint) -> bool;
    fn nimcp_brain_auto_resize(brain: *mut NimcpBrainHandle) -> bool;
    fn nimcp_brain_get_neuron_count(brain: *mut NimcpBrainHandle) -> c_uint;
    fn nimcp_brain_get_utilization_metrics(
        brain: *mut NimcpBrainHandle, utilization: *mut c_float,
        saturation: *mut c_float,
    ) -> bool;

    // --- Brain named snapshots ---
    fn nimcp_brain_snapshot_save(
        brain: *mut NimcpBrainHandle, name: *const c_char,
        description: *const c_char,
    ) -> c_int;
    fn nimcp_brain_snapshot_restore(
        brain: *mut NimcpBrainHandle, name: *const c_char,
    ) -> *mut NimcpBrainHandle;
    fn nimcp_brain_snapshot_list(
        brain: *mut NimcpBrainHandle, infos: *mut SnapshotInfo,
        max_count: c_uint, out_count: *mut c_uint,
    ) -> c_int;
    fn nimcp_brain_snapshot_delete(
        brain: *mut NimcpBrainHandle, name: *const c_char,
    ) -> c_int;

    // --- Brain COW ---
    fn nimcp_brain_clone_cow(brain: *mut NimcpBrainHandle) -> *mut NimcpBrainHandle;
    fn nimcp_brain_snapshot_cow(brain: *mut NimcpBrainHandle) -> *mut NimcpSnapshotHandle;
    fn nimcp_brain_restore_cow(
        brain: *mut NimcpBrainHandle, snapshot: *mut NimcpSnapshotHandle,
    ) -> c_int;
    fn nimcp_brain_snapshot_destroy(snapshot: *mut NimcpSnapshotHandle);

    // --- Brain working memory ---
    fn nimcp_brain_working_memory_add(
        brain: *mut NimcpBrainHandle, data: *const c_float,
        size: c_uint, salience: c_float,
    ) -> c_int;
    fn nimcp_brain_working_memory_get(
        brain: *mut NimcpBrainHandle, index: c_uint, size_out: *mut c_uint,
    ) -> *const c_float;
    fn nimcp_brain_working_memory_stats(
        brain: *mut NimcpBrainHandle, current_size: *mut c_uint,
        capacity: *mut c_uint,
    ) -> c_int;
    fn nimcp_brain_working_memory_refresh(
        brain: *mut NimcpBrainHandle, index: c_uint,
    ) -> c_int;

    // --- Brain workspace ---
    fn nimcp_brain_workspace_compete(
        brain: *mut NimcpBrainHandle, module: c_int,
        content: *const c_float, content_dim: c_uint, strength: c_float,
    ) -> c_int;
    fn nimcp_brain_workspace_read(
        brain: *mut NimcpBrainHandle, content: *mut c_float,
        max_dim: c_uint, actual_dim: *mut c_uint,
        source_module: *mut c_int,
    ) -> c_int;
    fn nimcp_brain_workspace_subscribe(
        brain: *mut NimcpBrainHandle, module: c_int,
    ) -> c_int;
    fn nimcp_brain_workspace_unsubscribe(
        brain: *mut NimcpBrainHandle, module: c_int,
    ) -> c_int;
    fn nimcp_brain_workspace_has_broadcast(
        brain: *mut NimcpBrainHandle, has_broadcast: *mut bool,
    ) -> c_int;
    fn nimcp_brain_workspace_stats(
        brain: *mut NimcpBrainHandle, total_broadcasts: *mut c_uint,
        total_competitions: *mut c_uint, avg_strength: *mut c_float,
    ) -> c_int;

    // --- Brain oscillations ---
    fn nimcp_enable_complex_oscillations(
        brain: *mut NimcpBrainHandle, enable: bool,
    ) -> bool;
    fn nimcp_is_complex_oscillations_enabled(brain: *mut NimcpBrainHandle) -> bool;
    fn nimcp_get_oscillation_phasor(
        brain: *mut NimcpBrainHandle, neuron_id: c_uint,
    ) -> Phasor;
    fn nimcp_get_phase_coherence(
        brain: *mut NimcpBrainHandle, neuron_ids: *const c_uint, count: c_uint,
    ) -> c_float;
    fn nimcp_get_pac_modulation(
        brain: *mut NimcpBrainHandle, theta_freq: c_float, gamma_freq: c_float,
    ) -> c_float;

    // --- Brain probe ---
    fn nimcp_brain_probe(
        brain: *mut NimcpBrainHandle, probe: *mut BrainProbeData,
    ) -> c_int;
    fn nimcp_brain_broadcast_probe(brain: *mut NimcpBrainHandle) -> c_int;

    // --- Network ---
    fn nimcp_network_create(
        num_inputs: c_uint, num_outputs: c_uint,
        num_hidden: c_uint, learning_rate: c_float,
    ) -> *mut NimcpNetworkHandle;
    fn nimcp_network_destroy(network: *mut NimcpNetworkHandle);
    fn nimcp_network_forward(
        network: *mut NimcpNetworkHandle, inputs: *const c_float,
        num_inputs: c_uint, outputs: *mut c_float, num_outputs: c_uint,
    ) -> c_int;
    fn nimcp_network_train(
        network: *mut NimcpNetworkHandle, inputs: *const c_float,
        num_inputs: c_uint, targets: *const c_float, num_targets: c_uint,
    ) -> c_int;

    // --- Ethics ---
    fn nimcp_ethics_create() -> *mut NimcpEthicsHandle;
    fn nimcp_ethics_destroy(ethics: *mut NimcpEthicsHandle);
    fn nimcp_ethics_check(
        ethics: *mut NimcpEthicsHandle, situation: *const c_float,
        num_features: c_uint, out_score: *mut c_float,
    ) -> c_int;

    // --- Knowledge ---
    fn nimcp_knowledge_create() -> *mut NimcpKnowledgeHandle;
    fn nimcp_knowledge_destroy(knowledge: *mut NimcpKnowledgeHandle);
    fn nimcp_knowledge_add_fact(
        knowledge: *mut NimcpKnowledgeHandle,
        subject: *const c_char, predicate: *const c_char, object: *const c_char,
    ) -> c_int;
    fn nimcp_knowledge_query(
        knowledge: *mut NimcpKnowledgeHandle, query: *const c_char,
        out_result: *mut c_char, max_result_len: c_uint,
    ) -> c_int;
}

// ============================================================================
// Library Lifecycle
// ============================================================================

pub fn init() -> Result<()> {
    check_status(unsafe { nimcp_init() })
}

pub fn shutdown() {
    unsafe { nimcp_shutdown() }
}

pub fn version() -> String {
    unsafe {
        let ptr = nimcp_version();
        if ptr.is_null() {
            "unknown".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

pub fn version_int() -> i32 {
    unsafe { nimcp_version_int() }
}

pub fn training_config_default() -> TrainingConfig {
    unsafe { nimcp_training_config_default() }
}

pub fn callback_config_default() -> CallbackConfig {
    unsafe { nimcp_callback_config_default() }
}

// ============================================================================
// Safe Data Types
// ============================================================================

#[derive(Debug)]
pub struct Prediction {
    pub label: String,
    pub confidence: f32,
}

#[derive(Debug)]
pub struct TrainingStats {
    pub total_steps: u64,
    pub total_loss: f32,
    pub current_lr: f32,
}

#[derive(Debug)]
pub struct UtilizationMetrics {
    pub utilization: f32,
    pub saturation: f32,
}

#[derive(Debug)]
pub struct WorkspaceReadResult {
    pub content: Vec<f32>,
    pub actual_dim: u32,
    pub source_module: CognitiveModule,
}

#[derive(Debug)]
pub struct WorkspaceStats {
    pub total_broadcasts: u32,
    pub total_competitions: u32,
    pub avg_strength: f32,
}

#[derive(Debug)]
pub struct CallbackStats {
    pub total_fired: u64,
    pub avg_time_us: f32,
    pub early_stops: u32,
}

#[derive(Debug)]
pub struct WorkingMemoryStats {
    pub current_size: u32,
    pub capacity: u32,
}

#[derive(Debug)]
pub struct BrainProbe {
    pub task_name: String,
    pub size: BrainSize,
    pub task: BrainTask,
    pub num_neurons: u32,
    pub num_synapses: u32,
    pub num_active_synapses: u32,
    pub total_inferences: u64,
    pub total_learning_steps: u64,
    pub avg_sparsity: f32,
    pub avg_inference_time_us: f32,
    pub current_learning_rate: f32,
    pub accuracy: f32,
    pub memory_bytes: usize,
    pub num_inputs: u32,
    pub num_outputs: u32,
    pub is_cow_clone: bool,
    pub cow_ref_count: u32,
    pub cow_shared_bytes: usize,
    pub cow_private_bytes: usize,
}

// ============================================================================
// Brain
// ============================================================================

type CallbackBoxPtr =
    *mut Box<dyn Fn(CallbackEvent, &CallbackMetrics) -> CallbackAction>;

pub struct Brain {
    handle: *mut NimcpBrainHandle,
    // Raw pointers to heap-allocated callback closures — C holds these as user_data.
    // We free them only when the Brain is dropped.
    _callback_ptrs: Vec<CallbackBoxPtr>,
}

impl Brain {
    pub fn new(
        name: &str, size: BrainSize, task: BrainTask,
        num_inputs: u32, num_outputs: u32,
    ) -> Result<Self> {
        let c_name = CString::new(name).map_err(|e| NimcpError::Invalid(e.to_string()))?;
        unsafe {
            let handle = nimcp_brain_create(
                c_name.as_ptr(), size as c_int, task as c_int,
                num_inputs, num_outputs,
            );
            if handle.is_null() {
                Err(NimcpError::Generic(get_error_string()))
            } else {
                Ok(Brain { handle, _callback_ptrs: Vec::new() })
            }
        }
    }

    pub fn load(filepath: &str) -> Result<Self> {
        let c_path = CString::new(filepath).map_err(|e| NimcpError::Invalid(e.to_string()))?;
        unsafe {
            let handle = nimcp_brain_load(c_path.as_ptr());
            if handle.is_null() {
                Err(NimcpError::Io(format!("Failed to load: {}", filepath)))
            } else {
                Ok(Brain { handle, _callback_ptrs: Vec::new() })
            }
        }
    }

    pub fn create_from_config(filepath: &str) -> Result<Self> {
        let c_path = CString::new(filepath).map_err(|e| NimcpError::Invalid(e.to_string()))?;
        unsafe {
            let handle = nimcp_brain_create_from_config(c_path.as_ptr());
            if handle.is_null() {
                Err(NimcpError::Generic(get_error_string()))
            } else {
                Ok(Brain { handle, _callback_ptrs: Vec::new() })
            }
        }
    }

    // --- Core ---

    pub fn learn(&mut self, features: &[f32], label: &str, confidence: f32) -> Result<()> {
        let c_label = CString::new(label).map_err(|e| NimcpError::Invalid(e.to_string()))?;
        check_status(unsafe {
            nimcp_brain_learn_example(
                self.handle, features.as_ptr(), features.len() as c_uint,
                c_label.as_ptr(), confidence,
            )
        })
    }

    pub fn predict(&self, features: &[f32]) -> Result<Prediction> {
        let mut label_buf = vec![0u8; 64];
        let mut confidence: f32 = 0.0;
        check_status(unsafe {
            nimcp_brain_predict(
                self.handle, features.as_ptr(), features.len() as c_uint,
                label_buf.as_mut_ptr() as *mut c_char, &mut confidence,
            )
        })?;
        let label = unsafe {
            CStr::from_ptr(label_buf.as_ptr() as *const c_char)
                .to_string_lossy().into_owned()
        };
        Ok(Prediction { label, confidence })
    }

    pub fn infer(&self, features: &[f32], outputs: &mut [f32]) -> Result<()> {
        check_status(unsafe {
            nimcp_brain_infer(
                self.handle, features.as_ptr(), features.len() as c_uint,
                outputs.as_mut_ptr(), outputs.len() as c_uint,
            )
        })
    }

    pub fn save(&self, filepath: &str) -> Result<()> {
        let c_path = CString::new(filepath).map_err(|e| NimcpError::Invalid(e.to_string()))?;
        check_status(unsafe { nimcp_brain_save(self.handle, c_path.as_ptr()) })
    }

    // --- Training Pipeline ---

    pub fn configure_training(&mut self, config: &TrainingConfig) -> Result<()> {
        check_status(unsafe {
            nimcp_brain_configure_training(self.handle, config as *const TrainingConfig)
        })
    }

    pub fn train_step(&mut self, features: &[f32], targets: &[f32]) -> Result<TrainingResult> {
        let mut result = std::mem::MaybeUninit::<TrainingResult>::zeroed();
        check_status(unsafe {
            nimcp_brain_train_step(
                self.handle, features.as_ptr(), features.len() as c_uint,
                targets.as_ptr(), targets.len() as c_uint,
                result.as_mut_ptr(),
            )
        })?;
        Ok(unsafe { result.assume_init() })
    }

    pub fn train_batch(
        &mut self, features: &[f32], targets: &[f32],
        batch_size: u32, num_features: u32, num_targets: u32,
    ) -> Result<TrainingResult> {
        let mut result = std::mem::MaybeUninit::<TrainingResult>::zeroed();
        check_status(unsafe {
            nimcp_brain_train_batch(
                self.handle, features.as_ptr(), targets.as_ptr(),
                batch_size, num_features, num_targets,
                result.as_mut_ptr(),
            )
        })?;
        Ok(unsafe { result.assume_init() })
    }

    pub fn get_training_stats(&self) -> Result<TrainingStats> {
        let mut steps: u64 = 0;
        let mut loss: f32 = 0.0;
        let mut lr: f32 = 0.0;
        check_status(unsafe {
            nimcp_brain_get_training_stats(self.handle, &mut steps, &mut loss, &mut lr)
        })?;
        Ok(TrainingStats { total_steps: steps, total_loss: loss, current_lr: lr })
    }

    pub fn step_scheduler(&mut self, validation_metric: f32) -> f32 {
        unsafe { nimcp_brain_step_scheduler(self.handle, validation_metric) }
    }

    // --- Callbacks ---

    pub fn enable_callbacks(&mut self, config: &CallbackConfig) -> Result<()> {
        check_status(unsafe {
            nimcp_brain_enable_callbacks(self.handle, config as *const CallbackConfig)
        })
    }

    pub fn disable_callbacks(&mut self) -> Result<()> {
        check_status(unsafe { nimcp_brain_disable_callbacks(self.handle) })
    }

    pub fn register_callback<F>(
        &mut self, event: CallbackEvent, callback: F, name: Option<&str>,
    ) -> Result<u32>
    where
        F: Fn(CallbackEvent, &CallbackMetrics) -> CallbackAction + 'static,
    {
        let boxed: Box<dyn Fn(CallbackEvent, &CallbackMetrics) -> CallbackAction> =
            Box::new(callback);
        let raw = Box::into_raw(Box::new(boxed));

        extern "C" fn trampoline(
            event: c_int, metrics: *const CallbackMetrics, user_data: *mut c_void,
        ) -> c_int {
            let cb = unsafe {
                &*(user_data
                    as *const Box<dyn Fn(CallbackEvent, &CallbackMetrics) -> CallbackAction>)
            };
            let evt = unsafe { std::mem::transmute::<c_int, CallbackEvent>(event) };
            let m = if metrics.is_null() {
                &CallbackMetrics {
                    step: 0, epoch: 0, loss: 0.0, loss_ema: 0.0,
                    learning_rate: 0.0, gradient_norm: 0.0, step_time_us: 0,
                    is_converging: false, is_diverging: false,
                }
            } else {
                unsafe { &*metrics }
            };
            cb(evt, m) as c_int
        }

        let c_name = name.map(|n| CString::new(n).unwrap());
        let name_ptr = c_name.as_ref().map_or(ptr::null(), |s| s.as_ptr());

        let id = unsafe {
            nimcp_brain_register_callback(
                self.handle, event as c_int, trampoline,
                raw as *mut c_void, name_ptr,
            )
        };

        if id == 0 {
            // Clean up on failure
            let _ = unsafe { Box::from_raw(raw) };
            return Err(NimcpError::Generic("Failed to register callback".into()));
        }

        // Keep the raw pointer alive — C library holds it as user_data.
        // We'll free it in Brain::drop().
        self._callback_ptrs.push(raw);

        Ok(id)
    }

    pub fn unregister_callback(&mut self, callback_id: u32) -> Result<()> {
        check_status(unsafe {
            nimcp_brain_unregister_callback(self.handle, callback_id)
        })
    }

    pub fn get_callback_stats(&self) -> Result<CallbackStats> {
        let mut fired: u64 = 0;
        let mut avg: f32 = 0.0;
        let mut early: u32 = 0;
        check_status(unsafe {
            nimcp_brain_get_callback_stats(self.handle, &mut fired, &mut avg, &mut early)
        })?;
        Ok(CallbackStats { total_fired: fired, avg_time_us: avg, early_stops: early })
    }

    // --- Resize ---

    pub fn resize(&mut self, neuron_count: u32) -> bool {
        unsafe { nimcp_brain_resize(self.handle, neuron_count) }
    }

    pub fn auto_resize(&mut self) -> bool {
        unsafe { nimcp_brain_auto_resize(self.handle) }
    }

    pub fn get_neuron_count(&self) -> u32 {
        unsafe { nimcp_brain_get_neuron_count(self.handle) }
    }

    pub fn get_utilization_metrics(&self) -> Option<UtilizationMetrics> {
        let mut util: f32 = 0.0;
        let mut sat: f32 = 0.0;
        let ok = unsafe {
            nimcp_brain_get_utilization_metrics(self.handle, &mut util, &mut sat)
        };
        if ok { Some(UtilizationMetrics { utilization: util, saturation: sat }) } else { None }
    }

    // --- Named Snapshots ---

    pub fn snapshot_save(&self, name: &str, description: Option<&str>) -> Result<()> {
        let c_name = CString::new(name).map_err(|e| NimcpError::Invalid(e.to_string()))?;
        let c_desc = description.map(|d| CString::new(d).unwrap());
        let desc_ptr = c_desc.as_ref().map_or(ptr::null(), |s| s.as_ptr());
        check_status(unsafe {
            nimcp_brain_snapshot_save(self.handle, c_name.as_ptr(), desc_ptr)
        })
    }

    pub fn snapshot_restore(&self, name: &str) -> Result<Brain> {
        let c_name = CString::new(name).map_err(|e| NimcpError::Invalid(e.to_string()))?;
        unsafe {
            let h = nimcp_brain_snapshot_restore(self.handle, c_name.as_ptr());
            if h.is_null() {
                Err(NimcpError::Generic("Failed to restore snapshot".into()))
            } else {
                Ok(Brain { handle: h, _callback_ptrs: Vec::new() })
            }
        }
    }

    pub fn snapshot_delete(&self, name: &str) -> Result<()> {
        let c_name = CString::new(name).map_err(|e| NimcpError::Invalid(e.to_string()))?;
        check_status(unsafe {
            nimcp_brain_snapshot_delete(self.handle, c_name.as_ptr())
        })
    }

    // --- COW ---

    pub fn clone_cow(&self) -> Result<Brain> {
        unsafe {
            let h = nimcp_brain_clone_cow(self.handle);
            if h.is_null() {
                Err(NimcpError::Generic("Failed to clone brain".into()))
            } else {
                Ok(Brain { handle: h, _callback_ptrs: Vec::new() })
            }
        }
    }

    pub fn snapshot_cow(&self) -> Result<BrainSnapshot> {
        unsafe {
            let h = nimcp_brain_snapshot_cow(self.handle);
            if h.is_null() {
                Err(NimcpError::Generic("Failed to create snapshot".into()))
            } else {
                Ok(BrainSnapshot { handle: h })
            }
        }
    }

    pub fn restore_cow(&mut self, snapshot: &BrainSnapshot) -> Result<()> {
        check_status(unsafe {
            nimcp_brain_restore_cow(self.handle, snapshot.handle)
        })
    }

    // --- Working Memory ---

    pub fn working_memory_add(&mut self, data: &[f32], salience: f32) -> Result<()> {
        check_status(unsafe {
            nimcp_brain_working_memory_add(
                self.handle, data.as_ptr(), data.len() as c_uint, salience,
            )
        })
    }

    pub fn working_memory_get(&self, index: u32) -> Option<Vec<f32>> {
        let mut size: u32 = 0;
        let ptr = unsafe {
            nimcp_brain_working_memory_get(self.handle, index, &mut size)
        };
        if ptr.is_null() || size == 0 {
            None
        } else {
            let mut result = vec![0.0f32; size as usize];
            unsafe { ptr::copy_nonoverlapping(ptr, result.as_mut_ptr(), size as usize) };
            Some(result)
        }
    }

    pub fn working_memory_stats(&self) -> Result<WorkingMemoryStats> {
        let mut cur: u32 = 0;
        let mut cap: u32 = 0;
        check_status(unsafe {
            nimcp_brain_working_memory_stats(self.handle, &mut cur, &mut cap)
        })?;
        Ok(WorkingMemoryStats { current_size: cur, capacity: cap })
    }

    pub fn working_memory_refresh(&mut self, index: u32) -> Result<()> {
        check_status(unsafe {
            nimcp_brain_working_memory_refresh(self.handle, index)
        })
    }

    // --- Workspace ---

    pub fn workspace_compete(
        &mut self, module: CognitiveModule, content: &[f32], strength: f32,
    ) -> Result<()> {
        check_status(unsafe {
            nimcp_brain_workspace_compete(
                self.handle, module as c_int, content.as_ptr(),
                content.len() as c_uint, strength,
            )
        })
    }

    pub fn workspace_read(&self, max_dim: u32) -> Result<WorkspaceReadResult> {
        let mut buf = vec![0.0f32; max_dim as usize];
        let mut actual_dim: u32 = 0;
        let mut source: c_int = 0;
        check_status(unsafe {
            nimcp_brain_workspace_read(
                self.handle, buf.as_mut_ptr(), max_dim,
                &mut actual_dim, &mut source,
            )
        })?;
        buf.truncate(actual_dim as usize);
        Ok(WorkspaceReadResult {
            content: buf,
            actual_dim,
            source_module: unsafe { std::mem::transmute::<c_int, CognitiveModule>(source) },
        })
    }

    pub fn workspace_subscribe(&mut self, module: CognitiveModule) -> Result<()> {
        check_status(unsafe {
            nimcp_brain_workspace_subscribe(self.handle, module as c_int)
        })
    }

    pub fn workspace_unsubscribe(&mut self, module: CognitiveModule) -> Result<()> {
        check_status(unsafe {
            nimcp_brain_workspace_unsubscribe(self.handle, module as c_int)
        })
    }

    pub fn workspace_has_broadcast(&self) -> Result<bool> {
        let mut has: bool = false;
        check_status(unsafe {
            nimcp_brain_workspace_has_broadcast(self.handle, &mut has)
        })?;
        Ok(has)
    }

    pub fn workspace_stats(&self) -> Result<WorkspaceStats> {
        let mut bc: u32 = 0;
        let mut comp: u32 = 0;
        let mut avg: f32 = 0.0;
        check_status(unsafe {
            nimcp_brain_workspace_stats(self.handle, &mut bc, &mut comp, &mut avg)
        })?;
        Ok(WorkspaceStats {
            total_broadcasts: bc, total_competitions: comp, avg_strength: avg,
        })
    }

    // --- Oscillations ---

    pub fn enable_oscillations(&mut self, enable: bool) -> bool {
        unsafe { nimcp_enable_complex_oscillations(self.handle, enable) }
    }

    pub fn is_oscillations_enabled(&self) -> bool {
        unsafe { nimcp_is_complex_oscillations_enabled(self.handle) }
    }

    pub fn get_phasor(&self, neuron_id: u32) -> Phasor {
        unsafe { nimcp_get_oscillation_phasor(self.handle, neuron_id) }
    }

    pub fn get_phase_coherence(&self, neuron_ids: &[u32]) -> f32 {
        unsafe {
            nimcp_get_phase_coherence(
                self.handle, neuron_ids.as_ptr(), neuron_ids.len() as c_uint,
            )
        }
    }

    pub fn get_pac_modulation(&self, theta_freq: f32, gamma_freq: f32) -> f32 {
        unsafe { nimcp_get_pac_modulation(self.handle, theta_freq, gamma_freq) }
    }

    // --- Probe ---

    pub fn probe(&self) -> Result<BrainProbe> {
        let mut raw = std::mem::MaybeUninit::<BrainProbeData>::zeroed();
        check_status(unsafe {
            nimcp_brain_probe(self.handle, raw.as_mut_ptr())
        })?;
        let raw = unsafe { raw.assume_init() };
        let task_name = {
            let end = raw.task_name.iter().position(|&b| b == 0).unwrap_or(64);
            String::from_utf8_lossy(&raw.task_name[..end]).into_owned()
        };
        Ok(BrainProbe {
            task_name,
            size: unsafe { std::mem::transmute::<c_int, BrainSize>(raw.size) },
            task: unsafe { std::mem::transmute::<c_int, BrainTask>(raw.task) },
            num_neurons: raw.num_neurons,
            num_synapses: raw.num_synapses,
            num_active_synapses: raw.num_active_synapses,
            total_inferences: raw.total_inferences,
            total_learning_steps: raw.total_learning_steps,
            avg_sparsity: raw.avg_sparsity,
            avg_inference_time_us: raw.avg_inference_time_us,
            current_learning_rate: raw.current_learning_rate,
            accuracy: raw.accuracy,
            memory_bytes: raw.memory_bytes,
            num_inputs: raw.num_inputs,
            num_outputs: raw.num_outputs,
            is_cow_clone: raw.is_cow_clone,
            cow_ref_count: raw.cow_ref_count,
            cow_shared_bytes: raw.cow_shared_bytes,
            cow_private_bytes: raw.cow_private_bytes,
        })
    }

    pub fn broadcast_probe(&self) -> Result<()> {
        check_status(unsafe { nimcp_brain_broadcast_probe(self.handle) })
    }
}

impl Drop for Brain {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { nimcp_brain_destroy(self.handle) }
        }
        // Free callback closure allocations (C no longer references them after destroy)
        for ptr in self._callback_ptrs.drain(..) {
            if !ptr.is_null() {
                let _ = unsafe { Box::from_raw(ptr) };
            }
        }
    }
}

// ============================================================================
// BrainSnapshot (COW)
// ============================================================================

pub struct BrainSnapshot {
    handle: *mut NimcpSnapshotHandle,
}

impl Drop for BrainSnapshot {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { nimcp_brain_snapshot_destroy(self.handle) }
        }
    }
}

// ============================================================================
// Network
// ============================================================================

pub struct Network {
    handle: *mut NimcpNetworkHandle,
    num_outputs: usize,
}

impl Network {
    pub fn new(
        num_inputs: u32, num_outputs: u32, num_hidden: u32, learning_rate: f32,
    ) -> Result<Self> {
        unsafe {
            let handle = nimcp_network_create(num_inputs, num_outputs, num_hidden, learning_rate);
            if handle.is_null() {
                Err(NimcpError::Generic(get_error_string()))
            } else {
                Ok(Network { handle, num_outputs: num_outputs as usize })
            }
        }
    }

    pub fn forward(&self, inputs: &[f32]) -> Result<Vec<f32>> {
        let mut outputs = vec![0.0f32; self.num_outputs];
        check_status(unsafe {
            nimcp_network_forward(
                self.handle, inputs.as_ptr(), inputs.len() as c_uint,
                outputs.as_mut_ptr(), outputs.len() as c_uint,
            )
        })?;
        Ok(outputs)
    }

    pub fn train(&mut self, inputs: &[f32], targets: &[f32]) -> Result<()> {
        check_status(unsafe {
            nimcp_network_train(
                self.handle, inputs.as_ptr(), inputs.len() as c_uint,
                targets.as_ptr(), targets.len() as c_uint,
            )
        })
    }
}

impl Drop for Network {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { nimcp_network_destroy(self.handle) }
        }
    }
}

// ============================================================================
// Ethics
// ============================================================================

pub struct Ethics {
    handle: *mut NimcpEthicsHandle,
}

impl Ethics {
    pub fn new() -> Result<Self> {
        unsafe {
            let handle = nimcp_ethics_create();
            if handle.is_null() {
                Err(NimcpError::Generic(get_error_string()))
            } else {
                Ok(Ethics { handle })
            }
        }
    }

    pub fn check(&self, situation: &[f32]) -> Result<f32> {
        let mut score: f32 = 0.0;
        check_status(unsafe {
            nimcp_ethics_check(
                self.handle, situation.as_ptr(),
                situation.len() as c_uint, &mut score,
            )
        })?;
        Ok(score)
    }
}

impl Drop for Ethics {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { nimcp_ethics_destroy(self.handle) }
        }
    }
}

// ============================================================================
// KnowledgeGraph
// ============================================================================

pub struct KnowledgeGraph {
    handle: *mut NimcpKnowledgeHandle,
}

impl KnowledgeGraph {
    pub fn new() -> Result<Self> {
        unsafe {
            let handle = nimcp_knowledge_create();
            if handle.is_null() {
                Err(NimcpError::Generic(get_error_string()))
            } else {
                Ok(KnowledgeGraph { handle })
            }
        }
    }

    pub fn add_fact(&mut self, subject: &str, predicate: &str, object: &str) -> Result<()> {
        let c_sub = CString::new(subject).map_err(|e| NimcpError::Invalid(e.to_string()))?;
        let c_pred = CString::new(predicate).map_err(|e| NimcpError::Invalid(e.to_string()))?;
        let c_obj = CString::new(object).map_err(|e| NimcpError::Invalid(e.to_string()))?;
        check_status(unsafe {
            nimcp_knowledge_add_fact(
                self.handle, c_sub.as_ptr(), c_pred.as_ptr(), c_obj.as_ptr(),
            )
        })
    }

    pub fn query(&self, query: &str, max_len: u32) -> Result<String> {
        let c_query = CString::new(query).map_err(|e| NimcpError::Invalid(e.to_string()))?;
        let mut buf = vec![0u8; max_len as usize];
        check_status(unsafe {
            nimcp_knowledge_query(
                self.handle, c_query.as_ptr(),
                buf.as_mut_ptr() as *mut c_char, max_len,
            )
        })?;
        let result = unsafe {
            CStr::from_ptr(buf.as_ptr() as *const c_char)
                .to_string_lossy().into_owned()
        };
        Ok(result)
    }
}

impl Drop for KnowledgeGraph {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { nimcp_knowledge_destroy(self.handle) }
        }
    }
}
