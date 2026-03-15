// Package nimcp provides Go bindings for the NIMCP C library via CGo.
//
// This package wraps the entire public C API (nimcp.h v2.6.3) including:
//   - Brain: high-level learning, training pipeline, callbacks, COW, workspace
//   - Network: low-level neural network
//   - Ethics: ethical evaluation
//   - KnowledgeGraph: fact storage and querying
//
// All handle types implement io.Closer for deterministic resource cleanup.
package nimcp

/*
#cgo CFLAGS: -I${SRCDIR}/../../../include -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR}/../../../build/lib -lnimcp -Wl,-rpath,${SRCDIR}/../../../build/lib
#include "nimcp.h"
#include "nimcp_go_helpers.h"
#include <stdlib.h>
#include <string.h>

// Forward declaration of Go callback dispatcher (defined via //export below)
// Note: no 'const' — CGo //export doesn't support const pointer params
extern nimcp_callback_action_t goCallbackTrampoline(
    nimcp_callback_event_t event,
    nimcp_callback_metrics_t* metrics,
    void* user_data);
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// ============================================================================
// Error Types
// ============================================================================

// ErrorCode represents NIMCP status codes.
type ErrorCode int

const (
	OK         ErrorCode = 0
	ErrGeneric ErrorCode = 1000
	ErrNullArg ErrorCode = 1003
	ErrInvalid ErrorCode = 1004
	ErrMemory  ErrorCode = 2000
	ErrIO      ErrorCode = 4000
)

// NimcpError represents an error returned by the NIMCP C library.
type NimcpError struct {
	Code    ErrorCode
	Message string
}

func (e *NimcpError) Error() string {
	return fmt.Sprintf("nimcp error %d: %s", e.Code, e.Message)
}

func checkStatus(status C.nimcp_status_t) error {
	if status == C.NIMCP_OK {
		return nil
	}
	code := ErrorCode(status)
	msg := C.GoString(C.nimcp_get_error())
	if msg == "" {
		msg = fmt.Sprintf("error code %d", code)
	}
	return &NimcpError{Code: code, Message: msg}
}

// ============================================================================
// Enums
// ============================================================================

// BrainSize represents brain size presets.
type BrainSize int

const (
	BrainTiny   BrainSize = 0
	BrainSmall  BrainSize = 1
	BrainMedium BrainSize = 2
	BrainLarge  BrainSize = 3
)

// BrainTask represents task templates.
type BrainTask int

const (
	TaskClassification  BrainTask = 0
	TaskRegression      BrainTask = 1
	TaskPatternMatching BrainTask = 2
	TaskSequence        BrainTask = 3
	TaskAssociation     BrainTask = 4
)

// NetworkType represents network architecture types.
type NetworkType int

const (
	NetworkAdaptive NetworkType = 0
	NetworkSNN      NetworkType = 1
	NetworkLNN      NetworkType = 2
	NetworkCNN      NetworkType = 3
	NetworkHybrid   NetworkType = 4
)

// SNNTrainMethod represents SNN training methods.
type SNNTrainMethod int

const (
	SNNTrainSTDP        SNNTrainMethod = 0
	SNNTrainRSTDP       SNNTrainMethod = 1
	SNNTrainEProp       SNNTrainMethod = 2
	SNNTrainSurrogate   SNNTrainMethod = 3
	SNNTrainHomeostatic SNNTrainMethod = 4
)

// LNNTrainMethod represents LNN training methods.
type LNNTrainMethod int

const (
	LNNTrainAdjoint LNNTrainMethod = 0
	LNNTrainBPTT    LNNTrainMethod = 1
	LNNTrainRTRL    LNNTrainMethod = 2
	LNNTrainEProp   LNNTrainMethod = 3
)

// LossType represents loss function types.
type LossType int

const (
	LossMSE          LossType = 0
	LossCrossEntropy LossType = 1
	LossBinaryCE     LossType = 2
	LossHuber        LossType = 3
	LossMAE          LossType = 4
	LossFocal        LossType = 5
	LossKLDiv        LossType = 6
)

// OptimizerType represents optimizer types.
type OptimizerType int

const (
	OptSGD      OptimizerType = 0
	OptMomentum OptimizerType = 1
	OptAdam     OptimizerType = 2
	OptAdamW    OptimizerType = 3
	OptRMSProp  OptimizerType = 4
	OptAdagrad  OptimizerType = 5
)

// SchedulerType represents LR scheduler types.
type SchedulerType int

const (
	SchedConstant        SchedulerType = 0
	SchedStep            SchedulerType = 1
	SchedExponential     SchedulerType = 2
	SchedCosine          SchedulerType = 3
	SchedWarmupCosine    SchedulerType = 4
	SchedReduceOnPlateau SchedulerType = 5
	SchedCyclic          SchedulerType = 6
)

// CallbackEvent represents training callback event types.
type CallbackEvent int

const (
	CBStepComplete   CallbackEvent = 0
	CBEpochComplete  CallbackEvent = 1
	CBLossComputed   CallbackEvent = 2
	CBWeightsUpdated CallbackEvent = 3
	CBLRChanged      CallbackEvent = 4
	CBConvergence    CallbackEvent = 5
	CBDivergence     CallbackEvent = 6
	CBCheckpoint     CallbackEvent = 7
)

// CallbackAction represents callback return actions.
type CallbackAction int

const (
	CallbackActionContinue   CallbackAction = 0
	CallbackActionStop       CallbackAction = 1
	CallbackActionSkip       CallbackAction = 2
	CallbackActionRollback   CallbackAction = 3
	CallbackActionReduceLR   CallbackAction = 4
	CallbackActionIncreaseLR CallbackAction = 5
)

// CognitiveModule represents cognitive module identifiers.
type CognitiveModule int

const (
	ModuleNone             CognitiveModule = 0
	ModulePerception       CognitiveModule = 1
	ModuleWorkingMemory    CognitiveModule = 2
	ModuleExecutive        CognitiveModule = 3
	ModuleTheoryOfMind     CognitiveModule = 4
	ModuleEthics           CognitiveModule = 5
	ModuleAttention        CognitiveModule = 6
	ModuleEmotion          CognitiveModule = 7
	ModuleSalience         CognitiveModule = 8
	ModuleMotor            CognitiveModule = 9
	ModuleLanguage         CognitiveModule = 10
	ModuleMetacognition    CognitiveModule = 11
	ModuleCuriosity        CognitiveModule = 12
	ModuleIntrospection    CognitiveModule = 13
	ModulePredictive       CognitiveModule = 14
	ModuleConsolidation    CognitiveModule = 15
	ModuleEpisodicMemory   CognitiveModule = 16
	ModuleSemanticMemory   CognitiveModule = 17
	ModuleWellbeing        CognitiveModule = 18
	ModuleMentalHealth     CognitiveModule = 19
	ModuleGoalMotivation   CognitiveModule = 20
	ModuleCognitiveControl CognitiveModule = 21
	ModuleCustomStart      CognitiveModule = 100
)

// ============================================================================
// Data Structures
// ============================================================================

// TrainingConfig holds training pipeline configuration.
type TrainingConfig struct {
	LossType      LossType
	OptimizerType OptimizerType
	SchedulerType SchedulerType

	LearningRate float32
	WeightDecay  float32
	Momentum     float32
	Beta1        float32
	Beta2        float32
	Epsilon      float32

	SchedulerStepSize uint32
	SchedulerGamma    float32
	WarmupSteps       uint32

	EnableGradientClipping bool
	GradientClipValue      float32

	EnableBiologicalModulation bool
	BiologicalBlend            float32

	NetworkType NetworkType

	SNNMethod        SNNTrainMethod
	SNNEligibilityTau float32
	SNNRewardTau      float32
	SNNSurrogateBeta  float32

	LNNMethod                  LNNTrainMethod
	LNNBPTTTruncation          uint32
	LNNUseAdjointCheckpointing bool
}

func (tc *TrainingConfig) toC() C.nimcp_training_config_t {
	var c C.nimcp_training_config_t
	c.loss_type = C.nimcp_api_loss_t(tc.LossType)
	c.optimizer_type = C.nimcp_api_optimizer_t(tc.OptimizerType)
	c.scheduler_type = C.nimcp_api_scheduler_t(tc.SchedulerType)
	c.learning_rate = C.float(tc.LearningRate)
	c.weight_decay = C.float(tc.WeightDecay)
	c.momentum = C.float(tc.Momentum)
	c.beta1 = C.float(tc.Beta1)
	c.beta2 = C.float(tc.Beta2)
	c.epsilon = C.float(tc.Epsilon)
	c.scheduler_step_size = C.uint(tc.SchedulerStepSize)
	c.scheduler_gamma = C.float(tc.SchedulerGamma)
	c.warmup_steps = C.uint(tc.WarmupSteps)
	c.enable_gradient_clipping = C.bool(tc.EnableGradientClipping)
	c.gradient_clip_value = C.float(tc.GradientClipValue)
	c.enable_biological_modulation = C.bool(tc.EnableBiologicalModulation)
	c.biological_blend = C.float(tc.BiologicalBlend)
	c.network_type = C.nimcp_network_type_t(tc.NetworkType)
	c.snn_method = C.nimcp_snn_train_method_t(tc.SNNMethod)
	c.snn_eligibility_tau = C.float(tc.SNNEligibilityTau)
	c.snn_reward_tau = C.float(tc.SNNRewardTau)
	c.snn_surrogate_beta = C.float(tc.SNNSurrogateBeta)
	c.lnn_method = C.nimcp_lnn_train_method_t(tc.LNNMethod)
	c.lnn_bptt_truncation = C.uint(tc.LNNBPTTTruncation)
	c.lnn_use_adjoint_checkpointing = C.bool(tc.LNNUseAdjointCheckpointing)
	return c
}

// DefaultTrainingConfig returns sensible default training configuration.
func DefaultTrainingConfig() TrainingConfig {
	c := C.nimcp_training_config_default()
	return TrainingConfig{
		LossType:                   LossType(c.loss_type),
		OptimizerType:              OptimizerType(c.optimizer_type),
		SchedulerType:              SchedulerType(c.scheduler_type),
		LearningRate:               float32(c.learning_rate),
		WeightDecay:                float32(c.weight_decay),
		Momentum:                   float32(c.momentum),
		Beta1:                      float32(c.beta1),
		Beta2:                      float32(c.beta2),
		Epsilon:                    float32(c.epsilon),
		SchedulerStepSize:          uint32(c.scheduler_step_size),
		SchedulerGamma:             float32(c.scheduler_gamma),
		WarmupSteps:                uint32(c.warmup_steps),
		EnableGradientClipping:     bool(c.enable_gradient_clipping),
		GradientClipValue:          float32(c.gradient_clip_value),
		EnableBiologicalModulation: bool(c.enable_biological_modulation),
		BiologicalBlend:            float32(c.biological_blend),
		NetworkType:                NetworkType(c.network_type),
		SNNMethod:                  SNNTrainMethod(c.snn_method),
		SNNEligibilityTau:          float32(c.snn_eligibility_tau),
		SNNRewardTau:               float32(c.snn_reward_tau),
		SNNSurrogateBeta:           float32(c.snn_surrogate_beta),
		LNNMethod:                  LNNTrainMethod(c.lnn_method),
		LNNBPTTTruncation:          uint32(c.lnn_bptt_truncation),
		LNNUseAdjointCheckpointing: bool(c.lnn_use_adjoint_checkpointing),
	}
}

// TrainingResult holds results from a training step.
type TrainingResult struct {
	Loss         float32
	LearningRate float32
	Step         uint32
	EarlyStopped bool
	GradientNorm float32
}

// CallbackConfig holds callback configuration.
type CallbackConfig struct {
	EnableAutoCheckpoint bool
	CheckpointInterval   uint32
	EnableEarlyStopping  bool
	Patience             uint32
	MinDelta             float32
	DivergenceThreshold  float32
	LogInterval          uint32
}

func (cc *CallbackConfig) toC() C.nimcp_callback_config_t {
	var c C.nimcp_callback_config_t
	c.enable_auto_checkpoint = C.bool(cc.EnableAutoCheckpoint)
	c.checkpoint_interval = C.uint(cc.CheckpointInterval)
	c.enable_early_stopping = C.bool(cc.EnableEarlyStopping)
	c.patience = C.uint(cc.Patience)
	c.min_delta = C.float(cc.MinDelta)
	c.divergence_threshold = C.float(cc.DivergenceThreshold)
	c.log_interval = C.uint(cc.LogInterval)
	return c
}

// DefaultCallbackConfig returns sensible default callback configuration.
func DefaultCallbackConfig() CallbackConfig {
	c := C.nimcp_callback_config_default()
	return CallbackConfig{
		EnableAutoCheckpoint: bool(c.enable_auto_checkpoint),
		CheckpointInterval:   uint32(c.checkpoint_interval),
		EnableEarlyStopping:  bool(c.enable_early_stopping),
		Patience:             uint32(c.patience),
		MinDelta:             float32(c.min_delta),
		DivergenceThreshold:  float32(c.divergence_threshold),
		LogInterval:          uint32(c.log_interval),
	}
}

// CallbackMetrics holds training metrics passed to callbacks.
type CallbackMetrics struct {
	Step         uint64
	Epoch        uint64
	Loss         float32
	LossEMA      float32
	LearningRate float32
	GradientNorm float32
	StepTimeUS   uint64
	IsConverging bool
	IsDiverging  bool
}

// SnapshotInfo holds snapshot metadata.
type SnapshotInfo struct {
	Name         string
	Description  string
	Timestamp    uint64
	FileSize     uint32
	IsCompressed bool
	IsEncrypted  bool
}

// BrainProbe holds comprehensive brain state statistics.
type BrainProbe struct {
	TaskName          string
	Size              BrainSize
	Task              BrainTask
	NumNeurons        uint32
	NumSynapses       uint32
	NumActiveSynapses uint32
	TotalInferences   uint64
	TotalLearningSteps uint64
	AvgSparsity       float32
	AvgInferenceTimeUS float32
	CurrentLearningRate float32
	Accuracy          float32
	MemoryBytes       uint64
	NumInputs         uint32
	NumOutputs        uint32
	IsCOWClone        bool
	COWRefCount       uint32
	COWSharedBytes    uint64
	COWPrivateBytes   uint64
}

// Phasor represents an oscillation phasor (amplitude + phase).
type Phasor struct {
	Amplitude float32
	Phase     float32
}

// ============================================================================
// Callback Registry (CGo callback trampoline)
// ============================================================================

// CallbackFunc is the Go callback function signature.
type CallbackFunc func(event CallbackEvent, metrics *CallbackMetrics) CallbackAction

var (
	callbackMu   sync.RWMutex
	callbackMap  = make(map[uint64]CallbackFunc)
	callbackNext uint64 = 1
)

func registerGoCallback(fn CallbackFunc) uint64 {
	callbackMu.Lock()
	defer callbackMu.Unlock()
	key := callbackNext
	callbackNext++
	callbackMap[key] = fn
	return key
}

func unregisterGoCallback(key uint64) {
	callbackMu.Lock()
	defer callbackMu.Unlock()
	delete(callbackMap, key)
}

//export goCallbackTrampoline
func goCallbackTrampoline(event C.nimcp_callback_event_t, metrics *C.nimcp_callback_metrics_t, userData unsafe.Pointer) C.nimcp_callback_action_t {
	key := uint64(uintptr(userData))
	callbackMu.RLock()
	cb, ok := callbackMap[key]
	callbackMu.RUnlock()
	if !ok {
		return C.nimcp_callback_action_t(CallbackActionContinue)
	}
	var m CallbackMetrics
	if metrics != nil {
		m = CallbackMetrics{
			Step:         uint64(metrics.step),
			Epoch:        uint64(metrics.epoch),
			Loss:         float32(metrics.loss),
			LossEMA:      float32(metrics.loss_ema),
			LearningRate: float32(metrics.learning_rate),
			GradientNorm: float32(metrics.gradient_norm),
			StepTimeUS:   uint64(metrics.step_time_us),
			IsConverging: bool(metrics.is_converging),
			IsDiverging:  bool(metrics.is_diverging),
		}
	}
	action := cb(CallbackEvent(event), &m)
	return C.nimcp_callback_action_t(action)
}

// ============================================================================
// Library Lifecycle
// ============================================================================

// Init initializes the NIMCP library. Call once at startup.
func Init() error {
	return checkStatus(C.nimcp_init())
}

// Shutdown cleans up the NIMCP library. Call once at exit.
func Shutdown() {
	C.nimcp_shutdown()
}

// Version returns the NIMCP version string.
func Version() string {
	return C.GoString(C.nimcp_version())
}

// VersionInt returns the NIMCP version as integer (MAJOR*10000 + MINOR*100 + PATCH).
func VersionInt() int {
	return int(C.nimcp_version_int())
}

// GetError returns the last error message from the C library.
func GetError() string {
	return C.GoString(C.nimcp_get_error())
}

// ============================================================================
// Brain
// ============================================================================

// Brain represents a NIMCP brain instance.
type Brain struct {
	handle       C.nimcp_brain_t
	callbackKeys []uint64 // track registered callback keys for cleanup
}

// NewBrain creates a new brain with preset configuration.
func NewBrain(name string, size BrainSize, task BrainTask, numInputs, numOutputs uint32) (*Brain, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	handle := C.nimcp_brain_create(cName, C.nimcp_brain_size_t(size), C.nimcp_brain_task_t(task),
		C.uint(numInputs), C.uint(numOutputs))
	if handle == nil {
		return nil, &NimcpError{Code: ErrGeneric, Message: GetError()}
	}
	return &Brain{handle: handle}, nil
}

// LoadBrain loads a brain from file.
func LoadBrain(filepath string) (*Brain, error) {
	cPath := C.CString(filepath)
	defer C.free(unsafe.Pointer(cPath))
	handle := C.nimcp_brain_load(cPath)
	if handle == nil {
		return nil, &NimcpError{Code: ErrIO, Message: GetError()}
	}
	return &Brain{handle: handle}, nil
}

// NewBrainFromConfig creates a brain from YAML/JSON config file.
func NewBrainFromConfig(configPath string) (*Brain, error) {
	cPath := C.CString(configPath)
	defer C.free(unsafe.Pointer(cPath))
	handle := C.nimcp_brain_create_from_config(cPath)
	if handle == nil {
		return nil, &NimcpError{Code: ErrGeneric, Message: GetError()}
	}
	return &Brain{handle: handle}, nil
}

// Close destroys the brain and frees resources.
func (b *Brain) Close() error {
	if b.handle != nil {
		C.nimcp_brain_destroy(b.handle)
		b.handle = nil
	}
	for _, key := range b.callbackKeys {
		unregisterGoCallback(key)
	}
	b.callbackKeys = nil
	return nil
}

// Learn teaches the brain from a single example.
func (b *Brain) Learn(features []float32, label string, confidence float32) error {
	cLabel := C.CString(label)
	defer C.free(unsafe.Pointer(cLabel))
	return checkStatus(C.nimcp_brain_learn_example(b.handle,
		(*C.float)(unsafe.Pointer(&features[0])), C.uint(len(features)),
		cLabel, C.float(confidence)))
}

// Predict makes a classification prediction.
func (b *Brain) Predict(features []float32) (label string, confidence float32, err error) {
	var cLabel [64]C.char
	var cConf C.float
	status := C.nimcp_brain_predict(b.handle,
		(*C.float)(unsafe.Pointer(&features[0])), C.uint(len(features)),
		&cLabel[0], &cConf)
	if err := checkStatus(status); err != nil {
		return "", 0, err
	}
	return C.GoString(&cLabel[0]), float32(cConf), nil
}

// Infer runs inference and returns raw output vector.
func (b *Brain) Infer(features []float32, numOutputs uint32) ([]float32, error) {
	outputs := make([]float32, numOutputs)
	status := C.nimcp_brain_infer(b.handle,
		(*C.float)(unsafe.Pointer(&features[0])), C.uint(len(features)),
		(*C.float)(unsafe.Pointer(&outputs[0])), C.uint(numOutputs))
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return outputs, nil
}

// Save saves the brain to file.
func (b *Brain) Save(filepath string) error {
	cPath := C.CString(filepath)
	defer C.free(unsafe.Pointer(cPath))
	return checkStatus(C.nimcp_brain_save(b.handle, cPath))
}

// --- Training Pipeline ---

// ConfigureTraining sets up the training pipeline.
func (b *Brain) ConfigureTraining(config *TrainingConfig) error {
	c := config.toC()
	return checkStatus(C.nimcp_brain_configure_training(b.handle, &c))
}

// TrainStep performs a single training step.
func (b *Brain) TrainStep(features, targets []float32) (*TrainingResult, error) {
	var cResult C.nimcp_training_result_t
	status := C.nimcp_brain_train_step(b.handle,
		(*C.float)(unsafe.Pointer(&features[0])), C.uint(len(features)),
		(*C.float)(unsafe.Pointer(&targets[0])), C.uint(len(targets)),
		&cResult)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return &TrainingResult{
		Loss:         float32(cResult.loss),
		LearningRate: float32(cResult.learning_rate),
		Step:         uint32(cResult.step),
		EarlyStopped: bool(cResult.early_stopped),
		GradientNorm: float32(cResult.gradient_norm),
	}, nil
}

// TrainBatch trains on a batch of examples.
func (b *Brain) TrainBatch(features, targets []float32, batchSize, numFeatures, numTargets uint32) (*TrainingResult, error) {
	var cResult C.nimcp_training_result_t
	status := C.nimcp_brain_train_batch(b.handle,
		(*C.float)(unsafe.Pointer(&features[0])),
		(*C.float)(unsafe.Pointer(&targets[0])),
		C.uint(batchSize), C.uint(numFeatures), C.uint(numTargets),
		&cResult)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return &TrainingResult{
		Loss:         float32(cResult.loss),
		LearningRate: float32(cResult.learning_rate),
		Step:         uint32(cResult.step),
		EarlyStopped: bool(cResult.early_stopped),
		GradientNorm: float32(cResult.gradient_norm),
	}, nil
}

// GetTrainingStats returns current training statistics.
func (b *Brain) GetTrainingStats() (totalSteps uint64, totalLoss float32, currentLR float32, err error) {
	var cSteps C.ulong
	var cLoss, cLR C.float
	status := C.nimcp_brain_get_training_stats(b.handle, &cSteps, &cLoss, &cLR)
	if err := checkStatus(status); err != nil {
		return 0, 0, 0, err
	}
	return uint64(cSteps), float32(cLoss), float32(cLR), nil
}

// StepScheduler updates the learning rate scheduler.
func (b *Brain) StepScheduler(validationMetric float32) float32 {
	return float32(C.nimcp_brain_step_scheduler(b.handle, C.float(validationMetric)))
}

// --- Callbacks ---

// EnableCallbacks enables the training callback system.
func (b *Brain) EnableCallbacks(config *CallbackConfig) error {
	c := config.toC()
	return checkStatus(C.nimcp_brain_enable_callbacks(b.handle, &c))
}

// DisableCallbacks disables the training callback system.
func (b *Brain) DisableCallbacks() error {
	return checkStatus(C.nimcp_brain_disable_callbacks(b.handle))
}

// RegisterCallback registers a Go callback for a training event.
// Returns the callback ID (>0) on success.
func (b *Brain) RegisterCallback(event CallbackEvent, fn CallbackFunc, name string) (uint32, error) {
	key := registerGoCallback(fn)

	var cName *C.char
	if name != "" {
		cName = C.CString(name)
		defer C.free(unsafe.Pointer(cName))
	}

	id := C.nimcp_brain_register_callback(b.handle,
		C.nimcp_callback_event_t(event),
		C.nimcp_training_callback_fn(C.goCallbackTrampoline),
		unsafe.Pointer(uintptr(key)),
		cName)
	if id == 0 {
		unregisterGoCallback(key)
		return 0, &NimcpError{Code: ErrGeneric, Message: "failed to register callback"}
	}
	b.callbackKeys = append(b.callbackKeys, key)
	return uint32(id), nil
}

// UnregisterCallback unregisters a callback by ID.
func (b *Brain) UnregisterCallback(callbackID uint32) error {
	return checkStatus(C.nimcp_brain_unregister_callback(b.handle, C.uint(callbackID)))
}

// GetCallbackStats returns callback statistics.
func (b *Brain) GetCallbackStats() (totalFired uint64, avgTimeUS float32, earlyStops uint32, err error) {
	var cFired C.ulong
	var cAvg C.float
	var cStops C.uint
	status := C.nimcp_brain_get_callback_stats(b.handle, &cFired, &cAvg, &cStops)
	if err := checkStatus(status); err != nil {
		return 0, 0, 0, err
	}
	return uint64(cFired), float32(cAvg), uint32(cStops), nil
}

// --- Resize ---

// Resize manually resizes brain to a specific neuron count.
func (b *Brain) Resize(newNeuronCount uint32) bool {
	return bool(C.nimcp_brain_resize(b.handle, C.uint(newNeuronCount)))
}

// AutoResize automatically resizes based on hardware and utilization.
func (b *Brain) AutoResize() bool {
	return bool(C.nimcp_brain_auto_resize(b.handle))
}

// GetNeuronCount returns current neuron count.
func (b *Brain) GetNeuronCount() uint32 {
	return uint32(C.nimcp_brain_get_neuron_count(b.handle))
}

// GetUtilizationMetrics returns brain utilization and saturation.
func (b *Brain) GetUtilizationMetrics() (utilization, saturation float32, ok bool) {
	var cUtil, cSat C.float
	result := C.nimcp_brain_get_utilization_metrics(b.handle, &cUtil, &cSat)
	return float32(cUtil), float32(cSat), bool(result)
}

// --- Named Snapshots ---

// SnapshotSave saves a named snapshot of the brain state.
func (b *Brain) SnapshotSave(name, description string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	var cDesc *C.char
	if description != "" {
		cDesc = C.CString(description)
		defer C.free(unsafe.Pointer(cDesc))
	}
	return checkStatus(C.nimcp_brain_snapshot_save(b.handle, cName, cDesc))
}

// SnapshotRestore restores brain from a named snapshot.
func (b *Brain) SnapshotRestore(name string) (*Brain, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	handle := C.nimcp_brain_snapshot_restore(b.handle, cName)
	if handle == nil {
		return nil, &NimcpError{Code: ErrGeneric, Message: GetError()}
	}
	return &Brain{handle: handle}, nil
}

// SnapshotList lists all available snapshots.
func (b *Brain) SnapshotList(maxCount uint32) ([]SnapshotInfo, error) {
	infos := make([]C.nimcp_brain_snapshot_info_t, maxCount)
	var outCount C.uint
	status := C.nimcp_brain_snapshot_list(b.handle, &infos[0], C.uint(maxCount), &outCount)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	result := make([]SnapshotInfo, int(outCount))
	for i := 0; i < int(outCount); i++ {
		result[i] = SnapshotInfo{
			Name:         C.GoString(&infos[i].name[0]),
			Description:  C.GoString(&infos[i].description[0]),
			Timestamp:    uint64(infos[i].timestamp),
			FileSize:     uint32(infos[i].file_size),
			IsCompressed: bool(infos[i].is_compressed),
			IsEncrypted:  bool(infos[i].is_encrypted),
		}
	}
	return result, nil
}

// SnapshotDelete deletes a named snapshot.
func (b *Brain) SnapshotDelete(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.nimcp_brain_snapshot_delete(b.handle, cName))
}

// --- Copy-on-Write ---

// BrainSnapshot represents a COW snapshot handle.
type BrainSnapshot struct {
	handle C.nimcp_brain_snapshot_t
}

// CloneCOW creates a lightweight COW clone of the brain.
func (b *Brain) CloneCOW() (*Brain, error) {
	handle := C.nimcp_brain_clone_cow(b.handle)
	if handle == nil {
		return nil, &NimcpError{Code: ErrGeneric, Message: GetError()}
	}
	return &Brain{handle: handle}, nil
}

// SnapshotCOW creates a zero-copy snapshot using COW.
func (b *Brain) SnapshotCOW() (*BrainSnapshot, error) {
	handle := C.nimcp_brain_snapshot_cow(b.handle)
	if handle == nil {
		return nil, &NimcpError{Code: ErrGeneric, Message: GetError()}
	}
	return &BrainSnapshot{handle: handle}, nil
}

// RestoreCOW restores brain state from a COW snapshot.
func (b *Brain) RestoreCOW(snapshot *BrainSnapshot) error {
	return checkStatus(C.nimcp_brain_restore_cow(b.handle, snapshot.handle))
}

// Close destroys the snapshot and releases references.
func (s *BrainSnapshot) Close() error {
	if s.handle != nil {
		C.nimcp_brain_snapshot_destroy(s.handle)
		s.handle = nil
	}
	return nil
}

// --- Working Memory ---

// WorkingMemoryAdd adds an item to working memory.
func (b *Brain) WorkingMemoryAdd(data []float32, salience float32) error {
	return checkStatus(C.nimcp_brain_working_memory_add(b.handle,
		(*C.float)(unsafe.Pointer(&data[0])), C.uint(len(data)),
		C.float(salience)))
}

// WorkingMemoryGet retrieves an item from working memory by index.
func (b *Brain) WorkingMemoryGet(index uint32) ([]float32, error) {
	var size C.uint
	ptr := C.nimcp_brain_working_memory_get(b.handle, C.uint(index), &size)
	if ptr == nil {
		return nil, &NimcpError{Code: ErrGeneric, Message: "working memory item not found"}
	}
	n := int(size)
	result := make([]float32, n)
	cSlice := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), n)
	copy(result, cSlice)
	return result, nil
}

// WorkingMemoryStats returns current size and capacity.
func (b *Brain) WorkingMemoryStats() (currentSize, capacity uint32, err error) {
	var cSize, cCap C.uint
	status := C.nimcp_brain_working_memory_stats(b.handle, &cSize, &cCap)
	if err := checkStatus(status); err != nil {
		return 0, 0, err
	}
	return uint32(cSize), uint32(cCap), nil
}

// WorkingMemoryRefresh refreshes an item to prevent decay.
func (b *Brain) WorkingMemoryRefresh(index uint32) error {
	return checkStatus(C.nimcp_brain_working_memory_refresh(b.handle, C.uint(index)))
}

// --- Global Workspace ---

// WorkspaceCompete submits content for global workspace competition.
func (b *Brain) WorkspaceCompete(module CognitiveModule, content []float32, strength float32) error {
	return checkStatus(C.nimcp_brain_workspace_compete(b.handle,
		C.nimcp_cognitive_module_t(module),
		(*C.float)(unsafe.Pointer(&content[0])), C.uint(len(content)),
		C.float(strength)))
}

// WorkspaceRead reads current global workspace broadcast.
func (b *Brain) WorkspaceRead(maxDim uint32) (content []float32, source CognitiveModule, err error) {
	buf := make([]float32, maxDim)
	var actualDim C.uint
	var cSource C.nimcp_cognitive_module_t
	status := C.nimcp_brain_workspace_read(b.handle,
		(*C.float)(unsafe.Pointer(&buf[0])), C.uint(maxDim),
		&actualDim, &cSource)
	if err := checkStatus(status); err != nil {
		return nil, ModuleNone, err
	}
	return buf[:int(actualDim)], CognitiveModule(cSource), nil
}

// WorkspaceSubscribe subscribes a module to workspace broadcasts.
func (b *Brain) WorkspaceSubscribe(module CognitiveModule) error {
	return checkStatus(C.nimcp_brain_workspace_subscribe(b.handle,
		C.nimcp_cognitive_module_t(module)))
}

// WorkspaceUnsubscribe unsubscribes a module from workspace broadcasts.
func (b *Brain) WorkspaceUnsubscribe(module CognitiveModule) error {
	return checkStatus(C.nimcp_brain_workspace_unsubscribe(b.handle,
		C.nimcp_cognitive_module_t(module)))
}

// WorkspaceHasBroadcast checks if workspace has an active broadcast.
func (b *Brain) WorkspaceHasBroadcast() (bool, error) {
	var has C.bool
	status := C.nimcp_brain_workspace_has_broadcast(b.handle, &has)
	if err := checkStatus(status); err != nil {
		return false, err
	}
	return bool(has), nil
}

// WorkspaceStats returns workspace statistics.
func (b *Brain) WorkspaceStats() (totalBroadcasts, totalCompetitions uint32, avgStrength float32, err error) {
	var cBcast, cComp C.uint
	var cAvg C.float
	status := C.nimcp_brain_workspace_stats(b.handle, &cBcast, &cComp, &cAvg)
	if err := checkStatus(status); err != nil {
		return 0, 0, 0, err
	}
	return uint32(cBcast), uint32(cComp), float32(cAvg), nil
}

// --- Oscillations ---

// EnableOscillations enables or disables complex oscillation features.
func (b *Brain) EnableOscillations(enable bool) bool {
	return bool(C.nimcp_enable_complex_oscillations(b.handle, C.bool(enable)))
}

// IsOscillationsEnabled checks if oscillations are enabled.
func (b *Brain) IsOscillationsEnabled() bool {
	return bool(C.nimcp_is_complex_oscillations_enabled(b.handle))
}

// GetPhasor returns the oscillation phasor for a neuron.
func (b *Brain) GetPhasor(neuronID uint32) Phasor {
	p := C.nimcp_get_oscillation_phasor(b.handle, C.uint(neuronID))
	return Phasor{Amplitude: float32(p.amplitude), Phase: float32(p.phase)}
}

// GetPhaseCoherence computes phase coherence across neurons.
func (b *Brain) GetPhaseCoherence(neuronIDs []uint32) float32 {
	return float32(C.nimcp_get_phase_coherence(b.handle,
		(*C.uint)(unsafe.Pointer(&neuronIDs[0])), C.uint(len(neuronIDs))))
}

// GetPACModulation computes phase-amplitude coupling modulation index.
func (b *Brain) GetPACModulation(thetaFreq, gammaFreq float32) float32 {
	return float32(C.nimcp_get_pac_modulation(b.handle, C.float(thetaFreq), C.float(gammaFreq)))
}

// --- Probe ---

// Probe returns comprehensive brain state statistics.
func (b *Brain) Probe() (*BrainProbe, error) {
	var cp C.nimcp_brain_probe_t
	status := C.nimcp_brain_probe(b.handle, &cp)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return &BrainProbe{
		TaskName:            C.GoString(&cp.task_name[0]),
		Size:                BrainSize(cp.size),
		Task:                BrainTask(cp.task),
		NumNeurons:          uint32(cp.num_neurons),
		NumSynapses:         uint32(cp.num_synapses),
		NumActiveSynapses:   uint32(cp.num_active_synapses),
		TotalInferences:     uint64(cp.total_inferences),
		TotalLearningSteps:  uint64(cp.total_learning_steps),
		AvgSparsity:         float32(cp.avg_sparsity),
		AvgInferenceTimeUS:  float32(cp.avg_inference_time_us),
		CurrentLearningRate: float32(cp.current_learning_rate),
		Accuracy:            float32(cp.accuracy),
		MemoryBytes:         uint64(cp.memory_bytes),
		NumInputs:           uint32(cp.num_inputs),
		NumOutputs:          uint32(cp.num_outputs),
		IsCOWClone:          bool(cp.is_cow_clone),
		COWRefCount:         uint32(cp.cow_ref_count),
		COWSharedBytes:      uint64(cp.cow_shared_bytes),
		COWPrivateBytes:     uint64(cp.cow_private_bytes),
	}, nil
}

// BroadcastProbe broadcasts brain probe data via bio-async.
func (b *Brain) BroadcastProbe() error {
	return checkStatus(C.nimcp_brain_broadcast_probe(b.handle))
}

// ============================================================================
// Network
// ============================================================================

// Network represents a low-level neural network.
type Network struct {
	handle     C.nimcp_network_t
	numOutputs int
}

// NewNetwork creates a new neural network.
func NewNetwork(numInputs, numOutputs, numHidden uint32, learningRate float32) (*Network, error) {
	handle := C.nimcp_network_create(C.uint(numInputs), C.uint(numOutputs),
		C.uint(numHidden), C.float(learningRate))
	if handle == nil {
		return nil, &NimcpError{Code: ErrGeneric, Message: GetError()}
	}
	return &Network{handle: handle, numOutputs: int(numOutputs)}, nil
}

// Close destroys the network.
func (n *Network) Close() error {
	if n.handle != nil {
		C.nimcp_network_destroy(n.handle)
		n.handle = nil
	}
	return nil
}

// Forward performs a forward pass.
func (n *Network) Forward(inputs []float32) ([]float32, error) {
	outputs := make([]float32, n.numOutputs)
	status := C.nimcp_network_forward(n.handle,
		(*C.float)(unsafe.Pointer(&inputs[0])), C.uint(len(inputs)),
		(*C.float)(unsafe.Pointer(&outputs[0])), C.uint(n.numOutputs))
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return outputs, nil
}

// Train trains on a single example.
func (n *Network) Train(inputs, targets []float32) error {
	return checkStatus(C.nimcp_network_train(n.handle,
		(*C.float)(unsafe.Pointer(&inputs[0])), C.uint(len(inputs)),
		(*C.float)(unsafe.Pointer(&targets[0])), C.uint(len(targets))))
}

// ============================================================================
// Ethics
// ============================================================================

// Ethics represents an ethics evaluation module.
type Ethics struct {
	handle C.nimcp_ethics_t
}

// NewEthics creates a new ethics module.
func NewEthics() (*Ethics, error) {
	handle := C.nimcp_ethics_create()
	if handle == nil {
		return nil, &NimcpError{Code: ErrGeneric, Message: GetError()}
	}
	return &Ethics{handle: handle}, nil
}

// Close destroys the ethics module.
func (e *Ethics) Close() error {
	if e.handle != nil {
		C.nimcp_ethics_destroy(e.handle)
		e.handle = nil
	}
	return nil
}

// Check evaluates the ethical score of a situation.
func (e *Ethics) Check(situation []float32) (float32, error) {
	var score C.float
	status := C.nimcp_ethics_check(e.handle,
		(*C.float)(unsafe.Pointer(&situation[0])), C.uint(len(situation)),
		&score)
	if err := checkStatus(status); err != nil {
		return 0, err
	}
	return float32(score), nil
}

// ============================================================================
// KnowledgeGraph
// ============================================================================

// KnowledgeGraph represents a knowledge graph.
type KnowledgeGraph struct {
	handle C.nimcp_knowledge_t
}

// NewKnowledgeGraph creates a new knowledge graph.
func NewKnowledgeGraph() (*KnowledgeGraph, error) {
	handle := C.nimcp_knowledge_create()
	if handle == nil {
		return nil, &NimcpError{Code: ErrGeneric, Message: GetError()}
	}
	return &KnowledgeGraph{handle: handle}, nil
}

// Close destroys the knowledge graph.
func (kg *KnowledgeGraph) Close() error {
	if kg.handle != nil {
		C.nimcp_knowledge_destroy(kg.handle)
		kg.handle = nil
	}
	return nil
}

// AddFact adds a subject-predicate-object triple.
func (kg *KnowledgeGraph) AddFact(subject, predicate, object string) error {
	cSubj := C.CString(subject)
	defer C.free(unsafe.Pointer(cSubj))
	cPred := C.CString(predicate)
	defer C.free(unsafe.Pointer(cPred))
	cObj := C.CString(object)
	defer C.free(unsafe.Pointer(cObj))
	return checkStatus(C.nimcp_knowledge_add_fact(kg.handle, cSubj, cPred, cObj))
}

// Query queries the knowledge graph.
func (kg *KnowledgeGraph) Query(query string) (string, error) {
	cQuery := C.CString(query)
	defer C.free(unsafe.Pointer(cQuery))
	var buf [1024]C.char
	status := C.nimcp_knowledge_query(kg.handle, cQuery, &buf[0], 1024)
	if err := checkStatus(status); err != nil {
		return "", err
	}
	return C.GoString(&buf[0]), nil
}

// ============================================================================
// Group 1 — Sensory/Multimodal
// ============================================================================

// SensoryOpts holds optional parameters for SubmitSensory.
type SensoryOpts struct {
	Width     uint32
	Height    uint32
	Channels  uint32
	NSegments uint32
}

// SubmitSensory stages sensory data for cross-modal cortex CNN processing.
// Supported modalities: "visual", "audio", "speech", "somatosensory".
// Optional opts (at most one) provides width/height/channels for visual,
// n_segments for somatosensory.
func (b *Brain) SubmitSensory(modality string, data []float32, opts ...SensoryOpts) error {
	if len(data) == 0 {
		return &NimcpError{Code: ErrInvalid, Message: "empty data slice"}
	}
	cMod := C.CString(modality)
	defer C.free(unsafe.Pointer(cMod))
	var o SensoryOpts
	if len(opts) > 0 {
		o = opts[0]
	}
	status := C.go_brain_submit_sensory(b.handle, cMod,
		(*C.float)(unsafe.Pointer(&data[0])), C.uint(len(data)),
		C.uint(o.Width), C.uint(o.Height),
		C.uint(o.Channels), C.uint(o.NSegments))
	return checkStatus(status)
}

// VisualCortexProcess runs an image through the brain's visual cortex CNN
// and returns the extracted feature vector.
func (b *Brain) VisualCortexProcess(pixels []float32, width, height, channels uint32) ([]float32, error) {
	if len(pixels) == 0 {
		return nil, &NimcpError{Code: ErrInvalid, Message: "empty pixel data"}
	}
	const maxFeat = 1024
	features := make([]float32, maxFeat)
	n := C.go_brain_visual_cortex_process(b.handle,
		(*C.float)(unsafe.Pointer(&pixels[0])), C.uint(len(pixels)),
		C.uint(width), C.uint(height), C.uint(channels),
		(*C.float)(unsafe.Pointer(&features[0])), C.uint(maxFeat))
	if n == 0 {
		return nil, &NimcpError{Code: ErrGeneric, Message: "visual cortex processing failed or unavailable"}
	}
	return features[:int(n)], nil
}

// ============================================================================
// Group 2 — Avatar/Metrics
// ============================================================================

// GetAvatarState returns the full avatar face/emotion/voice state.
func (b *Brain) GetAvatarState() (map[string]interface{}, error) {
	var state C.nimcp_avatar_state_t
	status := C.nimcp_brain_get_avatar_state(b.handle, &state)
	if err := checkStatus(status); err != nil {
		return nil, err
	}
	return map[string]interface{}{
		// Viseme / mouth
		"mouth_open":       float32(state.mouth_open),
		"lip_round":        float32(state.lip_round),
		"lip_upper":        float32(state.lip_upper),
		"lip_lower":        float32(state.lip_lower),
		"tongue_position":  float32(state.tongue_position),
		"current_viseme":   uint8(state.current_viseme),
		// FACS AUs
		"au1_inner_brow_raise": float32(state.au1_inner_brow_raise),
		"au2_outer_brow_raise": float32(state.au2_outer_brow_raise),
		"au4_brow_lower":       float32(state.au4_brow_lower),
		"au5_upper_lid_raise":  float32(state.au5_upper_lid_raise),
		"au6_cheek_raise":      float32(state.au6_cheek_raise),
		"au7_lid_tighten":      float32(state.au7_lid_tighten),
		"au9_nose_wrinkle":     float32(state.au9_nose_wrinkle),
		"au10_upper_lip_raise": float32(state.au10_upper_lip_raise),
		"au12_lip_corner_pull": float32(state.au12_lip_corner_pull),
		"au15_lip_corner_drop": float32(state.au15_lip_corner_drop),
		"au17_chin_raise":      float32(state.au17_chin_raise),
		"au20_lip_stretch":     float32(state.au20_lip_stretch),
		"au23_lip_tighten":     float32(state.au23_lip_tighten),
		"au25_lips_part":       float32(state.au25_lips_part),
		"au26_jaw_drop":        float32(state.au26_jaw_drop),
		"au28_lip_suck":        float32(state.au28_lip_suck),
		// Emotion
		"valence":           float32(state.valence),
		"arousal":           float32(state.arousal),
		"dominance":         float32(state.dominance),
		"emotion_id":        uint32(state.emotion_id),
		"emotion_intensity": float32(state.emotion_intensity),
		// Gaze + head
		"gaze_x":     float32(state.gaze_x),
		"gaze_y":     float32(state.gaze_y),
		"head_pitch":  float32(state.head_pitch),
		"head_yaw":    float32(state.head_yaw),
		"head_roll":   float32(state.head_roll),
		"blink":       float32(state.blink),
		// Voice
		"pitch_hz":      float32(state.pitch_hz),
		"speaking_rate": float32(state.speaking_rate),
		"volume":        float32(state.volume),
		// Metadata
		"timestamp_us": uint64(state.timestamp_us),
		"is_speaking":  bool(state.is_speaking),
	}, nil
}

// GetNetworkMetrics returns per-network training loss EMA and step counts.
func (b *Brain) GetNetworkMetrics() (map[string]interface{}, error) {
	var emaANN, emaCNN, emaSNN, emaLNN C.float
	var annSteps, cnnSteps, snnSteps, lnnSteps C.ulong
	ok := C.nimcp_brain_get_network_metrics(b.handle,
		&emaANN, &emaCNN, &emaSNN, &emaLNN,
		&annSteps, &cnnSteps, &snnSteps, &lnnSteps)
	if !bool(ok) {
		return nil, &NimcpError{Code: ErrGeneric, Message: "get_network_metrics failed"}
	}
	return map[string]interface{}{
		"ann_loss":  float32(emaANN),
		"cnn_loss":  float32(emaCNN),
		"snn_loss":  float32(emaSNN),
		"lnn_loss":  float32(emaLNN),
		"ann_steps": uint64(annSteps),
		"cnn_steps": uint64(cnnSteps),
		"snn_steps": uint64(snnSteps),
		"lnn_steps": uint64(lnnSteps),
	}, nil
}

// GetCortexCNNMetrics returns per-cortex CNN processor metrics.
// Keys: "visual", "audio", "speech", "somato".
func (b *Brain) GetCortexCNNMetrics() (map[string]interface{}, error) {
	typeKeys := [4]string{"visual", "audio", "speech", "somato"}
	result := make(map[string]interface{})
	for ci := 0; ci < 4; ci++ {
		var m C.go_cortex_cnn_metrics_t
		if bool(C.go_brain_get_cortex_cnn_metrics(b.handle, C.int(ci), &m)) {
			result[typeKeys[ci]] = map[string]interface{}{
				"last_loss":      float32(m.last_loss),
				"ema_loss":       float32(m.ema_loss),
				"forward_steps":  uint64(m.forward_steps),
				"backward_steps": uint64(m.backward_steps),
				"embedding_norm": float32(m.embedding_norm),
				"confidence":     float32(m.confidence),
				"embedding_dim":  uint32(m.embedding_dim),
				"num_params":     uint32(m.num_params),
			}
		}
	}
	return result, nil
}

// ============================================================================
// Group 3 — Core Inference
// ============================================================================

// DecideFull runs the full cognitive decision pipeline and returns a rich result.
func (b *Brain) DecideFull(features []float32) (map[string]interface{}, error) {
	if len(features) == 0 {
		return nil, &NimcpError{Code: ErrInvalid, Message: "empty features"}
	}
	var label [64]C.char
	var confidence C.float
	var explanation [256]C.char
	const maxOutput = 4096
	var outputVec [maxOutput]C.float
	outputSize := C.uint(maxOutput)
	var numActive C.uint
	var sparsity C.float
	var inferenceTimeUS C.ulong

	status := C.nimcp_brain_decide_full(b.handle,
		(*C.float)(unsafe.Pointer(&features[0])), C.uint(len(features)),
		&label[0], &confidence, &explanation[0],
		&outputVec[0], &outputSize,
		&numActive, &sparsity, &inferenceTimeUS)
	if err := checkStatus(status); err != nil {
		return nil, err
	}

	vecLen := int(outputSize)
	if vecLen > maxOutput {
		vecLen = maxOutput
	}
	vec := make([]float32, vecLen)
	for i := 0; i < vecLen; i++ {
		vec[i] = float32(outputVec[i])
	}

	return map[string]interface{}{
		"label":              C.GoString(&label[0]),
		"confidence":         float32(confidence),
		"explanation":        C.GoString(&explanation[0]),
		"output_vector":      vec,
		"num_active_neurons": uint32(numActive),
		"sparsity":           float32(sparsity),
		"inference_time_us":  uint64(inferenceTimeUS),
	}, nil
}

// GetTranscript returns the cognitive transcript from the last DecideFull call.
func (b *Brain) GetTranscript() ([]interface{}, error) {
	const maxEntries = 32
	var summaries [maxEntries][256]C.char
	var saliences [maxEntries]C.float
	var confidences [maxEntries]C.float
	var modules [maxEntries]*C.char

	count := C.nimcp_brain_get_last_transcript(b.handle,
		&summaries[0], &saliences[0], &confidences[0], &modules[0],
		C.uint(maxEntries))

	n := int(count)
	result := make([]interface{}, n)
	for i := 0; i < n; i++ {
		modName := "unknown"
		if modules[i] != nil {
			modName = C.GoString(modules[i])
		}
		result[i] = map[string]interface{}{
			"module":     modName,
			"summary":    C.GoString(&summaries[i][0]),
			"salience":   float32(saliences[i]),
			"confidence": float32(confidences[i]),
		}
	}
	return result, nil
}

// GetCognitiveStats returns per-module cognitive training statistics.
func (b *Brain) GetCognitiveStats() (map[string]interface{}, error) {
	var steps [13]C.uint
	var losses [13]C.float
	var count C.uint

	status := C.nimcp_brain_get_cognitive_stats(b.handle, &steps[0], &losses[0], &count)
	if err := checkStatus(status); err != nil {
		return nil, err
	}

	moduleNames := []string{
		"grounded_language", "knowledge", "vae", "fep_parietal",
		"physics_nn", "pred_hierarchy", "jepa", "creative",
		"self_heal", "intuition", "fep_orchestrator",
	}

	result := make(map[string]interface{})
	n := int(count)
	if n > 11 {
		n = 11
	}
	for i := 0; i < n; i++ {
		result[moduleNames[i]] = map[string]interface{}{
			"steps":     uint32(steps[i]),
			"last_loss": float32(losses[i]),
		}
	}
	return result, nil
}

// GetAccuracy returns the running label-match accuracy (EMA).
func (b *Brain) GetAccuracy() float32 {
	return float32(C.nimcp_brain_get_accuracy(b.handle))
}

// ============================================================================
// Group 4 — LNN/SNN/CNN
// ============================================================================

// LNNCreate creates an NCP-architecture LNN temporal processor on the brain.
// Default values: nSensory=128, nInter=64, nCommand=32, nOutput=64.
func (b *Brain) LNNCreate(nSensory, nInter, nCommand, nOutput uint32) error {
	ok := C.go_brain_lnn_create(b.handle,
		C.uint(nSensory), C.uint(nInter),
		C.uint(nCommand), C.uint(nOutput))
	if !bool(ok) {
		return &NimcpError{Code: ErrGeneric, Message: "failed to create LNN network"}
	}
	return nil
}

// LNNGetStats returns LNN network statistics.
func (b *Brain) LNNGetStats() (map[string]interface{}, error) {
	var stats C.go_lnn_stats_t
	if !bool(C.go_brain_lnn_get_stats(b.handle, &stats)) {
		return nil, &NimcpError{Code: ErrGeneric, Message: "LNN stats unavailable (network not created?)"}
	}
	return map[string]interface{}{
		"forward_steps":  uint64(stats.forward_steps),
		"backward_steps": uint64(stats.backward_steps),
		"total_ode_evals": uint64(stats.ode_evaluations),
		"avg_tau":         float32(stats.avg_tau),
		"state_norm":      float32(stats.state_norm),
		"gradient_norm":   float32(stats.gradient_norm),
		"nan_count":       uint32(stats.nan_count),
		"inf_count":       uint32(stats.inf_count),
	}, nil
}

// SNNGetStats returns SNN network statistics.
func (b *Brain) SNNGetStats() (map[string]interface{}, error) {
	var stats C.go_snn_stats_t
	if !bool(C.go_brain_snn_get_stats(b.handle, &stats)) {
		return nil, &NimcpError{Code: ErrGeneric, Message: "SNN stats unavailable (network not created?)"}
	}
	return map[string]interface{}{
		"total_steps":        uint64(stats.total_steps),
		"total_spikes":       uint64(stats.total_spikes),
		"mean_firing_rate":   float32(stats.mean_firing_rate),
		"max_firing_rate":    float32(stats.max_firing_rate),
		"sparsity":           float32(stats.sparsity),
		"synchrony":          float32(stats.synchrony),
		"spikes_per_sample":  float32(stats.spikes_per_sample),
		"silent_neurons":     uint32(stats.silent_neurons),
		"hyperactive_neurons": uint32(stats.hyperactive_neurons),
		"health":             int(stats.health),
		"memory_usage_bytes": uint64(stats.memory_usage_bytes),
	}, nil
}

// CNNGetStats returns CNN trainer statistics.
func (b *Brain) CNNGetStats() (map[string]interface{}, error) {
	var stats C.go_cnn_stats_t
	if !bool(C.go_brain_cnn_get_stats(b.handle, &stats)) {
		return nil, &NimcpError{Code: ErrGeneric, Message: "CNN stats unavailable (trainer not created?)"}
	}
	return map[string]interface{}{
		"num_layers":     uint32(stats.num_layers),
		"num_parameters": uint64(stats.num_parameters),
		"num_labels":     uint32(stats.num_labels),
		"active":         bool(stats.active),
	}, nil
}

// ============================================================================
// Group 5 — Configuration
// ============================================================================

// SetFastTraining enables or disables fast training mode.
func (b *Brain) SetFastTraining(enabled bool) error {
	if !bool(C.go_brain_set_fast_training(b.handle, C.bool(enabled))) {
		return &NimcpError{Code: ErrGeneric, Message: "set_fast_training failed (brain not initialized?)"}
	}
	return nil
}

// SetTaskType sets the brain's task strategy.
// Valid values: "regression", "classification", "pattern", "association".
func (b *Brain) SetTaskType(taskType string) error {
	cTask := C.CString(taskType)
	defer C.free(unsafe.Pointer(cTask))
	if !bool(C.go_brain_set_task_type(b.handle, cTask)) {
		return &NimcpError{Code: ErrInvalid, Message: fmt.Sprintf("unknown task type: %q (use regression/classification/pattern/association)", taskType)}
	}
	return nil
}

// EnableBiologicalPlasticity wires or unwires the biological plasticity bridge
// (TPB + EDP + coordinator) into the learn path.
func (b *Brain) EnableBiologicalPlasticity(enabled bool) error {
	if !bool(C.go_brain_enable_biological_plasticity(b.handle, C.bool(enabled))) {
		return &NimcpError{Code: ErrGeneric, Message: "enable_biological_plasticity failed (brain not initialized?)"}
	}
	return nil
}

// EnableMultiNetwork enables ensemble training across all network architectures
// (ANN, SNN, LNN, CNN).
func (b *Brain) EnableMultiNetwork() error {
	rc := C.go_brain_enable_multi_network(b.handle)
	if rc < 0 {
		return &NimcpError{Code: ErrGeneric, Message: "failed to enable multi-network training"}
	}
	return nil
}

// ============================================================================
// Group 6 — Brain State
// ============================================================================

// MedullaGetArousal returns the medulla arousal level [0.0, 1.0].
func (b *Brain) MedullaGetArousal() float32 {
	return float32(C.go_brain_medulla_get_arousal(b.handle))
}

// SleepGetPressure returns the current sleep pressure (adenosine) [0.0, 1.0].
func (b *Brain) SleepGetPressure() float32 {
	return float32(C.go_brain_sleep_get_pressure(b.handle))
}

// BGGetDopamine returns the basal ganglia dopamine level [0.0, 1.0].
func (b *Brain) BGGetDopamine() float32 {
	return float32(C.go_brain_bg_get_dopamine(b.handle))
}

// SubstrateGetHealth returns the computational substrate health status.
// Returns one of: "OPTIMAL", "STRESSED", "COMPROMISED", "CRITICAL", "UNKNOWN".
func (b *Brain) SubstrateGetHealth() string {
	return C.GoString(C.go_brain_substrate_get_health(b.handle))
}

// FocusAttention hints the attention system to prioritize a modality.
// Modality: "visual", "audio", "speech", "somatosensory".
func (b *Brain) FocusAttention(modality string) error {
	cMod := C.CString(modality)
	defer C.free(unsafe.Pointer(cMod))
	if !bool(C.go_brain_focus_attention(b.handle, cMod)) {
		return &NimcpError{Code: ErrGeneric, Message: "focus_attention failed"}
	}
	return nil
}
