#!/usr/bin/env perl
#
# NIMCP Perl bindings using FFI::Platypus
# Wraps the unified nimcp.h C API (v2.6.4)
#

package NIMCP;

use 5.010;
use strict;
use warnings;

our $VERSION = '2.6.4';

use FFI::Platypus 2.00;
use FFI::Platypus::Memory qw( malloc calloc free );
use FFI::Platypus::Buffer qw( scalar_to_buffer buffer_to_scalar );
use FFI::Platypus::Record;

# Record type for nimcp_oscillation_phasor_t (returned by value)
package NIMCP::PhasorRecord {
    use FFI::Platypus::Record;
    record_layout_1(
        float => 'amplitude',
        float => 'phase',
    );
}
package NIMCP;

# ============================================================================
# FFI Setup
# ============================================================================

my $ffi = FFI::Platypus->new(api => 2, lang => 'C');

# Find libnimcp.so: check LD_LIBRARY_PATH, then common locations
my @search_paths;
if ($ENV{LD_LIBRARY_PATH}) {
    push @search_paths, map { "$_/libnimcp.so" } split(/:/, $ENV{LD_LIBRARY_PATH});
}
push @search_paths, qw(
    libnimcp.so
    ./libnimcp.so
    ../lib/libnimcp.so
);

my $found_lib;
for my $path (@search_paths) {
    if (-f $path) {
        $found_lib = $path;
        last;
    }
}
$ffi->lib($found_lib // 'libnimcp.so');

# ============================================================================
# Constants
# ============================================================================

use constant {
    # Brain sizes
    BRAIN_TINY   => 0,
    BRAIN_SMALL  => 1,
    BRAIN_MEDIUM => 2,
    BRAIN_LARGE  => 3,

    # Task types
    TASK_CLASSIFICATION   => 0,
    TASK_REGRESSION       => 1,
    TASK_PATTERN_MATCHING => 2,
    TASK_SEQUENCE         => 3,
    TASK_ASSOCIATION      => 4,

    # Network types
    NETWORK_ADAPTIVE => 0,
    NETWORK_SNN      => 1,
    NETWORK_LNN      => 2,
    NETWORK_CNN      => 3,
    NETWORK_HYBRID   => 4,

    # SNN training methods
    SNN_TRAIN_STDP        => 0,
    SNN_TRAIN_R_STDP      => 1,
    SNN_TRAIN_EPROP       => 2,
    SNN_TRAIN_SURROGATE   => 3,
    SNN_TRAIN_HOMEOSTATIC => 4,

    # LNN training methods
    LNN_TRAIN_ADJOINT => 0,
    LNN_TRAIN_BPTT    => 1,
    LNN_TRAIN_RTRL    => 2,
    LNN_TRAIN_EPROP   => 3,

    # Status codes
    OK            => 0,
    ERROR         => 1000,
    ERROR_NULL    => 1003,
    ERROR_INVALID => 1004,
    ERROR_MEMORY  => 2000,
    ERROR_IO      => 4000,

    # Loss types
    LOSS_MSE           => 0,
    LOSS_CROSS_ENTROPY => 1,
    LOSS_BINARY_CE     => 2,
    LOSS_HUBER         => 3,
    LOSS_MAE           => 4,
    LOSS_FOCAL         => 5,
    LOSS_KL_DIV        => 6,

    # Optimizer types
    OPT_SGD      => 0,
    OPT_MOMENTUM => 1,
    OPT_ADAM     => 2,
    OPT_ADAMW    => 3,
    OPT_RMSPROP  => 4,
    OPT_ADAGRAD  => 5,

    # Scheduler types
    SCHED_CONSTANT          => 0,
    SCHED_STEP              => 1,
    SCHED_EXPONENTIAL       => 2,
    SCHED_COSINE            => 3,
    SCHED_WARMUP_COSINE     => 4,
    SCHED_REDUCE_ON_PLATEAU => 5,
    SCHED_CYCLIC            => 6,

    # Callback events
    CB_STEP_COMPLETE   => 0,
    CB_EPOCH_COMPLETE  => 1,
    CB_LOSS_COMPUTED   => 2,
    CB_WEIGHTS_UPDATED => 3,
    CB_LR_CHANGED      => 4,
    CB_CONVERGENCE     => 5,
    CB_DIVERGENCE      => 6,
    CB_CHECKPOINT      => 7,
    CB_EVENT_COUNT     => 8,

    # Callback actions
    CB_ACTION_CONTINUE    => 0,
    CB_ACTION_STOP        => 1,
    CB_ACTION_SKIP        => 2,
    CB_ACTION_ROLLBACK    => 3,
    CB_ACTION_REDUCE_LR   => 4,
    CB_ACTION_INCREASE_LR => 5,

    # Cognitive modules
    MODULE_NONE            => 0,
    MODULE_PERCEPTION      => 1,
    MODULE_WORKING_MEMORY  => 2,
    MODULE_EXECUTIVE       => 3,
    MODULE_THEORY_OF_MIND  => 4,
    MODULE_ETHICS          => 5,
    MODULE_ATTENTION       => 6,
    MODULE_EMOTION         => 7,
    MODULE_SALIENCE        => 8,
    MODULE_MOTOR           => 9,
    MODULE_LANGUAGE        => 10,
    MODULE_METACOGNITION   => 11,
    MODULE_CURIOSITY       => 12,
    MODULE_INTROSPECTION   => 13,
    MODULE_PREDICTIVE      => 14,
    MODULE_CONSOLIDATION   => 15,
    MODULE_EPISODIC_MEMORY => 16,
    MODULE_SEMANTIC_MEMORY => 17,
    MODULE_WELLBEING       => 18,
    MODULE_MENTAL_HEALTH   => 19,
    MODULE_GOAL_MOTIVATION => 20,
    MODULE_COGNITIVE_CONTROL => 21,
    MODULE_CUSTOM_START    => 100,
};

# ============================================================================
# FFI Function Declarations
# ============================================================================
# Types: float[] = input float array (pass arrayref)
#        float*  = output pointer to single float (pass \$scalar)
#        opaque  = raw C pointer (handle types, buffers)
#        string  = input const char* (pass Perl string)

# Library lifecycle
$ffi->attach('nimcp_init'        => []           => 'int');
$ffi->attach('nimcp_shutdown'    => []           => 'void');
$ffi->attach('nimcp_version'     => []           => 'string');
$ffi->attach('nimcp_version_int' => []           => 'int');
$ffi->attach('nimcp_get_error'   => []           => 'string');

# Brain core
$ffi->attach('nimcp_brain_create'  => ['string', 'int', 'int', 'uint32', 'uint32'] => 'opaque');
$ffi->attach('nimcp_brain_destroy' => ['opaque'] => 'void');
$ffi->attach('nimcp_brain_learn_example' =>
    ['opaque', 'float[]', 'uint32', 'string', 'float'] => 'int');
$ffi->attach('nimcp_brain_predict' =>
    ['opaque', 'float[]', 'uint32', 'opaque', 'float*'] => 'int');
$ffi->attach('nimcp_brain_infer' =>
    ['opaque', 'float[]', 'uint32', 'float[]', 'uint32'] => 'int');
$ffi->attach('nimcp_brain_save' => ['opaque', 'string'] => 'int');
$ffi->attach('nimcp_brain_load' => ['string'] => 'opaque');
$ffi->attach('nimcp_brain_create_from_config' => ['string'] => 'opaque');

# Brain creation variants
$ffi->attach('nimcp_brain_create_full' =>
    ['string', 'int', 'uint32', 'uint32', 'uint32'] => 'opaque');
$ffi->attach('nimcp_brain_create_with_neurons' =>
    ['string', 'int', 'uint32', 'uint32', 'uint32'] => 'opaque');

# Brain training
$ffi->attach('nimcp_brain_configure_training' => ['opaque', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_train_step' =>
    ['opaque', 'float[]', 'uint32', 'float[]', 'uint32', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_train_batch' =>
    ['opaque', 'float[]', 'float[]', 'uint32', 'uint32', 'uint32', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_get_training_stats' =>
    ['opaque', 'uint64*', 'float*', 'float*'] => 'int');
$ffi->attach('nimcp_brain_step_scheduler' => ['opaque', 'float'] => 'float');

# Brain learning — vector / batch / knowledge / language
$ffi->attach('nimcp_brain_learn_vector' =>
    ['opaque', 'float[]', 'uint32', 'float[]', 'uint32', 'string', 'float'] => 'int');
$ffi->attach('nimcp_brain_learn_vector_batch' =>
    ['opaque', 'opaque', 'opaque', 'uint32', 'uint32', 'uint32', 'float'] => 'float');
$ffi->attach('nimcp_brain_learn_batch' =>
    ['opaque', 'opaque', 'opaque', 'opaque', 'opaque', 'uint32', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_learn_knowledge' =>
    ['opaque', 'string', 'int'] => 'int');
$ffi->attach('nimcp_brain_learn_language' =>
    ['opaque', 'string', 'float*'] => 'int');
$ffi->attach('nimcp_brain_learn_language_pair' =>
    ['opaque', 'string', 'string', 'float', 'float*'] => 'int');

# Brain cognitive training
$ffi->attach('nimcp_brain_train_cognitive' =>
    ['opaque', 'string', 'int', 'string', 'float', 'float*'] => 'int');
$ffi->attach('nimcp_brain_train_language' =>
    ['opaque', 'string', 'string', 'float', 'float*'] => 'int');

# Brain experience
$ffi->attach('nimcp_brain_experience' =>
    ['opaque', 'float[]', 'uint32', 'float[]', 'uint32', 'float', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_experience_configure' =>
    ['opaque', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_experience_correct' =>
    ['opaque', 'float[]', 'uint32'] => 'float');
$ffi->attach('nimcp_brain_experience_attend' =>
    ['opaque', 'string', 'float'] => 'int');

# Brain metrics / stats
$ffi->attach('nimcp_brain_get_last_loss'          => ['opaque'] => 'float');
$ffi->attach('nimcp_brain_get_last_gradient_norm'  => ['opaque'] => 'float');
$ffi->attach('nimcp_brain_get_accuracy'            => ['opaque'] => 'float');
$ffi->attach('nimcp_brain_get_avatar_state'        => ['opaque', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_get_cognitive_stats'     =>
    ['opaque', 'opaque', 'opaque', 'uint32*'] => 'int');
$ffi->attach('nimcp_brain_get_last_transcript'     =>
    ['opaque', 'opaque', 'opaque', 'opaque', 'opaque', 'uint32'] => 'uint32');
$ffi->attach('nimcp_brain_get_network_metrics'     =>
    ['opaque', 'float*', 'float*', 'float*', 'float*',
     'uint64*', 'uint64*', 'uint64*', 'uint64*'] => 'int');
$ffi->attach('nimcp_brain_get_cortex_cnn_metrics'  =>
    ['opaque', 'opaque', 'opaque', 'opaque', 'opaque', 'opaque', 'uint32*'] => 'int');

# Brain prediction variants
$ffi->attach('nimcp_brain_predict_fast' =>
    ['opaque', 'float[]', 'uint32', 'opaque', 'float*'] => 'int');
$ffi->attach('nimcp_brain_predict_in_domain' =>
    ['opaque', 'float[]', 'uint32', 'string', 'opaque', 'float*'] => 'int');

# Brain decide / speak / generate / comprehend / produce / grounded / creative
$ffi->attach('nimcp_brain_decide_full' =>
    ['opaque', 'float[]', 'uint32',
     'opaque', 'float*', 'opaque',
     'float[]', 'uint32*', 'uint32*', 'float*', 'uint64*'] => 'int');
$ffi->attach('nimcp_brain_speak' =>
    ['opaque', 'float[]', 'uint32', 'opaque', 'uint32', 'float*', 'float*'] => 'int');
$ffi->attach('nimcp_brain_generate_text' =>
    ['opaque', 'string', 'float[]', 'uint32', 'opaque', 'uint32', 'float*', 'float*'] => 'int');
$ffi->attach('nimcp_brain_comprehend' =>
    ['opaque', 'string', 'float[]', 'uint32', 'float*'] => 'int');
$ffi->attach('nimcp_brain_produce_text' =>
    ['opaque', 'float[]', 'uint32', 'opaque', 'uint32', 'float*'] => 'int');
$ffi->attach('nimcp_brain_grounded_respond' =>
    ['opaque', 'string', 'opaque', 'uint32', 'float*'] => 'int');
$ffi->attach('nimcp_brain_creative_blend' =>
    ['opaque', 'float[]', 'float[]', 'uint32', 'float', 'opaque', 'uint32'] => 'int');
$ffi->attach('nimcp_brain_ground_word' =>
    ['opaque', 'string', 'float[]', 'uint32', 'uint32', 'float'] => 'int');

# Brain freeze / frozen
$ffi->attach('nimcp_brain_freeze'    => ['opaque'] => 'int');
$ffi->attach('nimcp_brain_is_frozen' => ['opaque'] => 'int');

# Brain enable flags
$ffi->attach('nimcp_brain_enable_mixed_precision' =>
    ['opaque', 'int'] => 'int');
$ffi->attach('nimcp_brain_enable_gradient_checkpointing' =>
    ['opaque', 'int', 'uint32'] => 'int');
$ffi->attach('nimcp_brain_enable_hemispheric' =>
    ['opaque', 'int'] => 'int');
$ffi->attach('nimcp_brain_enable_recurrent' =>
    ['opaque', 'int', 'uint32', 'float', 'float'] => 'int');
$ffi->attach('nimcp_brain_enable_bptt' =>
    ['opaque', 'int', 'uint32', 'float'] => 'int');
$ffi->attach('nimcp_brain_enable_biological_plasticity' =>
    ['opaque', 'int'] => 'int');
$ffi->attach('nimcp_brain_enable_multi_network' =>
    ['opaque'] => 'int');

# Brain set / config
$ffi->attach('nimcp_brain_set_fast_training'  => ['opaque', 'int'] => 'int');
$ffi->attach('nimcp_brain_set_task_type'      => ['opaque', 'string'] => 'int');
$ffi->attach('nimcp_brain_set_training_mode'  => ['opaque', 'int'] => 'void');
$ffi->attach('nimcp_brain_set_network_ablation' =>
    ['opaque', 'int', 'int', 'int'] => 'void');

# Brain sensory / cortex
$ffi->attach('nimcp_brain_submit_sensory' =>
    ['opaque', 'string', 'float[]', 'uint32', 'uint32', 'uint32', 'uint32', 'uint32'] => 'int');
$ffi->attach('nimcp_brain_visual_cortex_process' =>
    ['opaque', 'float[]', 'uint32', 'uint32', 'uint32', 'uint32',
     'float[]', 'uint32', 'uint32*'] => 'int');

# Brain neuromodulation / sleep / substrate
$ffi->attach('nimcp_brain_medulla_get_arousal'    => ['opaque'] => 'float');
$ffi->attach('nimcp_brain_bg_get_dopamine'        => ['opaque'] => 'float');
$ffi->attach('nimcp_brain_sleep_get_pressure'     => ['opaque'] => 'float');
$ffi->attach('nimcp_brain_substrate_get_health'   =>
    ['opaque', 'opaque', 'uint32'] => 'int');

# Brain sub-network creation / stats
$ffi->attach('nimcp_brain_lnn_create' =>
    ['opaque', 'uint32', 'uint32', 'uint32', 'uint32'] => 'int');
$ffi->attach('nimcp_brain_lnn_get_stats' =>
    ['opaque', 'uint64*', 'uint64*', 'uint64*', 'float*',
     'float*', 'float*', 'uint32*', 'uint32*'] => 'int');
$ffi->attach('nimcp_brain_snn_get_stats' =>
    ['opaque', 'uint64*', 'uint64*', 'float*', 'float*',
     'float*', 'uint32*', 'uint32*', 'opaque', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_cnn_get_stats' =>
    ['opaque', 'uint32*', 'opaque', 'uint32*', 'opaque'] => 'int');

# Brain rubric
$ffi->attach('nimcp_brain_rubric' => ['opaque', 'opaque'] => 'int');

# Brain cloud
$ffi->attach('nimcp_brain_connect_cloud' =>
    ['opaque', 'opaque', 'float', 'int'] => 'int');
$ffi->attach('nimcp_brain_disconnect_cloud' => ['opaque'] => 'int');

# Brain callbacks
$ffi->attach('nimcp_brain_enable_callbacks'    => ['opaque', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_disable_callbacks'   => ['opaque'] => 'int');
$ffi->attach('nimcp_brain_register_callback'   =>
    ['opaque', 'int', 'opaque', 'opaque', 'string'] => 'uint32');
$ffi->attach('nimcp_brain_unregister_callback' => ['opaque', 'uint32'] => 'int');
$ffi->attach('nimcp_brain_get_callback_stats'  =>
    ['opaque', 'uint64*', 'float*', 'uint32*'] => 'int');

# Brain resize
$ffi->attach('nimcp_brain_resize'    => ['opaque', 'uint32'] => 'int');
$ffi->attach('nimcp_brain_auto_resize' => ['opaque'] => 'int');
$ffi->attach('nimcp_brain_get_neuron_count' => ['opaque'] => 'uint32');
$ffi->attach('nimcp_brain_get_utilization_metrics' =>
    ['opaque', 'float*', 'float*'] => 'int');

# Brain named snapshots
$ffi->attach('nimcp_brain_snapshot_save'    => ['opaque', 'string', 'string'] => 'int');
$ffi->attach('nimcp_brain_snapshot_restore' => ['opaque', 'string'] => 'opaque');
$ffi->attach('nimcp_brain_snapshot_list'    =>
    ['opaque', 'opaque', 'uint32', 'uint32*'] => 'int');
$ffi->attach('nimcp_brain_snapshot_delete'  => ['opaque', 'string'] => 'int');

# Brain probe
$ffi->attach('nimcp_brain_probe'           => ['opaque', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_broadcast_probe' => ['opaque'] => 'int');

# Brain COW
$ffi->attach('nimcp_brain_clone_cow'        => ['opaque'] => 'opaque');
$ffi->attach('nimcp_brain_snapshot_cow'     => ['opaque'] => 'opaque');
$ffi->attach('nimcp_brain_restore_cow'      => ['opaque', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_snapshot_destroy' => ['opaque'] => 'void');

# Brain working memory
$ffi->attach('nimcp_brain_working_memory_add' =>
    ['opaque', 'float[]', 'uint32', 'float'] => 'int');
$ffi->attach('nimcp_brain_working_memory_get' =>
    ['opaque', 'uint32', 'uint32*'] => 'opaque');
$ffi->attach('nimcp_brain_working_memory_stats' =>
    ['opaque', 'uint32*', 'uint32*'] => 'int');
$ffi->attach('nimcp_brain_working_memory_refresh' => ['opaque', 'uint32'] => 'int');

# Brain workspace
$ffi->attach('nimcp_brain_workspace_compete' =>
    ['opaque', 'int', 'float[]', 'uint32', 'float'] => 'int');
$ffi->attach('nimcp_brain_workspace_read' =>
    ['opaque', 'float[]', 'uint32', 'uint32*', 'int*'] => 'int');
$ffi->attach('nimcp_brain_workspace_subscribe'   => ['opaque', 'int'] => 'int');
$ffi->attach('nimcp_brain_workspace_unsubscribe' => ['opaque', 'int'] => 'int');
$ffi->attach('nimcp_brain_workspace_has_broadcast' =>
    ['opaque', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_workspace_stats' =>
    ['opaque', 'uint32*', 'uint32*', 'float*'] => 'int');

# Brain oscillations
$ffi->attach('nimcp_enable_complex_oscillations'    => ['opaque', 'int'] => 'int');
$ffi->attach('nimcp_is_complex_oscillations_enabled' => ['opaque'] => 'int');
$ffi->type('record(NIMCP::PhasorRecord)' => 'nimcp_phasor_t');
$ffi->attach('nimcp_get_oscillation_phasor' => ['opaque', 'uint32'] => 'nimcp_phasor_t');
$ffi->attach('nimcp_get_phase_coherence' => ['opaque', 'uint32[]', 'uint32'] => 'float');
$ffi->attach('nimcp_get_pac_modulation'  => ['opaque', 'float', 'float'] => 'float');

# Network API
$ffi->attach('nimcp_network_create'  =>
    ['uint32', 'uint32', 'uint32', 'float'] => 'opaque');
$ffi->attach('nimcp_network_destroy' => ['opaque'] => 'void');
$ffi->attach('nimcp_network_forward' =>
    ['opaque', 'float[]', 'uint32', 'float[]', 'uint32'] => 'int');
$ffi->attach('nimcp_network_train' =>
    ['opaque', 'float[]', 'uint32', 'float[]', 'uint32'] => 'int');

# Ethics API
$ffi->attach('nimcp_ethics_create'  => []     => 'opaque');
$ffi->attach('nimcp_ethics_destroy' => ['opaque'] => 'void');
$ffi->attach('nimcp_ethics_check'   =>
    ['opaque', 'float[]', 'uint32', 'float*'] => 'int');

# Knowledge API
$ffi->attach('nimcp_knowledge_create'   => []     => 'opaque');
$ffi->attach('nimcp_knowledge_destroy'  => ['opaque'] => 'void');
$ffi->attach('nimcp_knowledge_add_fact' =>
    ['opaque', 'string', 'string', 'string'] => 'int');
$ffi->attach('nimcp_knowledge_query' =>
    ['opaque', 'string', 'opaque', 'uint32'] => 'int');

# ============================================================================
# Helper functions
# ============================================================================

sub _check_status {
    my ($status, $context) = @_;
    return if $status == 0;
    my $msg = nimcp_get_error() // 'Unknown error';
    die "NIMCP error ($status) in $context: $msg\n";
}

# Allocate a zeroed C buffer
sub _alloc_buf {
    my ($size) = @_;
    return calloc(1, $size);
}

# ============================================================================
# Library Lifecycle
# ============================================================================

sub init {
    my $status = nimcp_init();
    _check_status($status, 'init');
    return 1;
}

sub shutdown { nimcp_shutdown() }

sub version { return nimcp_version() }

sub version_int { return nimcp_version_int() }

sub get_error { return nimcp_get_error() }

# ============================================================================
# NIMCP::Brain
# ============================================================================

package NIMCP::Brain;

sub new {
    my ($class, %args) = @_;
    my $name        = $args{name}        // 'default';
    my $size        = $args{size}        // NIMCP::BRAIN_SMALL;
    my $task        = $args{task}        // NIMCP::TASK_CLASSIFICATION;
    my $num_inputs  = $args{num_inputs}  // 10;
    my $num_outputs = $args{num_outputs} // 10;

    my $handle = NIMCP::nimcp_brain_create($name, $size, $task, $num_inputs, $num_outputs);
    die "Failed to create brain: " . NIMCP::nimcp_get_error() unless $handle;

    return bless {
        _handle     => $handle,
        _num_inputs => $num_inputs,
        _num_outputs => $num_outputs,
        _callbacks  => [],
    }, $class;
}

sub create_with_neurons {
    my ($class, %args) = @_;
    my $name         = $args{name}         // 'default';
    my $task         = $args{task}         // NIMCP::TASK_CLASSIFICATION;
    my $num_inputs   = $args{num_inputs}   // 10;
    my $num_outputs  = $args{num_outputs}  // 10;
    my $neuron_count = $args{neuron_count} // die "neuron_count required";

    my $handle = NIMCP::nimcp_brain_create_with_neurons(
        $name, $task, $num_inputs, $num_outputs, $neuron_count
    );
    die "Failed to create brain: " . NIMCP::nimcp_get_error() unless $handle;

    return bless {
        _handle      => $handle,
        _num_inputs  => $num_inputs,
        _num_outputs => $num_outputs,
        _callbacks   => [],
    }, $class;
}

sub create_full {
    my ($class, %args) = @_;
    my $name         = $args{name}         // 'default';
    my $task         = $args{task}         // NIMCP::TASK_CLASSIFICATION;
    my $num_inputs   = $args{num_inputs}   // 10;
    my $num_outputs  = $args{num_outputs}  // 10;
    my $neuron_count = $args{neuron_count} // die "neuron_count required";

    my $handle = NIMCP::nimcp_brain_create_full(
        $name, $task, $num_inputs, $num_outputs, $neuron_count
    );
    die "Failed to create brain: " . NIMCP::nimcp_get_error() unless $handle;

    return bless {
        _handle      => $handle,
        _num_inputs  => $num_inputs,
        _num_outputs => $num_outputs,
        _callbacks   => [],
    }, $class;
}

sub load {
    my ($class, $filepath) = @_;
    die "filepath required" unless $filepath;
    my $handle = NIMCP::nimcp_brain_load($filepath);
    die "Failed to load brain: " . NIMCP::nimcp_get_error() unless $handle;
    return bless { _handle => $handle, _callbacks => [] }, $class;
}

sub create_from_config {
    my ($class, $config_path) = @_;
    die "config_path required" unless $config_path;
    my $handle = NIMCP::nimcp_brain_create_from_config($config_path);
    die "Failed to create brain from config: " . NIMCP::nimcp_get_error() unless $handle;
    return bless { _handle => $handle, _callbacks => [] }, $class;
}

sub learn {
    my ($self, $features, $label, $confidence) = @_;
    $confidence //= 1.0;
    die "features must be array ref" unless ref $features eq 'ARRAY';
    my $n = scalar @$features;
    my $status = NIMCP::nimcp_brain_learn_example(
        $self->{_handle}, $features, $n, $label, $confidence
    );
    NIMCP::_check_status($status, 'learn');
    return $self;
}

sub learn_vector {
    my ($self, $features, $target, $label, $confidence) = @_;
    die "features must be array ref" unless ref $features eq 'ARRAY';
    die "target must be array ref" unless ref $target eq 'ARRAY';
    $label //= undef;
    $confidence //= 1.0;
    my $nf = scalar @$features;
    my $nt = scalar @$target;
    my $status = NIMCP::nimcp_brain_learn_vector(
        $self->{_handle}, $features, $nf, $target, $nt, $label, $confidence
    );
    NIMCP::_check_status($status, 'learn_vector');
    return $self;
}

sub learn_vector_batch {
    my ($self, $features_2d, $targets_2d, $learning_rate) = @_;
    die "features_2d must be array ref" unless ref $features_2d eq 'ARRAY';
    die "targets_2d must be array ref"  unless ref $targets_2d eq 'ARRAY';
    $learning_rate //= 0.001;
    my $num_examples  = scalar @$features_2d;
    my $num_features  = scalar @{$features_2d->[0]};
    my $target_size   = scalar @{$targets_2d->[0]};
    # Pack feature pointers and target pointers into contiguous arrays
    # For FFI we flatten and pass as opaque pointers to pointer arrays
    my @feat_flat = map { @$_ } @$features_2d;
    my @tgt_flat  = map { @$_ } @$targets_2d;
    my $feat_packed = pack("f*", @feat_flat);
    my $tgt_packed  = pack("f*", @tgt_flat);
    # Build array of pointers (each pointing into the packed buffer)
    my ($feat_buf, $feat_buf_sz) = FFI::Platypus::Buffer::scalar_to_buffer($feat_packed);
    my ($tgt_buf, $tgt_buf_sz)   = FFI::Platypus::Buffer::scalar_to_buffer($tgt_packed);
    my $ptr_size = $ffi->sizeof('opaque');
    my $feat_ptrs = '';
    my $tgt_ptrs  = '';
    for my $i (0 .. $num_examples - 1) {
        $feat_ptrs .= pack($ptr_size == 8 ? 'Q' : 'L', $feat_buf + $i * $num_features * 4);
        $tgt_ptrs  .= pack($ptr_size == 8 ? 'Q' : 'L', $tgt_buf  + $i * $target_size * 4);
    }
    my ($feat_ptrs_buf, undef) = FFI::Platypus::Buffer::scalar_to_buffer($feat_ptrs);
    my ($tgt_ptrs_buf, undef)  = FFI::Platypus::Buffer::scalar_to_buffer($tgt_ptrs);
    my $avg_loss = NIMCP::nimcp_brain_learn_vector_batch(
        $self->{_handle}, $feat_ptrs_buf, $tgt_ptrs_buf,
        $num_features, $target_size, $num_examples, $learning_rate
    );
    return $avg_loss;
}

sub learn_batch {
    my ($self, $features_2d, $labels, $confidences) = @_;
    die "features_2d must be array ref" unless ref $features_2d eq 'ARRAY';
    die "labels must be array ref"      unless ref $labels eq 'ARRAY';
    my $num_examples = scalar @$features_2d;
    my $num_features = scalar @{$features_2d->[0]};
    # Build pointer arrays for features, labels, confidences
    my $ptr_size = $ffi->sizeof('opaque');
    # Pack each feature row into a buffer and collect pointers
    my @feat_bufs;
    my $feat_ptrs = '';
    for my $i (0 .. $num_examples - 1) {
        my $packed = pack("f*", @{$features_2d->[$i]});
        my ($buf, undef) = FFI::Platypus::Buffer::scalar_to_buffer($packed);
        push @feat_bufs, $packed;  # prevent GC
        $feat_ptrs .= pack($ptr_size == 8 ? 'Q' : 'L', $buf);
    }
    my ($feat_ptrs_buf, undef) = FFI::Platypus::Buffer::scalar_to_buffer($feat_ptrs);
    # num_features_array = NULL (uniform)
    # Build label string pointer array
    my @label_bufs;
    my $label_ptrs = '';
    for my $i (0 .. $num_examples - 1) {
        my $str = $labels->[$i] . "\0";
        my ($buf, undef) = FFI::Platypus::Buffer::scalar_to_buffer($str);
        push @label_bufs, $str;
        $label_ptrs .= pack($ptr_size == 8 ? 'Q' : 'L', $buf);
    }
    my ($label_ptrs_buf, undef) = FFI::Platypus::Buffer::scalar_to_buffer($label_ptrs);
    # Confidences
    my $conf_buf;
    if ($confidences && ref $confidences eq 'ARRAY') {
        my $conf_packed = pack("f*", @$confidences);
        ($conf_buf, undef) = FFI::Platypus::Buffer::scalar_to_buffer($conf_packed);
    } else {
        $conf_buf = undef;
    }
    # losses_out
    my $losses_packed = pack("f*", (0.0) x $num_examples);
    my ($losses_buf, undef) = FFI::Platypus::Buffer::scalar_to_buffer($losses_packed);
    my $status = NIMCP::nimcp_brain_learn_batch(
        $self->{_handle}, $feat_ptrs_buf, undef, $label_ptrs_buf,
        $conf_buf, $num_examples, $losses_buf
    );
    NIMCP::_check_status($status, 'learn_batch');
    my $losses_data = FFI::Platypus::Buffer::buffer_to_scalar($losses_buf, $num_examples * 4);
    return [unpack("f$num_examples", $losses_data)];
}

sub learn_knowledge {
    my ($self, $text, $domain) = @_;
    $domain //= 10;  # general
    my $status = NIMCP::nimcp_brain_learn_knowledge($self->{_handle}, $text, $domain);
    NIMCP::_check_status($status, 'learn_knowledge');
    return $self;
}

sub learn_language {
    my ($self, $text) = @_;
    my $loss = 0.0;
    my $status = NIMCP::nimcp_brain_learn_language($self->{_handle}, $text, \$loss);
    NIMCP::_check_status($status, 'learn_language');
    return $loss;
}

sub learn_language_pair {
    my ($self, $input_text, $target_text, $learning_rate) = @_;
    $learning_rate //= 0.0;
    my $loss = 0.0;
    my $status = NIMCP::nimcp_brain_learn_language_pair(
        $self->{_handle}, $input_text, $target_text, $learning_rate, \$loss
    );
    NIMCP::_check_status($status, 'learn_language_pair');
    return $loss;
}

sub train_cognitive {
    my ($self, $text, $domain, $target_text, $learning_rate) = @_;
    $domain //= -1;
    $learning_rate //= 0.0;
    my $loss = 0.0;
    my $status = NIMCP::nimcp_brain_train_cognitive(
        $self->{_handle}, $text, $domain, $target_text, $learning_rate, \$loss
    );
    NIMCP::_check_status($status, 'train_cognitive');
    return $loss;
}

sub train_language {
    my ($self, $input_text, $target_text, $learning_rate) = @_;
    $learning_rate //= 0.0;
    my $loss = 0.0;
    my $status = NIMCP::nimcp_brain_train_language(
        $self->{_handle}, $input_text, $target_text, $learning_rate, \$loss
    );
    NIMCP::_check_status($status, 'train_language');
    return $loss;
}

sub predict {
    my ($self, $features) = @_;
    die "features must be array ref" unless ref $features eq 'ARRAY';
    my $n = scalar @$features;
    # Allocate output buffer for label (64 bytes)
    my $label_ptr = FFI::Platypus::Memory::calloc(1, 64);
    my $conf = 0.0;
    my $status = NIMCP::nimcp_brain_predict(
        $self->{_handle}, $features, $n, $label_ptr, \$conf
    );
    my $label = FFI::Platypus::Buffer::buffer_to_scalar($label_ptr, 64);
    FFI::Platypus::Memory::free($label_ptr);
    NIMCP::_check_status($status, 'predict');
    $label =~ s/\0.*//s;  # trim after null terminator
    return wantarray ? ($label, $conf) : { label => $label, confidence => $conf };
}

sub predict_fast {
    my ($self, $features) = @_;
    die "features must be array ref" unless ref $features eq 'ARRAY';
    my $n = scalar @$features;
    my $label_ptr = FFI::Platypus::Memory::calloc(1, 64);
    my $conf = 0.0;
    my $status = NIMCP::nimcp_brain_predict_fast(
        $self->{_handle}, $features, $n, $label_ptr, \$conf
    );
    my $label = FFI::Platypus::Buffer::buffer_to_scalar($label_ptr, 64);
    FFI::Platypus::Memory::free($label_ptr);
    NIMCP::_check_status($status, 'predict_fast');
    $label =~ s/\0.*//s;
    return wantarray ? ($label, $conf) : { label => $label, confidence => $conf };
}

sub predict_in_domain {
    my ($self, $features, $domain_prefix) = @_;
    die "features must be array ref" unless ref $features eq 'ARRAY';
    my $n = scalar @$features;
    my $label_ptr = FFI::Platypus::Memory::calloc(1, 64);
    my $conf = 0.0;
    my $status = NIMCP::nimcp_brain_predict_in_domain(
        $self->{_handle}, $features, $n, $domain_prefix, $label_ptr, \$conf
    );
    my $label = FFI::Platypus::Buffer::buffer_to_scalar($label_ptr, 64);
    FFI::Platypus::Memory::free($label_ptr);
    NIMCP::_check_status($status, 'predict_in_domain');
    $label =~ s/\0.*//s;
    return wantarray ? ($label, $conf) : { label => $label, confidence => $conf };
}

sub decide_full {
    my ($self, $features) = @_;
    die "features must be array ref" unless ref $features eq 'ARRAY';
    my $n = scalar @$features;
    my $label_ptr = FFI::Platypus::Memory::calloc(1, 64);
    my $expl_ptr  = FFI::Platypus::Memory::calloc(1, 512);
    my $conf = 0.0;
    my $num_outputs = $self->{_num_outputs} // 128;
    my @out_vec = (0.0) x $num_outputs;
    my $out_size = 0;
    my $active_neurons = 0;
    my $sparsity = 0.0;
    my $inference_us = 0;
    my $status = NIMCP::nimcp_brain_decide_full(
        $self->{_handle}, $features, $n,
        $label_ptr, \$conf, $expl_ptr,
        \@out_vec, \$out_size, \$active_neurons, \$sparsity, \$inference_us
    );
    my $label = FFI::Platypus::Buffer::buffer_to_scalar($label_ptr, 64);
    my $expl  = FFI::Platypus::Buffer::buffer_to_scalar($expl_ptr, 512);
    FFI::Platypus::Memory::free($label_ptr);
    FFI::Platypus::Memory::free($expl_ptr);
    NIMCP::_check_status($status, 'decide_full');
    $label =~ s/\0.*//s;
    $expl  =~ s/\0.*//s;
    return {
        label             => $label,
        confidence        => $conf,
        explanation       => $expl,
        output_vector     => [@out_vec[0 .. ($out_size > 0 ? $out_size - 1 : 0)]],
        active_neurons    => $active_neurons,
        sparsity          => $sparsity,
        inference_time_us => $inference_us,
    };
}

sub speak {
    my ($self, $semantic_input, $text_max_len) = @_;
    $text_max_len //= 1024;
    my $text_ptr = FFI::Platypus::Memory::calloc(1, $text_max_len);
    my $conf = 0.0;
    my $fluency = 0.0;
    my ($sem_arr, $sem_dim);
    if ($semantic_input && ref $semantic_input eq 'ARRAY') {
        $sem_arr = $semantic_input;
        $sem_dim = scalar @$semantic_input;
    } else {
        $sem_arr = [];
        $sem_dim = 0;
    }
    my $status = NIMCP::nimcp_brain_speak(
        $self->{_handle}, $sem_arr, $sem_dim, $text_ptr, $text_max_len, \$conf, \$fluency
    );
    my $text = FFI::Platypus::Buffer::buffer_to_scalar($text_ptr, $text_max_len);
    FFI::Platypus::Memory::free($text_ptr);
    NIMCP::_check_status($status, 'speak');
    $text =~ s/\0.*//s;
    return {
        text       => $text,
        confidence => $conf,
        fluency    => $fluency,
    };
}

sub generate_text {
    my ($self, %args) = @_;
    my $prompt        = $args{prompt};
    my $semantic_input = $args{semantic_input};
    my $text_max_len  = $args{text_max_len} // 1024;
    my $text_ptr = FFI::Platypus::Memory::calloc(1, $text_max_len);
    my $conf = 0.0;
    my $perplexity = 0.0;
    my ($sem_arr, $sem_dim);
    if ($semantic_input && ref $semantic_input eq 'ARRAY') {
        $sem_arr = $semantic_input;
        $sem_dim = scalar @$semantic_input;
    } else {
        $sem_arr = [];
        $sem_dim = 0;
    }
    my $status = NIMCP::nimcp_brain_generate_text(
        $self->{_handle}, $prompt, $sem_arr, $sem_dim,
        $text_ptr, $text_max_len, \$conf, \$perplexity
    );
    my $text = FFI::Platypus::Buffer::buffer_to_scalar($text_ptr, $text_max_len);
    FFI::Platypus::Memory::free($text_ptr);
    NIMCP::_check_status($status, 'generate_text');
    $text =~ s/\0.*//s;
    return {
        text       => $text,
        confidence => $conf,
        perplexity => $perplexity,
    };
}

sub comprehend {
    my ($self, $text, $semantic_dim) = @_;
    $semantic_dim //= 128;
    my @semantic = (0.0) x $semantic_dim;
    my $conf = 0.0;
    my $status = NIMCP::nimcp_brain_comprehend(
        $self->{_handle}, $text, \@semantic, $semantic_dim, \$conf
    );
    NIMCP::_check_status($status, 'comprehend');
    return {
        semantic   => \@semantic,
        confidence => $conf,
    };
}

sub produce_text {
    my ($self, $intent, $text_max_len) = @_;
    die "intent must be array ref" unless ref $intent eq 'ARRAY';
    $text_max_len //= 1024;
    my $intent_dim = scalar @$intent;
    my $text_ptr = FFI::Platypus::Memory::calloc(1, $text_max_len);
    my $conf = 0.0;
    my $status = NIMCP::nimcp_brain_produce_text(
        $self->{_handle}, $intent, $intent_dim, $text_ptr, $text_max_len, \$conf
    );
    my $text = FFI::Platypus::Buffer::buffer_to_scalar($text_ptr, $text_max_len);
    FFI::Platypus::Memory::free($text_ptr);
    NIMCP::_check_status($status, 'produce_text');
    $text =~ s/\0.*//s;
    return {
        text       => $text,
        confidence => $conf,
    };
}

sub grounded_respond {
    my ($self, $input_text, $response_max) = @_;
    $response_max //= 1024;
    my $resp_ptr = FFI::Platypus::Memory::calloc(1, $response_max);
    my $conf = 0.0;
    my $status = NIMCP::nimcp_brain_grounded_respond(
        $self->{_handle}, $input_text, $resp_ptr, $response_max, \$conf
    );
    my $response = FFI::Platypus::Buffer::buffer_to_scalar($resp_ptr, $response_max);
    FFI::Platypus::Memory::free($resp_ptr);
    NIMCP::_check_status($status, 'grounded_respond');
    $response =~ s/\0.*//s;
    return {
        response   => $response,
        confidence => $conf,
    };
}

sub creative_blend {
    my ($self, $vector_a, $vector_b, $blend_ratio, $text_max) = @_;
    die "vector_a must be array ref" unless ref $vector_a eq 'ARRAY';
    die "vector_b must be array ref" unless ref $vector_b eq 'ARRAY';
    $blend_ratio //= 0.5;
    $text_max //= 1024;
    my $vec_dim = scalar @$vector_a;
    my $text_ptr = FFI::Platypus::Memory::calloc(1, $text_max);
    my $status = NIMCP::nimcp_brain_creative_blend(
        $self->{_handle}, $vector_a, $vector_b, $vec_dim, $blend_ratio, $text_ptr, $text_max
    );
    my $text = FFI::Platypus::Buffer::buffer_to_scalar($text_ptr, $text_max);
    FFI::Platypus::Memory::free($text_ptr);
    NIMCP::_check_status($status, 'creative_blend');
    $text =~ s/\0.*//s;
    return $text;
}

sub ground_word {
    my ($self, $word, $features, $modality, $attention) = @_;
    die "features must be array ref" unless ref $features eq 'ARRAY';
    $modality  //= 0;
    $attention //= 1.0;
    my $dim = scalar @$features;
    my $status = NIMCP::nimcp_brain_ground_word(
        $self->{_handle}, $word, $features, $dim, $modality, $attention
    );
    NIMCP::_check_status($status, 'ground_word');
    return $self;
}

sub infer {
    my ($self, $features, $num_outputs) = @_;
    die "features must be array ref" unless ref $features eq 'ARRAY';
    $num_outputs //= $self->{_num_outputs} // 10;
    my $n = scalar @$features;
    my @out = (0.0) x $num_outputs;
    my $status = NIMCP::nimcp_brain_infer(
        $self->{_handle}, $features, $n, \@out, $num_outputs
    );
    NIMCP::_check_status($status, 'infer');
    return \@out;
}

sub save {
    my ($self, $filepath) = @_;
    my $status = NIMCP::nimcp_brain_save($self->{_handle}, $filepath);
    NIMCP::_check_status($status, 'save');
    return $self;
}

# --- Metrics / Stats ---

sub get_accuracy {
    my ($self) = @_;
    return NIMCP::nimcp_brain_get_accuracy($self->{_handle});
}

sub get_last_gradient_norm {
    my ($self) = @_;
    return NIMCP::nimcp_brain_get_last_gradient_norm($self->{_handle});
}

sub get_last_loss {
    my ($self) = @_;
    return NIMCP::nimcp_brain_get_last_loss($self->{_handle});
}

sub get_network_metrics {
    my ($self) = @_;
    my $ema_ann = 0.0; my $ema_cnn = 0.0; my $ema_snn = 0.0; my $ema_lnn = 0.0;
    my $ann_steps = 0; my $cnn_steps = 0; my $snn_steps = 0; my $lnn_steps = 0;
    my $ok = NIMCP::nimcp_brain_get_network_metrics(
        $self->{_handle},
        \$ema_ann, \$ema_cnn, \$ema_snn, \$ema_lnn,
        \$ann_steps, \$cnn_steps, \$snn_steps, \$lnn_steps
    );
    return undef unless $ok;
    return {
        ema_ann => $ema_ann, ema_cnn => $ema_cnn,
        ema_snn => $ema_snn, ema_lnn => $ema_lnn,
        ann_steps => $ann_steps, cnn_steps => $cnn_steps,
        snn_steps => $snn_steps, lnn_steps => $lnn_steps,
    };
}

sub get_cognitive_stats {
    my ($self) = @_;
    my $max_modules = 13;
    my $stats_buf  = FFI::Platypus::Memory::calloc($max_modules, 4);
    my $losses_buf = FFI::Platypus::Memory::calloc($max_modules, 4);
    my $count = 0;
    my $status = NIMCP::nimcp_brain_get_cognitive_stats(
        $self->{_handle}, $stats_buf, $losses_buf, \$count
    );
    my $stats_data  = FFI::Platypus::Buffer::buffer_to_scalar($stats_buf,  $max_modules * 4);
    my $losses_data = FFI::Platypus::Buffer::buffer_to_scalar($losses_buf, $max_modules * 4);
    FFI::Platypus::Memory::free($stats_buf);
    FFI::Platypus::Memory::free($losses_buf);
    NIMCP::_check_status($status, 'get_cognitive_stats');
    my @steps  = unpack("I$count", $stats_data);
    my @losses = unpack("f$count", $losses_data);
    return {
        steps  => \@steps,
        losses => \@losses,
        count  => $count,
    };
}

sub get_avatar_state {
    my ($self) = @_;
    # nimcp_avatar_state_t is a large struct; allocate generously
    my $buf_size = 2048;
    my $buf_ptr = FFI::Platypus::Memory::calloc(1, $buf_size);
    my $status = NIMCP::nimcp_brain_get_avatar_state($self->{_handle}, $buf_ptr);
    my $buf = FFI::Platypus::Buffer::buffer_to_scalar($buf_ptr, $buf_size);
    FFI::Platypus::Memory::free($buf_ptr);
    NIMCP::_check_status($status, 'get_avatar_state');
    return $buf;  # raw bytes; caller must unpack per struct layout
}

sub get_last_transcript {
    my ($self, $max_entries) = @_;
    $max_entries //= 32;
    # Each entry: char[256], plus float salience, float confidence, const char* module
    my $entry_buf = FFI::Platypus::Memory::calloc($max_entries, 256);
    my $sal_buf   = FFI::Platypus::Memory::calloc($max_entries, 4);
    my $conf_buf  = FFI::Platypus::Memory::calloc($max_entries, 4);
    my $mod_buf   = FFI::Platypus::Memory::calloc($max_entries, $ffi->sizeof('opaque'));
    my $count = NIMCP::nimcp_brain_get_last_transcript(
        $self->{_handle}, $entry_buf, $sal_buf, $conf_buf, $mod_buf, $max_entries
    );
    my @entries;
    if ($count > 0) {
        my $e_data = FFI::Platypus::Buffer::buffer_to_scalar($entry_buf, $max_entries * 256);
        my $s_data = FFI::Platypus::Buffer::buffer_to_scalar($sal_buf,   $count * 4);
        my $c_data = FFI::Platypus::Buffer::buffer_to_scalar($conf_buf,  $count * 4);
        my @sals  = unpack("f$count", $s_data);
        my @confs = unpack("f$count", $c_data);
        for my $i (0 .. $count - 1) {
            my $text = unpack('Z256', substr($e_data, $i * 256, 256));
            push @entries, {
                text       => $text,
                salience   => $sals[$i],
                confidence => $confs[$i],
            };
        }
    }
    FFI::Platypus::Memory::free($entry_buf);
    FFI::Platypus::Memory::free($sal_buf);
    FFI::Platypus::Memory::free($conf_buf);
    FFI::Platypus::Memory::free($mod_buf);
    return \@entries;
}

sub get_cortex_cnn_metrics {
    my ($self) = @_;
    my $max_cortices = 4;
    my $types_buf   = FFI::Platypus::Memory::calloc($max_cortices, 4);
    my $losses_buf  = FFI::Platypus::Memory::calloc($max_cortices, 4);
    my $fwd_buf     = FFI::Platypus::Memory::calloc($max_cortices, 8);
    my $bwd_buf     = FFI::Platypus::Memory::calloc($max_cortices, 8);
    my $norms_buf   = FFI::Platypus::Memory::calloc($max_cortices, 4);
    my $count = 0;
    my $status = NIMCP::nimcp_brain_get_cortex_cnn_metrics(
        $self->{_handle}, $types_buf, $losses_buf, $fwd_buf, $bwd_buf, $norms_buf, \$count
    );
    my @metrics;
    if ($status == 0 && $count > 0) {
        my $t_data = FFI::Platypus::Buffer::buffer_to_scalar($types_buf,  $count * 4);
        my $l_data = FFI::Platypus::Buffer::buffer_to_scalar($losses_buf, $count * 4);
        my $f_data = FFI::Platypus::Buffer::buffer_to_scalar($fwd_buf,    $count * 8);
        my $b_data = FFI::Platypus::Buffer::buffer_to_scalar($bwd_buf,    $count * 8);
        my $n_data = FFI::Platypus::Buffer::buffer_to_scalar($norms_buf,  $count * 4);
        my @types  = unpack("i$count", $t_data);
        my @losses = unpack("f$count", $l_data);
        my @fwds   = unpack("Q$count", $f_data);
        my @bwds   = unpack("Q$count", $b_data);
        my @norms  = unpack("f$count", $n_data);
        for my $i (0 .. $count - 1) {
            push @metrics, {
                type       => $types[$i],
                loss       => $losses[$i],
                fwd_steps  => $fwds[$i],
                bwd_steps  => $bwds[$i],
                embed_norm => $norms[$i],
            };
        }
    }
    FFI::Platypus::Memory::free($types_buf);
    FFI::Platypus::Memory::free($losses_buf);
    FFI::Platypus::Memory::free($fwd_buf);
    FFI::Platypus::Memory::free($bwd_buf);
    FFI::Platypus::Memory::free($norms_buf);
    NIMCP::_check_status($status, 'get_cortex_cnn_metrics');
    return \@metrics;
}

# --- Experience ---

sub experience {
    my ($self, $input, $output_size, $teacher_reward) = @_;
    die "input must be array ref" unless ref $input eq 'ARRAY';
    $output_size //= $self->{_num_outputs} // 10;
    $teacher_reward //= 0.0;
    my $input_size = scalar @$input;
    my @output = (0.0) x $output_size;
    # brain_experience_result_t: allocate generously
    my $result_buf = FFI::Platypus::Memory::calloc(1, 256);
    my $status = NIMCP::nimcp_brain_experience(
        $self->{_handle}, $input, $input_size, \@output, $output_size,
        $teacher_reward, $result_buf
    );
    FFI::Platypus::Memory::free($result_buf);
    NIMCP::_check_status($status, 'experience');
    return \@output;
}

sub experience_configure {
    my ($self, $config_ptr) = @_;
    my $status = NIMCP::nimcp_brain_experience_configure($self->{_handle}, $config_ptr);
    NIMCP::_check_status($status, 'experience_configure');
    return $self;
}

sub experience_correct {
    my ($self, $expected) = @_;
    die "expected must be array ref" unless ref $expected eq 'ARRAY';
    my $n = scalar @$expected;
    return NIMCP::nimcp_brain_experience_correct($self->{_handle}, $expected, $n);
}

sub experience_attend {
    my ($self, $modality, $strength) = @_;
    $strength //= 1.0;
    my $status = NIMCP::nimcp_brain_experience_attend($self->{_handle}, $modality, $strength);
    NIMCP::_check_status($status, 'experience_attend');
    return $self;
}

# --- Freeze ---

sub freeze {
    my ($self) = @_;
    my $status = NIMCP::nimcp_brain_freeze($self->{_handle});
    NIMCP::_check_status($status, 'freeze');
    return $self;
}

sub is_frozen {
    my ($self) = @_;
    return NIMCP::nimcp_brain_is_frozen($self->{_handle}) ? 1 : 0;
}

# --- Enable flags ---

sub enable_mixed_precision {
    my ($self, $enable) = @_;
    $enable //= 1;
    my $status = NIMCP::nimcp_brain_enable_mixed_precision($self->{_handle}, $enable ? 1 : 0);
    NIMCP::_check_status($status, 'enable_mixed_precision');
    return $self;
}

sub enable_gradient_checkpointing {
    my ($self, $enable, $checkpoint_interval) = @_;
    $enable //= 1;
    $checkpoint_interval //= 0;
    my $status = NIMCP::nimcp_brain_enable_gradient_checkpointing(
        $self->{_handle}, $enable ? 1 : 0, $checkpoint_interval
    );
    NIMCP::_check_status($status, 'enable_gradient_checkpointing');
    return $self;
}

sub enable_hemispheric {
    my ($self, $enable) = @_;
    $enable //= 1;
    my $status = NIMCP::nimcp_brain_enable_hemispheric($self->{_handle}, $enable ? 1 : 0);
    NIMCP::_check_status($status, 'enable_hemispheric');
    return $self;
}

sub enable_recurrent {
    my ($self, $enable, $max_iterations, $confidence_threshold, $blend_alpha) = @_;
    $enable //= 1;
    $max_iterations //= 3;
    $confidence_threshold //= 0.9;
    $blend_alpha //= 0.5;
    my $status = NIMCP::nimcp_brain_enable_recurrent(
        $self->{_handle}, $enable ? 1 : 0, $max_iterations, $confidence_threshold, $blend_alpha
    );
    NIMCP::_check_status($status, 'enable_recurrent');
    return $self;
}

sub enable_bptt {
    my ($self, $enable, $window_size, $discount) = @_;
    $enable //= 1;
    $window_size //= 8;
    $discount //= 0.95;
    my $status = NIMCP::nimcp_brain_enable_bptt(
        $self->{_handle}, $enable ? 1 : 0, $window_size, $discount
    );
    NIMCP::_check_status($status, 'enable_bptt');
    return $self;
}

sub enable_multi_network {
    my ($self) = @_;
    my $status = NIMCP::nimcp_brain_enable_multi_network($self->{_handle});
    NIMCP::_check_status($status, 'enable_multi_network');
    return $self;
}

sub enable_biological_plasticity {
    my ($self, $enable) = @_;
    $enable //= 1;
    my $status = NIMCP::nimcp_brain_enable_biological_plasticity($self->{_handle}, $enable ? 1 : 0);
    NIMCP::_check_status($status, 'enable_biological_plasticity');
    return $self;
}

# --- Set / Config ---

sub set_fast_training {
    my ($self, $enabled) = @_;
    $enabled //= 1;
    my $status = NIMCP::nimcp_brain_set_fast_training($self->{_handle}, $enabled ? 1 : 0);
    NIMCP::_check_status($status, 'set_fast_training');
    return $self;
}

sub set_task_type {
    my ($self, $task_type) = @_;
    my $status = NIMCP::nimcp_brain_set_task_type($self->{_handle}, $task_type);
    NIMCP::_check_status($status, 'set_task_type');
    return $self;
}

sub set_training_mode {
    my ($self, $active) = @_;
    $active //= 1;
    NIMCP::nimcp_brain_set_training_mode($self->{_handle}, $active ? 1 : 0);
    return $self;
}

sub set_network_ablation {
    my ($self, $train_cnn, $train_snn, $train_lnn) = @_;
    $train_cnn //= -1;
    $train_snn //= -1;
    $train_lnn //= -1;
    NIMCP::nimcp_brain_set_network_ablation(
        $self->{_handle}, $train_cnn, $train_snn, $train_lnn
    );
    return $self;
}

# --- Sensory / Cortex ---

sub submit_sensory {
    my ($self, $modality, $data, $width, $height, $channels, $n_segments) = @_;
    die "data must be array ref" unless ref $data eq 'ARRAY';
    $width    //= 0;
    $height   //= 0;
    $channels //= 0;
    $n_segments //= 0;
    my $num_elements = scalar @$data;
    my $status = NIMCP::nimcp_brain_submit_sensory(
        $self->{_handle}, $modality, $data, $num_elements,
        $width, $height, $channels, $n_segments
    );
    NIMCP::_check_status($status, 'submit_sensory');
    return $self;
}

sub visual_cortex_process {
    my ($self, $pixels, $width, $height, $channels, $max_features) = @_;
    die "pixels must be array ref" unless ref $pixels eq 'ARRAY';
    $max_features //= 256;
    my $num_pixels = scalar @$pixels;
    my @features = (0.0) x $max_features;
    my $feature_count = 0;
    my $status = NIMCP::nimcp_brain_visual_cortex_process(
        $self->{_handle}, $pixels, $num_pixels, $width, $height, $channels,
        \@features, $max_features, \$feature_count
    );
    NIMCP::_check_status($status, 'visual_cortex_process');
    return [@features[0 .. ($feature_count > 0 ? $feature_count - 1 : 0)]];
}

# --- Neuromodulation / Sleep / Substrate ---

sub medulla_get_arousal {
    my ($self) = @_;
    return NIMCP::nimcp_brain_medulla_get_arousal($self->{_handle});
}

sub bg_get_dopamine {
    my ($self) = @_;
    return NIMCP::nimcp_brain_bg_get_dopamine($self->{_handle});
}

sub sleep_get_pressure {
    my ($self) = @_;
    return NIMCP::nimcp_brain_sleep_get_pressure($self->{_handle});
}

sub substrate_get_health {
    my ($self) = @_;
    my $max_len = 64;
    my $buf_ptr = FFI::Platypus::Memory::calloc(1, $max_len);
    my $status = NIMCP::nimcp_brain_substrate_get_health($self->{_handle}, $buf_ptr, $max_len);
    my $health = FFI::Platypus::Buffer::buffer_to_scalar($buf_ptr, $max_len);
    FFI::Platypus::Memory::free($buf_ptr);
    NIMCP::_check_status($status, 'substrate_get_health');
    $health =~ s/\0.*//s;
    return $health;
}

# --- Sub-network creation / stats ---

sub lnn_create {
    my ($self, $n_sensory, $n_inter, $n_command, $n_output) = @_;
    my $status = NIMCP::nimcp_brain_lnn_create(
        $self->{_handle}, $n_sensory, $n_inter, $n_command, $n_output
    );
    NIMCP::_check_status($status, 'lnn_create');
    return $self;
}

sub lnn_get_stats {
    my ($self) = @_;
    my $fwd = 0; my $bwd = 0; my $ode = 0;
    my $avg_tau = 0.0; my $state_norm = 0.0; my $grad_norm = 0.0;
    my $nan_count = 0; my $inf_count = 0;
    my $status = NIMCP::nimcp_brain_lnn_get_stats(
        $self->{_handle}, \$fwd, \$bwd, \$ode, \$avg_tau,
        \$state_norm, \$grad_norm, \$nan_count, \$inf_count
    );
    NIMCP::_check_status($status, 'lnn_get_stats');
    return {
        forward_steps  => $fwd,
        backward_steps => $bwd,
        ode_evals      => $ode,
        avg_tau        => $avg_tau,
        state_norm     => $state_norm,
        gradient_norm  => $grad_norm,
        nan_count      => $nan_count,
        inf_count      => $inf_count,
    };
}

sub snn_get_stats {
    my ($self) = @_;
    my $total_steps = 0; my $total_spikes = 0;
    my $mean_rate = 0.0; my $sparsity = 0.0; my $synchrony = 0.0;
    my $silent = 0; my $hyperactive = 0;
    # health (int) and memory_bytes (size_t) via opaque buffers
    my $health_buf = FFI::Platypus::Memory::calloc(1, 4);
    my $mem_buf    = FFI::Platypus::Memory::calloc(1, 8);
    my $status = NIMCP::nimcp_brain_snn_get_stats(
        $self->{_handle}, \$total_steps, \$total_spikes,
        \$mean_rate, \$sparsity, \$synchrony,
        \$silent, \$hyperactive, $health_buf, $mem_buf
    );
    my $health_data = FFI::Platypus::Buffer::buffer_to_scalar($health_buf, 4);
    my $mem_data    = FFI::Platypus::Buffer::buffer_to_scalar($mem_buf, 8);
    FFI::Platypus::Memory::free($health_buf);
    FFI::Platypus::Memory::free($mem_buf);
    NIMCP::_check_status($status, 'snn_get_stats');
    my ($health) = unpack('i', $health_data);
    my ($memory) = unpack('Q', $mem_data);
    return {
        total_steps         => $total_steps,
        total_spikes        => $total_spikes,
        mean_firing_rate    => $mean_rate,
        sparsity            => $sparsity,
        synchrony           => $synchrony,
        silent_neurons      => $silent,
        hyperactive_neurons => $hyperactive,
        health              => $health,
        memory_bytes        => $memory,
    };
}

sub cnn_get_stats {
    my ($self) = @_;
    my $num_layers = 0;
    my $num_labels = 0;
    my $params_buf = FFI::Platypus::Memory::calloc(1, 8);
    my $active_buf = FFI::Platypus::Memory::calloc(1, 4);
    my $status = NIMCP::nimcp_brain_cnn_get_stats(
        $self->{_handle}, \$num_layers, $params_buf, \$num_labels, $active_buf
    );
    my $params_data = FFI::Platypus::Buffer::buffer_to_scalar($params_buf, 8);
    my $active_data = FFI::Platypus::Buffer::buffer_to_scalar($active_buf, 1);
    FFI::Platypus::Memory::free($params_buf);
    FFI::Platypus::Memory::free($active_buf);
    NIMCP::_check_status($status, 'cnn_get_stats');
    my ($num_params) = unpack('Q', $params_data);
    my ($active) = unpack('C', $active_data);
    return {
        num_layers     => $num_layers,
        num_parameters => $num_params,
        num_labels     => $num_labels,
        active         => $active ? 1 : 0,
    };
}

# --- Rubric ---

sub rubric {
    my ($self) = @_;
    # nimcp_rubric_t is a complex struct; allocate generously
    my $buf_size = 4096;
    my $buf_ptr = FFI::Platypus::Memory::calloc(1, $buf_size);
    my $status = NIMCP::nimcp_brain_rubric($self->{_handle}, $buf_ptr);
    my $buf = FFI::Platypus::Buffer::buffer_to_scalar($buf_ptr, $buf_size);
    FFI::Platypus::Memory::free($buf_ptr);
    NIMCP::_check_status($status, 'rubric');
    return $buf;  # raw bytes; caller must unpack per struct layout
}

# --- Cloud ---

sub connect_cloud {
    my ($self, $cloud_brain, $confidence_threshold, $enable_distillation) = @_;
    die "cloud_brain must be NIMCP::Brain" unless ref $cloud_brain eq 'NIMCP::Brain';
    $confidence_threshold //= 0.5;
    $enable_distillation  //= 1;
    my $status = NIMCP::nimcp_brain_connect_cloud(
        $self->{_handle}, $cloud_brain->{_handle},
        $confidence_threshold, $enable_distillation ? 1 : 0
    );
    NIMCP::_check_status($status, 'connect_cloud');
    return $self;
}

sub disconnect_cloud {
    my ($self) = @_;
    my $status = NIMCP::nimcp_brain_disconnect_cloud($self->{_handle});
    NIMCP::_check_status($status, 'disconnect_cloud');
    return $self;
}

# --- Training Pipeline ---

sub configure_training {
    my ($self, %config) = @_;
    # Build nimcp_training_config_t struct in memory
    # Layout verified from C struct definition in nimcp.h
    my $packed = pack('i i i f f f f f f I f I C x x x f C x x x f i i f f f i I C x x x',
        $config{loss_type}       // NIMCP::LOSS_CROSS_ENTROPY,
        $config{optimizer_type}  // NIMCP::OPT_ADAM,
        $config{scheduler_type}  // NIMCP::SCHED_COSINE,
        $config{learning_rate}   // 0.001,
        $config{weight_decay}    // 0.0,
        $config{momentum}        // 0.9,
        $config{beta1}           // 0.9,
        $config{beta2}           // 0.999,
        $config{epsilon}         // 1e-8,
        $config{scheduler_step_size} // 100,
        $config{scheduler_gamma}     // 0.1,
        $config{warmup_steps}        // 0,
        $config{enable_gradient_clipping} ? 1 : 0,
        $config{gradient_clip_value} // 1.0,
        $config{enable_biological_modulation} ? 1 : 0,
        $config{biological_blend}    // 0.5,
        $config{network_type}        // NIMCP::NETWORK_ADAPTIVE,
        $config{snn_method}          // NIMCP::SNN_TRAIN_STDP,
        $config{snn_eligibility_tau} // 20.0,
        $config{snn_reward_tau}      // 100.0,
        $config{snn_surrogate_beta}  // 5.0,
        $config{lnn_method}          // NIMCP::LNN_TRAIN_ADJOINT,
        $config{lnn_bptt_truncation} // 100,
        $config{lnn_use_adjoint_checkpointing} // 1 ? 1 : 0,
    );
    my ($config_ptr, $config_size) = FFI::Platypus::Buffer::scalar_to_buffer($packed);
    my $status = NIMCP::nimcp_brain_configure_training($self->{_handle}, $config_ptr);
    NIMCP::_check_status($status, 'configure_training');
    return $self;
}

sub train_step {
    my ($self, $features, $targets) = @_;
    die "features must be array ref" unless ref $features eq 'ARRAY';
    die "targets must be array ref" unless ref $targets eq 'ARRAY';
    my $nf = scalar @$features;
    my $nt = scalar @$targets;
    # nimcp_training_result_t: float loss, float lr, uint32 step, bool early_stopped(+pad), float grad_norm
    my $result_packed = pack('f f I C x x x f', 0.0, 0.0, 0, 0, 0.0);
    my ($result_ptr, $result_size) = FFI::Platypus::Buffer::scalar_to_buffer($result_packed);
    my $status = NIMCP::nimcp_brain_train_step(
        $self->{_handle}, $features, $nf, $targets, $nt, $result_ptr
    );
    NIMCP::_check_status($status, 'train_step');
    my $result_data = FFI::Platypus::Buffer::buffer_to_scalar($result_ptr, $result_size);
    my ($loss, $lr, $step, $early, $grad) = unpack('f f I C x x x f', $result_data);
    return {
        loss          => $loss,
        learning_rate => $lr,
        step          => $step,
        early_stopped => $early ? 1 : 0,
        gradient_norm => $grad,
    };
}

sub train_batch {
    my ($self, $features_2d, $targets_2d, $num_features, $num_targets) = @_;
    die "features_2d must be array ref" unless ref $features_2d eq 'ARRAY';
    die "targets_2d must be array ref" unless ref $targets_2d eq 'ARRAY';
    my $batch_size = scalar @$features_2d;
    $num_features //= scalar @{$features_2d->[0]};
    $num_targets  //= scalar @{$targets_2d->[0]};
    my @feat_flat = map { @$_ } @$features_2d;
    my @tgt_flat  = map { @$_ } @$targets_2d;
    my $result_packed = pack('f f I C x x x f', 0.0, 0.0, 0, 0, 0.0);
    my ($result_ptr, $result_size) = FFI::Platypus::Buffer::scalar_to_buffer($result_packed);
    my $status = NIMCP::nimcp_brain_train_batch(
        $self->{_handle}, \@feat_flat, \@tgt_flat,
        $batch_size, $num_features, $num_targets, $result_ptr
    );
    NIMCP::_check_status($status, 'train_batch');
    my $result_data = FFI::Platypus::Buffer::buffer_to_scalar($result_ptr, $result_size);
    my ($loss, $lr, $step, $early, $grad) = unpack('f f I C x x x f', $result_data);
    return {
        loss          => $loss,
        learning_rate => $lr,
        step          => $step,
        early_stopped => $early ? 1 : 0,
        gradient_norm => $grad,
    };
}

sub get_training_stats {
    my ($self) = @_;
    my $steps = 0;
    my $loss  = 0.0;
    my $lr    = 0.0;
    my $status = NIMCP::nimcp_brain_get_training_stats(
        $self->{_handle}, \$steps, \$loss, \$lr
    );
    NIMCP::_check_status($status, 'get_training_stats');
    return {
        total_steps => $steps,
        total_loss  => $loss,
        current_lr  => $lr,
    };
}

sub step_scheduler {
    my ($self, $validation_metric) = @_;
    $validation_metric //= 0.0;
    return NIMCP::nimcp_brain_step_scheduler($self->{_handle}, $validation_metric);
}

# --- Callbacks ---

sub enable_callbacks {
    my ($self, %config) = @_;
    # nimcp_callback_config_t:
    # bool(+pad) enable_auto_checkpoint, uint32 checkpoint_interval,
    # bool(+pad) enable_early_stopping, uint32 patience, float min_delta,
    # float divergence_threshold, uint32 log_interval
    my $packed = pack('C x x x I C x x x I f f I',
        $config{enable_auto_checkpoint} ? 1 : 0,
        $config{checkpoint_interval} // 1000,
        $config{enable_early_stopping} ? 1 : 0,
        $config{patience}            // 10,
        $config{min_delta}           // 0.0001,
        $config{divergence_threshold} // 10.0,
        $config{log_interval}        // 0,
    );
    my ($cfg_ptr, $cfg_size) = FFI::Platypus::Buffer::scalar_to_buffer($packed);
    my $status = NIMCP::nimcp_brain_enable_callbacks($self->{_handle}, $cfg_ptr);
    NIMCP::_check_status($status, 'enable_callbacks');
    return $self;
}

sub disable_callbacks {
    my ($self) = @_;
    my $status = NIMCP::nimcp_brain_disable_callbacks($self->{_handle});
    NIMCP::_check_status($status, 'disable_callbacks');
    return $self;
}

sub register_callback {
    my ($self, $event, $callback, $name) = @_;
    die "callback must be a code ref" unless ref $callback eq 'CODE';
    $name //= 'perl_callback';

    # Create a C closure for the callback trampoline
    my $closure = $ffi->closure(sub {
        my ($evt, $metrics_ptr, $user_data) = @_;
        my $action = $callback->($evt);
        return $action // NIMCP::CB_ACTION_CONTINUE;
    });
    $closure->sticky;  # prevent GC

    my $cb_ptr = $ffi->cast('(int,opaque,opaque)->int', 'opaque', $closure);
    my $cb_id = NIMCP::nimcp_brain_register_callback(
        $self->{_handle}, $event, $cb_ptr, undef, $name
    );
    die "Failed to register callback" unless $cb_id > 0;
    push @{$self->{_callbacks}}, $closure;
    return $cb_id;
}

sub unregister_callback {
    my ($self, $callback_id) = @_;
    my $status = NIMCP::nimcp_brain_unregister_callback($self->{_handle}, $callback_id);
    NIMCP::_check_status($status, 'unregister_callback');
    return $self;
}

sub get_callback_stats {
    my ($self) = @_;
    my $total = 0;
    my $avg   = 0.0;
    my $stops = 0;
    my $status = NIMCP::nimcp_brain_get_callback_stats(
        $self->{_handle}, \$total, \$avg, \$stops
    );
    NIMCP::_check_status($status, 'get_callback_stats');
    return {
        total_fired  => $total,
        avg_time_us  => $avg,
        early_stops  => $stops,
    };
}

# --- Resize ---

sub resize {
    my ($self, $new_count) = @_;
    return NIMCP::nimcp_brain_resize($self->{_handle}, $new_count);
}

sub auto_resize {
    my ($self) = @_;
    return NIMCP::nimcp_brain_auto_resize($self->{_handle});
}

sub get_neuron_count {
    my ($self) = @_;
    return NIMCP::nimcp_brain_get_neuron_count($self->{_handle});
}

sub get_utilization_metrics {
    my ($self) = @_;
    my $util = 0.0;
    my $sat  = 0.0;
    my $ok = NIMCP::nimcp_brain_get_utilization_metrics($self->{_handle}, \$util, \$sat);
    return undef unless $ok == 0;
    return {
        utilization => $util,
        saturation  => $sat,
    };
}

# --- Named Snapshots ---

sub snapshot_save {
    my ($self, $name, $description) = @_;
    $description //= '';
    my $status = NIMCP::nimcp_brain_snapshot_save($self->{_handle}, $name, $description);
    NIMCP::_check_status($status, 'snapshot_save');
    return $self;
}

sub snapshot_restore {
    my ($self, $name) = @_;
    my $new_handle = NIMCP::nimcp_brain_snapshot_restore($self->{_handle}, $name);
    die "Failed to restore snapshot: " . NIMCP::nimcp_get_error() unless $new_handle;
    return bless { _handle => $new_handle, _callbacks => [] }, ref($self);
}

sub snapshot_list {
    my ($self, $max_count) = @_;
    $max_count //= 32;
    # nimcp_brain_snapshot_info_t: char[128] + char[512] + uint64 + uint32 + bool + bool + pad = 656
    my $info_size = 656;
    my $buf_size = $info_size * $max_count;
    my $buf_ptr = FFI::Platypus::Memory::calloc(1, $buf_size);
    my $count = 0;
    my $status = NIMCP::nimcp_brain_snapshot_list(
        $self->{_handle}, $buf_ptr, $max_count, \$count
    );
    my @snapshots;
    if ($status == 0 && $count > 0) {
        my $buf = FFI::Platypus::Buffer::buffer_to_scalar($buf_ptr, $buf_size);
        for my $i (0 .. $count - 1) {
            my $offset = $i * $info_size;
            my $name = unpack('Z128', substr($buf, $offset, 128));
            my $desc = unpack('Z512', substr($buf, $offset + 128, 512));
            push @snapshots, { name => $name, description => $desc };
        }
    }
    FFI::Platypus::Memory::free($buf_ptr);
    NIMCP::_check_status($status, 'snapshot_list');
    return \@snapshots;
}

sub snapshot_delete {
    my ($self, $name) = @_;
    my $status = NIMCP::nimcp_brain_snapshot_delete($self->{_handle}, $name);
    NIMCP::_check_status($status, 'snapshot_delete');
    return $self;
}

# --- Probe ---

sub probe {
    my ($self) = @_;
    # nimcp_brain_probe_t is ~200 bytes. Allocate generously.
    my $buf_size = 512;
    my $buf_ptr = FFI::Platypus::Memory::calloc(1, $buf_size);
    my $status = NIMCP::nimcp_brain_probe($self->{_handle}, $buf_ptr);
    my $buf = FFI::Platypus::Buffer::buffer_to_scalar($buf_ptr, $buf_size);
    FFI::Platypus::Memory::free($buf_ptr);
    NIMCP::_check_status($status, 'probe');
    # Parse fields: task_name[64], size(int), task(int), num_neurons(uint32),
    # num_synapses(uint32), num_active_synapses(uint32), pad(4),
    # total_inferences(uint64), total_learning_steps(uint64)
    my $task_name = unpack('Z64', substr($buf, 0, 64));
    my ($size_val, $task_val, $neurons, $synapses, $active_syn) =
        unpack('i i I I I', substr($buf, 64, 20));
    my ($total_inf, $total_learn) = unpack('Q Q', substr($buf, 88, 16));
    return {
        task_name             => $task_name,
        size                  => $size_val,
        task                  => $task_val,
        num_neurons           => $neurons,
        num_synapses          => $synapses,
        num_active_synapses   => $active_syn,
        total_inferences      => $total_inf,
        total_learning_steps  => $total_learn,
    };
}

sub broadcast_probe {
    my ($self) = @_;
    my $status = NIMCP::nimcp_brain_broadcast_probe($self->{_handle});
    NIMCP::_check_status($status, 'broadcast_probe');
    return $self;
}

# --- COW (Copy-on-Write) ---

sub clone_cow {
    my ($self) = @_;
    my $handle = NIMCP::nimcp_brain_clone_cow($self->{_handle});
    die "Failed to clone brain: " . NIMCP::nimcp_get_error() unless $handle;
    return bless { _handle => $handle, _callbacks => [] }, ref($self);
}

sub snapshot_cow {
    my ($self) = @_;
    my $snap = NIMCP::nimcp_brain_snapshot_cow($self->{_handle});
    die "Failed to create COW snapshot" unless $snap;
    return NIMCP::BrainSnapshot->_new($snap);
}

sub restore_cow {
    my ($self, $snapshot) = @_;
    die "snapshot must be NIMCP::BrainSnapshot" unless ref $snapshot eq 'NIMCP::BrainSnapshot';
    my $status = NIMCP::nimcp_brain_restore_cow($self->{_handle}, $snapshot->{_handle});
    NIMCP::_check_status($status, 'restore_cow');
    return $self;
}

# --- Working Memory ---

sub working_memory_add {
    my ($self, $data, $salience) = @_;
    die "data must be array ref" unless ref $data eq 'ARRAY';
    $salience //= 0.5;
    my $n = scalar @$data;
    my $status = NIMCP::nimcp_brain_working_memory_add(
        $self->{_handle}, $data, $n, $salience
    );
    NIMCP::_check_status($status, 'working_memory_add');
    return $self;
}

sub working_memory_get {
    my ($self, $index) = @_;
    $index //= 0;
    my $size_out = 0;
    my $ptr = NIMCP::nimcp_brain_working_memory_get($self->{_handle}, $index, \$size_out);
    return undef unless $ptr;
    return [] unless $size_out > 0;
    my $bytes = $size_out * 4;
    my $buf = FFI::Platypus::Buffer::buffer_to_scalar($ptr, $bytes);
    return [unpack("f$size_out", $buf)];
}

sub working_memory_stats {
    my ($self) = @_;
    my $current  = 0;
    my $capacity = 0;
    my $status = NIMCP::nimcp_brain_working_memory_stats(
        $self->{_handle}, \$current, \$capacity
    );
    NIMCP::_check_status($status, 'working_memory_stats');
    return {
        current_size => $current,
        capacity     => $capacity,
    };
}

sub working_memory_refresh {
    my ($self, $index) = @_;
    my $status = NIMCP::nimcp_brain_working_memory_refresh($self->{_handle}, $index);
    NIMCP::_check_status($status, 'working_memory_refresh');
    return $self;
}

# --- Workspace ---

sub workspace_compete {
    my ($self, $module, $content, $strength) = @_;
    die "content must be array ref" unless ref $content eq 'ARRAY';
    $strength //= 0.5;
    my $dim = scalar @$content;
    return NIMCP::nimcp_brain_workspace_compete(
        $self->{_handle}, $module, $content, $dim, $strength
    );
}

sub workspace_read {
    my ($self, $max_dim) = @_;
    $max_dim //= 256;
    my @buf = (0.0) x $max_dim;
    my $actual_dim = 0;
    my $source = 0;
    my $status = NIMCP::nimcp_brain_workspace_read(
        $self->{_handle}, \@buf, $max_dim, \$actual_dim, \$source
    );
    return undef unless $status == 0;
    return {
        content => [@buf[0 .. ($actual_dim > 0 ? $actual_dim - 1 : 0)]],
        source_module => $source,
    };
}

sub workspace_subscribe {
    my ($self, $module) = @_;
    my $status = NIMCP::nimcp_brain_workspace_subscribe($self->{_handle}, $module);
    NIMCP::_check_status($status, 'workspace_subscribe');
    return $self;
}

sub workspace_unsubscribe {
    my ($self, $module) = @_;
    my $status = NIMCP::nimcp_brain_workspace_unsubscribe($self->{_handle}, $module);
    NIMCP::_check_status($status, 'workspace_unsubscribe');
    return $self;
}

sub workspace_has_broadcast {
    my ($self) = @_;
    # bool* output - use opaque buffer
    my $buf_ptr = FFI::Platypus::Memory::calloc(1, 4);
    my $status = NIMCP::nimcp_brain_workspace_has_broadcast($self->{_handle}, $buf_ptr);
    my $data = FFI::Platypus::Buffer::buffer_to_scalar($buf_ptr, 1);
    FFI::Platypus::Memory::free($buf_ptr);
    return 0 unless $status == 0;
    return unpack('C', $data) ? 1 : 0;
}

sub workspace_stats {
    my ($self) = @_;
    my $broadcasts  = 0;
    my $competitions = 0;
    my $avg_strength = 0.0;
    my $status = NIMCP::nimcp_brain_workspace_stats(
        $self->{_handle}, \$broadcasts, \$competitions, \$avg_strength
    );
    NIMCP::_check_status($status, 'workspace_stats');
    return {
        total_broadcasts   => $broadcasts,
        total_competitions => $competitions,
        avg_strength       => $avg_strength,
    };
}

# --- Oscillations ---

sub enable_oscillations {
    my ($self, $enable) = @_;
    $enable //= 1;
    return NIMCP::nimcp_enable_complex_oscillations($self->{_handle}, $enable ? 1 : 0);
}

sub is_oscillations_enabled {
    my ($self) = @_;
    return NIMCP::nimcp_is_complex_oscillations_enabled($self->{_handle});
}

sub get_phasor {
    my ($self, $neuron_id) = @_;
    my $phasor = NIMCP::nimcp_get_oscillation_phasor($self->{_handle}, $neuron_id);
    return {
        amplitude => $phasor->amplitude,
        phase     => $phasor->phase,
    };
}

sub get_phase_coherence {
    my ($self, $neuron_ids) = @_;
    die "neuron_ids must be array ref" unless ref $neuron_ids eq 'ARRAY';
    my $count = scalar @$neuron_ids;
    return NIMCP::nimcp_get_phase_coherence($self->{_handle}, $neuron_ids, $count);
}

sub get_pac_modulation {
    my ($self, $theta_freq, $gamma_freq) = @_;
    return NIMCP::nimcp_get_pac_modulation($self->{_handle}, $theta_freq, $gamma_freq);
}

sub DESTROY {
    my ($self) = @_;
    if ($self->{_handle}) {
        NIMCP::nimcp_brain_destroy($self->{_handle});
        $self->{_handle} = undef;
    }
}

# ============================================================================
# NIMCP::BrainSnapshot
# ============================================================================

package NIMCP::BrainSnapshot;

sub _new {
    my ($class, $handle) = @_;
    return bless { _handle => $handle }, $class;
}

sub DESTROY {
    my ($self) = @_;
    if ($self->{_handle}) {
        NIMCP::nimcp_brain_snapshot_destroy($self->{_handle});
        $self->{_handle} = undef;
    }
}

# ============================================================================
# NIMCP::Network
# ============================================================================

package NIMCP::Network;

sub new {
    my ($class, %args) = @_;
    my $num_inputs  = $args{num_inputs}  // die "num_inputs required";
    my $num_outputs = $args{num_outputs} // die "num_outputs required";
    my $num_hidden  = $args{num_hidden}  // 100;
    my $lr          = $args{learning_rate} // 0.01;

    my $handle = NIMCP::nimcp_network_create($num_inputs, $num_outputs, $num_hidden, $lr);
    die "Failed to create network: " . NIMCP::nimcp_get_error() unless $handle;

    return bless {
        _handle      => $handle,
        _num_inputs  => $num_inputs,
        _num_outputs => $num_outputs,
    }, $class;
}

sub forward {
    my ($self, $inputs) = @_;
    die "inputs must be array ref" unless ref $inputs eq 'ARRAY';
    my $ni = scalar @$inputs;
    my $no = $self->{_num_outputs};
    my @out = (0.0) x $no;
    my $status = NIMCP::nimcp_network_forward($self->{_handle}, $inputs, $ni, \@out, $no);
    NIMCP::_check_status($status, 'network_forward');
    return \@out;
}

sub train {
    my ($self, $inputs, $targets) = @_;
    die "inputs must be array ref" unless ref $inputs eq 'ARRAY';
    die "targets must be array ref" unless ref $targets eq 'ARRAY';
    my $ni = scalar @$inputs;
    my $nt = scalar @$targets;
    my $status = NIMCP::nimcp_network_train($self->{_handle}, $inputs, $ni, $targets, $nt);
    NIMCP::_check_status($status, 'network_train');
    return $self;
}

sub DESTROY {
    my ($self) = @_;
    if ($self->{_handle}) {
        NIMCP::nimcp_network_destroy($self->{_handle});
        $self->{_handle} = undef;
    }
}

# ============================================================================
# NIMCP::Ethics
# ============================================================================

package NIMCP::Ethics;

sub new {
    my ($class) = @_;
    my $handle = NIMCP::nimcp_ethics_create();
    die "Failed to create ethics module: " . NIMCP::nimcp_get_error() unless $handle;
    return bless { _handle => $handle }, $class;
}

sub check {
    my ($self, $situation) = @_;
    die "situation must be array ref" unless ref $situation eq 'ARRAY';
    my $n = scalar @$situation;
    my $score = 0.0;
    my $status = NIMCP::nimcp_ethics_check($self->{_handle}, $situation, $n, \$score);
    NIMCP::_check_status($status, 'ethics_check');
    return $score;
}

sub DESTROY {
    my ($self) = @_;
    if ($self->{_handle}) {
        NIMCP::nimcp_ethics_destroy($self->{_handle});
        $self->{_handle} = undef;
    }
}

# ============================================================================
# NIMCP::KnowledgeGraph
# ============================================================================

package NIMCP::KnowledgeGraph;

sub new {
    my ($class) = @_;
    my $handle = NIMCP::nimcp_knowledge_create();
    die "Failed to create knowledge graph: " . NIMCP::nimcp_get_error() unless $handle;
    return bless { _handle => $handle }, $class;
}

sub add_fact {
    my ($self, $subject, $predicate, $object) = @_;
    my $status = NIMCP::nimcp_knowledge_add_fact($self->{_handle}, $subject, $predicate, $object);
    NIMCP::_check_status($status, 'knowledge_add_fact');
    return $self;
}

sub query {
    my ($self, $query_str) = @_;
    my $buf_size = 1024;
    my $buf_ptr = FFI::Platypus::Memory::calloc(1, $buf_size);
    my $status = NIMCP::nimcp_knowledge_query($self->{_handle}, $query_str, $buf_ptr, $buf_size);
    my $result = FFI::Platypus::Buffer::buffer_to_scalar($buf_ptr, $buf_size);
    FFI::Platypus::Memory::free($buf_ptr);
    NIMCP::_check_status($status, 'knowledge_query');
    $result =~ s/\0.*//s;
    return $result;
}

sub DESTROY {
    my ($self) = @_;
    if ($self->{_handle}) {
        NIMCP::nimcp_knowledge_destroy($self->{_handle});
        $self->{_handle} = undef;
    }
}

1;

__END__

=head1 NAME

NIMCP - Perl bindings for Neural Interface Message Communication Protocol (v2.6.4)

=head1 SYNOPSIS

    use NIMCP;

    NIMCP::init();
    print "NIMCP version: " . NIMCP::version() . "\n";

    my $brain = NIMCP::Brain->new(
        name => 'classifier', size => NIMCP::BRAIN_SMALL,
        task => NIMCP::TASK_CLASSIFICATION, num_inputs => 5, num_outputs => 3,
    );

    $brain->learn([1.0, 2.0, 3.0, 4.0, 5.0], 'class_a', 0.95);
    my ($label, $confidence) = $brain->predict([1.5, 2.5, 3.5, 4.5, 5.5]);

    # Fast prediction (training loop)
    my ($label2, $conf2) = $brain->predict_fast([1.0, 2.0, 3.0, 4.0, 5.0]);

    # Language
    $brain->learn_language("The cat sat on the mat");
    my $result = $brain->speak();
    print "Brain says: $result->{text}\n";

    # Cloud
    my $cloud = NIMCP::Brain->create_full(
        name => 'cloud', task => NIMCP::TASK_CLASSIFICATION,
        num_inputs => 5, num_outputs => 3, neuron_count => 100000,
    );
    $brain->connect_cloud($cloud, 0.5, 1);

    my $net = NIMCP::Network->new(num_inputs => 10, num_outputs => 5);
    my $outputs = $net->forward([1..10]);

    NIMCP::shutdown();

=head1 DESCRIPTION

Complete Perl bindings for the NIMCP C library using FFI::Platypus.
Wraps the entire public C API including Brain, Network, Ethics,
KnowledgeGraph, training pipeline, callbacks, COW snapshots,
working memory, global workspace, oscillations, language production,
grounded language, cognitive training, sub-network stats, sensory
processing, cloud inference, and biological plasticity controls.

=head1 LICENSE

MIT License

=cut
