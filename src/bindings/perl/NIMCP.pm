#!/usr/bin/env perl
#
# NIMCP Perl bindings using FFI::Platypus
# Wraps the unified nimcp.h C API (v2.6.3)
#

package NIMCP;

use 5.010;
use strict;
use warnings;

our $VERSION = '2.6.3';

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

# Brain training
$ffi->attach('nimcp_brain_configure_training' => ['opaque', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_train_step' =>
    ['opaque', 'float[]', 'uint32', 'float[]', 'uint32', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_train_batch' =>
    ['opaque', 'float[]', 'float[]', 'uint32', 'uint32', 'uint32', 'opaque'] => 'int');
$ffi->attach('nimcp_brain_get_training_stats' =>
    ['opaque', 'uint64*', 'float*', 'float*'] => 'int');
$ffi->attach('nimcp_brain_step_scheduler' => ['opaque', 'float'] => 'float');

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

NIMCP - Perl bindings for Neural Interface Message Communication Protocol (v2.6.3)

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

    my $net = NIMCP::Network->new(num_inputs => 10, num_outputs => 5);
    my $outputs = $net->forward([1..10]);

    NIMCP::shutdown();

=head1 DESCRIPTION

Complete Perl bindings for the NIMCP C library using FFI::Platypus.
Wraps the entire public C API including Brain, Network, Ethics,
KnowledgeGraph, training pipeline, callbacks, COW snapshots,
working memory, global workspace, and oscillations.

=head1 LICENSE

MIT License

=cut
