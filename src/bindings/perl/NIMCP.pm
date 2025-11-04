#!/usr/bin/env perl
#
# NIMCP Perl bindings using XS (eXternal Subroutine)
# Wraps the unified nimcp.h C API
#

package NIMCP;

use 5.010;
use strict;
use warnings;

our $VERSION = '2.6.1';

require XSLoader;
XSLoader::load('NIMCP', $VERSION);

# Export constants
use constant {
    BRAIN_TINY   => 0,
    BRAIN_SMALL  => 1,
    BRAIN_MEDIUM => 2,
    BRAIN_LARGE  => 3,

    TASK_CLASSIFICATION   => 0,
    TASK_REGRESSION       => 1,
    TASK_PATTERN_MATCHING => 2,
    TASK_SEQUENCE         => 3,
    TASK_ASSOCIATION      => 4,

    STATUS_OK            => 0,
    STATUS_ERROR         => -1,
    STATUS_ERROR_NULL    => -2,
    STATUS_ERROR_INVALID => -3,
    STATUS_ERROR_MEMORY  => -4,
    STATUS_ERROR_IO      => -5,
};

# Initialize library
sub init {
    return _nimcp_init();
}

# Get version
sub version {
    return _nimcp_version();
}

# Brain package
package NIMCP::Brain;

sub new {
    my ($class, %args) = @_;

    my $name = $args{name} || 'default';
    my $size = $args{size} || $NIMCP::BRAIN_SMALL;
    my $task = $args{task} || $NIMCP::TASK_CLASSIFICATION;
    my $num_inputs = $args{num_inputs} || 10;
    my $num_outputs = $args{num_outputs} || 10;

    my $handle = NIMCP::_brain_create($name, $size, $task, $num_inputs, $num_outputs);

    die "Failed to create brain: " . NIMCP::_get_error() unless $handle;

    my $self = {
        _handle => $handle,
    };

    return bless $self, $class;
}

sub learn {
    my ($self, $features, $label, $confidence) = @_;
    $confidence //= 1.0;

    my $status = NIMCP::_brain_learn($self->{_handle}, $features, $label, $confidence);

    die "Learn failed: " . NIMCP::_get_error() if $status != 0;

    return $self;
}

sub predict {
    my ($self, $features) = @_;

    my ($label, $confidence) = NIMCP::_brain_predict($self->{_handle}, $features);

    die "Predict failed: " . NIMCP::_get_error() unless defined $label;

    return wantarray ? ($label, $confidence) : { label => $label, confidence => $confidence };
}

sub save {
    my ($self, $filepath) = @_;

    my $status = NIMCP::_brain_save($self->{_handle}, $filepath);

    die "Save failed: " . NIMCP::_get_error() if $status != 0;

    return $self;
}

sub load {
    my ($class, $filepath) = @_;

    my $handle = NIMCP::_brain_load($filepath);

    die "Load failed: " . NIMCP::_get_error() unless $handle;

    my $self = {
        _handle => $handle,
    };

    return bless $self, $class;
}

sub DESTROY {
    my ($self) = @_;
    NIMCP::_brain_destroy($self->{_handle}) if $self->{_handle};
}

# Network package
package NIMCP::Network;

sub new {
    my ($class, %args) = @_;

    my $num_inputs = $args{num_inputs} || die "num_inputs required";
    my $num_outputs = $args{num_outputs} || die "num_outputs required";
    my $num_hidden = $args{num_hidden} || 100;
    my $learning_rate = $args{learning_rate} || 0.01;

    my $handle = NIMCP::_network_create($num_inputs, $num_outputs, $num_hidden, $learning_rate);

    die "Failed to create network: " . NIMCP::_get_error() unless $handle;

    my $self = {
        _handle => $handle,
        num_outputs => $num_outputs,
    };

    return bless $self, $class;
}

sub forward {
    my ($self, $inputs) = @_;

    my $outputs = NIMCP::_network_forward($self->{_handle}, $inputs, $self->{num_outputs});

    die "Forward failed: " . NIMCP::_get_error() unless defined $outputs;

    return $outputs;
}

sub DESTROY {
    my ($self) = @_;
    NIMCP::_network_destroy($self->{_handle}) if $self->{_handle};
}

1;

__END__

=head1 NAME

NIMCP - Perl bindings for Neural Interface Message Communication Protocol

=head1 SYNOPSIS

    use NIMCP;

    # Initialize library
    NIMCP::init();

    # Create a brain
    my $brain = NIMCP::Brain->new(
        name => 'classifier',
        size => NIMCP::BRAIN_SMALL,
        task => NIMCP::TASK_CLASSIFICATION,
        num_inputs => 5,
        num_outputs => 3
    );

    # Learn from examples
    $brain->learn([1.0, 2.0, 3.0, 4.0, 5.0], 'class_a', 0.95);

    # Make predictions
    my ($label, $confidence) = $brain->predict([1.5, 2.5, 3.5, 4.5, 5.5]);
    print "Predicted: $label with confidence $confidence\n";

    # Save/load
    $brain->save('brain.dat');
    my $loaded_brain = NIMCP::Brain->load('brain.dat');

    # Create a neural network
    my $network = NIMCP::Network->new(
        num_inputs => 10,
        num_outputs => 5,
        num_hidden => 20,
        learning_rate => 0.01
    );

    my $outputs = $network->forward([1.0, 2.0, ..., 10.0]);

=head1 DESCRIPTION

NIMCP provides Perl bindings to the NIMCP C library for neural network
and cognitive computing applications.

=head1 AUTHOR

NIMCP Contributors

=head1 LICENSE

MIT License

=cut
