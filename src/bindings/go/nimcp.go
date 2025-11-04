// Package nimcp provides Go bindings for NIMCP using CGo
package nimcp

/*
#cgo CFLAGS: -I../../include
#cgo LDFLAGS: -L../../../build/src/lib -lnimcp_core -Wl,-rpath,${SRCDIR}/../../../build/src/lib
#include "../../include/nimcp.h"
#include <stdlib.h>
*/
import "C"
import (
	"errors"
	"unsafe"
)

// BrainSize represents brain size presets
type BrainSize int

const (
	BrainTiny   BrainSize = 0
	BrainSmall  BrainSize = 1
	BrainMedium BrainSize = 2
	BrainLarge  BrainSize = 3
)

// BrainTask represents task templates
type BrainTask int

const (
	TaskClassification  BrainTask = 0
	TaskRegression      BrainTask = 1
	TaskPatternMatching BrainTask = 2
	TaskSequence        BrainTask = 3
	TaskAssociation     BrainTask = 4
)

// Brain represents a NIMCP brain instance
type Brain struct {
	handle C.nimcp_brain_t
}

// Network represents a NIMCP neural network instance
type Network struct {
	handle     C.nimcp_network_t
	numOutputs int
}

// Init initializes the NIMCP library
func Init() error {
	status := C.nimcp_init()
	if status != 0 {
		return errors.New("failed to initialize NIMCP")
	}
	return nil
}

// Version returns the NIMCP version string
func Version() string {
	return C.GoString(C.nimcp_version())
}

// GetError returns the last error message
func GetError() string {
	return C.GoString(C.nimcp_get_error())
}

// NewBrain creates a new brain
func NewBrain(name string, size BrainSize, task BrainTask, numInputs, numOutputs uint32) (*Brain, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	handle := C.nimcp_brain_create(cName, C.int(size), C.int(task), C.uint(numInputs), C.uint(numOutputs))
	if handle == nil {
		return nil, errors.New(GetError())
	}

	return &Brain{handle: handle}, nil
}

// Destroy frees the brain resources
func (b *Brain) Destroy() {
	if b.handle != nil {
		C.nimcp_brain_destroy(b.handle)
		b.handle = nil
	}
}

// Learn teaches the brain from an example
func (b *Brain) Learn(features []float32, label string, confidence float32) error {
	cLabel := C.CString(label)
	defer C.free(unsafe.Pointer(cLabel))

	status := C.nimcp_brain_learn_example(
		b.handle,
		(*C.float)(unsafe.Pointer(&features[0])),
		C.uint(len(features)),
		cLabel,
		C.float(confidence),
	)

	if status != 0 {
		return errors.New(GetError())
	}
	return nil
}

// Predict makes a prediction
func (b *Brain) Predict(features []float32) (string, float32, error) {
	var label [64]C.char
	var confidence C.float

	status := C.nimcp_brain_predict(
		b.handle,
		(*C.float)(unsafe.Pointer(&features[0])),
		C.uint(len(features)),
		&label[0],
		&confidence,
	)

	if status != 0 {
		return "", 0, errors.New(GetError())
	}

	return C.GoString(&label[0]), float32(confidence), nil
}

// Save saves the brain to a file
func (b *Brain) Save(filepath string) error {
	cPath := C.CString(filepath)
	defer C.free(unsafe.Pointer(cPath))

	status := C.nimcp_brain_save(b.handle, cPath)
	if status != 0 {
		return errors.New(GetError())
	}
	return nil
}

// LoadBrain loads a brain from a file
func LoadBrain(filepath string) (*Brain, error) {
	cPath := C.CString(filepath)
	defer C.free(unsafe.Pointer(cPath))

	handle := C.nimcp_brain_load(cPath)
	if handle == nil {
		return nil, errors.New(GetError())
	}

	return &Brain{handle: handle}, nil
}

// NewNetwork creates a new neural network
func NewNetwork(numInputs, numOutputs, numHidden uint32, learningRate float32) (*Network, error) {
	handle := C.nimcp_network_create(
		C.uint(numInputs),
		C.uint(numOutputs),
		C.uint(numHidden),
		C.float(learningRate),
	)

	if handle == nil {
		return nil, errors.New(GetError())
	}

	return &Network{
		handle:     handle,
		numOutputs: int(numOutputs),
	}, nil
}

// Destroy frees the network resources
func (n *Network) Destroy() {
	if n.handle != nil {
		C.nimcp_network_destroy(n.handle)
		n.handle = nil
	}
}

// Forward performs a forward pass through the network
func (n *Network) Forward(inputs []float32) ([]float32, error) {
	outputs := make([]float32, n.numOutputs)

	status := C.nimcp_network_forward(
		n.handle,
		(*C.float)(unsafe.Pointer(&inputs[0])),
		C.uint(len(inputs)),
		(*C.float)(unsafe.Pointer(&outputs[0])),
		C.uint(len(outputs)),
	)

	if status != 0 {
		return nil, errors.New(GetError())
	}

	return outputs, nil
}
