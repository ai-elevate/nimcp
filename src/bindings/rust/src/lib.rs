//! NIMCP Rust Bindings
//!
//! Safe Rust wrapper around the NIMCP C library.
//! Uses only the public nimcp.h API via FFI.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_uint};
use std::ptr;

/// Opaque handle types
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

/// Brain size presets
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum BrainSize {
    Tiny = 0,
    Small = 1,
    Medium = 2,
    Large = 3,
}

/// Brain task types
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum BrainTask {
    Classification = 0,
    Regression = 1,
    PatternMatching = 2,
    Sequence = 3,
    Association = 4,
}

/// Return codes
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Status {
    Ok = 0,
    Error = -1,
    ErrorNullArg = -2,
    ErrorInvalid = -3,
    ErrorMemory = -4,
    ErrorIO = -5,
}

/// FFI declarations
extern "C" {
    fn nimcp_init() -> c_int;
    fn nimcp_shutdown();
    fn nimcp_version() -> *const c_char;
    fn nimcp_get_error() -> *const c_char;

    fn nimcp_brain_create(
        name: *const c_char,
        size: c_int,
        task: c_int,
        num_inputs: c_uint,
        num_outputs: c_uint,
    ) -> *mut NimcpBrainHandle;
    fn nimcp_brain_destroy(brain: *mut NimcpBrainHandle);
    fn nimcp_brain_learn_example(
        brain: *mut NimcpBrainHandle,
        features: *const c_float,
        num_features: c_uint,
        label: *const c_char,
        confidence: c_float,
    ) -> c_int;
    fn nimcp_brain_predict(
        brain: *mut NimcpBrainHandle,
        features: *const c_float,
        num_features: c_uint,
        out_label: *mut c_char,
        out_confidence: *mut c_float,
    ) -> c_int;
    fn nimcp_brain_save(brain: *mut NimcpBrainHandle, filepath: *const c_char) -> c_int;
    fn nimcp_brain_load(filepath: *const c_char) -> *mut NimcpBrainHandle;

    fn nimcp_network_create(
        num_inputs: c_uint,
        num_outputs: c_uint,
        num_hidden: c_uint,
        learning_rate: c_float,
    ) -> *mut NimcpNetworkHandle;
    fn nimcp_network_destroy(network: *mut NimcpNetworkHandle);
    fn nimcp_network_forward(
        network: *mut NimcpNetworkHandle,
        inputs: *const c_float,
        num_inputs: c_uint,
        outputs: *mut c_float,
        num_outputs: c_uint,
    ) -> c_int;
}

/// Safe Rust wrapper for NIMCP Brain
pub struct Brain {
    handle: *mut NimcpBrainHandle,
}

impl Brain {
    pub fn new(name: &str, size: BrainSize, task: BrainTask, num_inputs: u32, num_outputs: u32) -> Result<Self, String> {
        let c_name = CString::new(name).map_err(|e| e.to_string())?;

        unsafe {
            let handle = nimcp_brain_create(
                c_name.as_ptr(),
                size as c_int,
                task as c_int,
                num_inputs,
                num_outputs,
            );

            if handle.is_null() {
                let error = CStr::from_ptr(nimcp_get_error()).to_string_lossy();
                Err(error.into_owned())
            } else {
                Ok(Brain { handle })
            }
        }
    }

    pub fn learn(&mut self, features: &[f32], label: &str, confidence: f32) -> Result<(), String> {
        let c_label = CString::new(label).map_err(|e| e.to_string())?;

        unsafe {
            let status = nimcp_brain_learn_example(
                self.handle,
                features.as_ptr(),
                features.len() as c_uint,
                c_label.as_ptr(),
                confidence,
            );

            if status == Status::Ok as c_int {
                Ok(())
            } else {
                let error = CStr::from_ptr(nimcp_get_error()).to_string_lossy();
                Err(error.into_owned())
            }
        }
    }

    pub fn predict(&self, features: &[f32]) -> Result<(String, f32), String> {
        let mut label_buf = vec![0u8; 64];
        let mut confidence: c_float = 0.0;

        unsafe {
            let status = nimcp_brain_predict(
                self.handle,
                features.as_ptr(),
                features.len() as c_uint,
                label_buf.as_mut_ptr() as *mut c_char,
                &mut confidence,
            );

            if status == Status::Ok as c_int {
                let label = CStr::from_ptr(label_buf.as_ptr() as *const c_char)
                    .to_string_lossy()
                    .into_owned();
                Ok((label, confidence))
            } else {
                let error = CStr::from_ptr(nimcp_get_error()).to_string_lossy();
                Err(error.into_owned())
            }
        }
    }

    pub fn save(&self, filepath: &str) -> Result<(), String> {
        let c_filepath = CString::new(filepath).map_err(|e| e.to_string())?;

        unsafe {
            let status = nimcp_brain_save(self.handle, c_filepath.as_ptr());

            if status == Status::Ok as c_int {
                Ok(())
            } else {
                let error = CStr::from_ptr(nimcp_get_error()).to_string_lossy();
                Err(error.into_owned())
            }
        }
    }

    pub fn load(filepath: &str) -> Result<Self, String> {
        let c_filepath = CString::new(filepath).map_err(|e| e.to_string())?;

        unsafe {
            let handle = nimcp_brain_load(c_filepath.as_ptr());

            if handle.is_null() {
                let error = CStr::from_ptr(nimcp_get_error()).to_string_lossy();
                Err(error.into_owned())
            } else {
                Ok(Brain { handle })
            }
        }
    }
}

impl Drop for Brain {
    fn drop(&mut self) {
        unsafe {
            nimcp_brain_destroy(self.handle);
        }
    }
}

/// Safe Rust wrapper for NIMCP Network
pub struct Network {
    handle: *mut NimcpNetworkHandle,
    num_outputs: usize,
}

impl Network {
    pub fn new(num_inputs: u32, num_outputs: u32, num_hidden: u32, learning_rate: f32) -> Result<Self, String> {
        unsafe {
            let handle = nimcp_network_create(num_inputs, num_outputs, num_hidden, learning_rate);

            if handle.is_null() {
                let error = CStr::from_ptr(nimcp_get_error()).to_string_lossy();
                Err(error.into_owned())
            } else {
                Ok(Network {
                    handle,
                    num_outputs: num_outputs as usize,
                })
            }
        }
    }

    pub fn forward(&self, inputs: &[f32]) -> Result<Vec<f32>, String> {
        let mut outputs = vec![0.0f32; self.num_outputs];

        unsafe {
            let status = nimcp_network_forward(
                self.handle,
                inputs.as_ptr(),
                inputs.len() as c_uint,
                outputs.as_mut_ptr(),
                outputs.len() as c_uint,
            );

            if status == Status::Ok as c_int {
                Ok(outputs)
            } else {
                let error = CStr::from_ptr(nimcp_get_error()).to_string_lossy();
                Err(error.into_owned())
            }
        }
    }
}

impl Drop for Network {
    fn drop(&mut self) {
        unsafe {
            nimcp_network_destroy(self.handle);
        }
    }
}

/// Initialize the NIMCP library
pub fn init() -> Result<(), String> {
    unsafe {
        let status = nimcp_init();
        if status == Status::Ok as c_int {
            Ok(())
        } else {
            Err("Failed to initialize NIMCP".to_string())
        }
    }
}

/// Get NIMCP version
pub fn version() -> String {
    unsafe {
        let c_str = nimcp_version();
        CStr::from_ptr(c_str).to_string_lossy().into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }

    #[test]
    fn test_brain_create() {
        init().unwrap();
        let brain = Brain::new("test", BrainSize::Tiny, BrainTask::Classification, 5, 3);
        assert!(brain.is_ok());
    }

    #[test]
    fn test_network_create() {
        init().unwrap();
        let network = Network::new(10, 5, 20, 0.01);
        assert!(network.is_ok());
    }
}
