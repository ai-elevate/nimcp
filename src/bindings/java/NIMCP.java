/**
 * Java bindings for NIMCP using JNI
 * Wraps the unified nimcp.h C API
 */
package com.nimcp;

public class NIMCP {
    static {
        System.loadLibrary("nimcp_core");
        System.loadLibrary("nimcp_jni");
    }

    // Native initialization
    private static native int nativeInit();
    private static native void nativeShutdown();
    private static native String nativeVersion();
    private static native String nativeGetError();

    // Initialize the library
    public static void init() throws NIMCPException {
        if (nativeInit() != 0) {
            throw new NIMCPException("Failed to initialize NIMCP");
        }
    }

    // Get version
    public static String version() {
        return nativeVersion();
    }

    // Brain class
    public static class Brain {
        private long handle; // Opaque pointer to nimcp_brain_t

        public enum Size {
            TINY(0), SMALL(1), MEDIUM(2), LARGE(3);
            private final int value;
            Size(int value) { this.value = value; }
            public int getValue() { return value; }
        }

        public enum Task {
            CLASSIFICATION(0), REGRESSION(1), PATTERN_MATCHING(2),
            SEQUENCE(3), ASSOCIATION(4);
            private final int value;
            Task(int value) { this.value = value; }
            public int getValue() { return value; }
        }

        private native long nativeCreate(String name, int size, int task, int numInputs, int numOutputs);
        private native void nativeDestroy(long handle);
        private native int nativeLearn(long handle, float[] features, String label, float confidence);
        private native String[] nativePredict(long handle, float[] features); // Returns [label, confidence]
        private native int nativeSave(long handle, String filepath);
        private static native long nativeLoad(String filepath);

        public Brain(String name, Size size, Task task, int numInputs, int numOutputs) throws NIMCPException {
            this.handle = nativeCreate(name, size.getValue(), task.getValue(), numInputs, numOutputs);
            if (this.handle == 0) {
                throw new NIMCPException(nativeGetError());
            }
        }

        public void learn(float[] features, String label, float confidence) throws NIMCPException {
            if (nativeLearn(handle, features, label, confidence) != 0) {
                throw new NIMCPException(nativeGetError());
            }
        }

        public Prediction predict(float[] features) throws NIMCPException {
            String[] result = nativePredict(handle, features);
            if (result == null) {
                throw new NIMCPException(nativeGetError());
            }
            return new Prediction(result[0], Float.parseFloat(result[1]));
        }

        public void save(String filepath) throws NIMCPException {
            if (nativeSave(handle, filepath) != 0) {
                throw new NIMCPException(nativeGetError());
            }
        }

        public static Brain load(String filepath) throws NIMCPException {
            Brain brain = new Brain();
            brain.handle = nativeLoad(filepath);
            if (brain.handle == 0) {
                throw new NIMCPException(nativeGetError());
            }
            return brain;
        }

        private Brain() {} // For load()

        @Override
        protected void finalize() throws Throwable {
            if (handle != 0) {
                nativeDestroy(handle);
            }
            super.finalize();
        }
    }

    // Network class
    public static class Network {
        private long handle;
        private int numOutputs;

        private native long nativeCreate(int numInputs, int numOutputs, int numHidden, float learningRate);
        private native void nativeDestroy(long handle);
        private native float[] nativeForward(long handle, float[] inputs);

        public Network(int numInputs, int numOutputs, int numHidden, float learningRate) throws NIMCPException {
            this.handle = nativeCreate(numInputs, numOutputs, numHidden, learningRate);
            this.numOutputs = numOutputs;
            if (this.handle == 0) {
                throw new NIMCPException(nativeGetError());
            }
        }

        public float[] forward(float[] inputs) throws NIMCPException {
            float[] outputs = nativeForward(handle, inputs);
            if (outputs == null) {
                throw new NIMCPException(nativeGetError());
            }
            return outputs;
        }

        @Override
        protected void finalize() throws Throwable {
            if (handle != 0) {
                nativeDestroy(handle);
            }
            super.finalize();
        }
    }

    // Helper classes
    public static class Prediction {
        public final String label;
        public final float confidence;

        public Prediction(String label, float confidence) {
            this.label = label;
            this.confidence = confidence;
        }
    }

    public static class NIMCPException extends Exception {
        public NIMCPException(String message) {
            super(message);
        }
    }
}
