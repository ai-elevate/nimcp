/**
 * C# bindings for NIMCP using P/Invoke
 * Wraps the unified nimcp.h C API
 */

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace NIMCP
{
    public enum BrainSize
    {
        Tiny = 0,
        Small = 1,
        Medium = 2,
        Large = 3
    }

    public enum BrainTask
    {
        Classification = 0,
        Regression = 1,
        PatternMatching = 2,
        Sequence = 3,
        Association = 4
    }

    public enum Status
    {
        Ok = 0,
        Error = -1,
        ErrorNullArg = -2,
        ErrorInvalid = -3,
        ErrorMemory = -4,
        ErrorIO = -5
    }

    // Native methods
    internal static class Native
    {
        const string DllName = "nimcp_core";

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_init();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nimcp_shutdown();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_version();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_get_error();

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_brain_create(
            [MarshalAs(UnmanagedType.LPStr)] string name,
            int size, int task, uint numInputs, uint numOutputs);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nimcp_brain_destroy(IntPtr brain);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_learn_example(
            IntPtr brain, float[] features, uint numFeatures,
            [MarshalAs(UnmanagedType.LPStr)] string label, float confidence);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_predict(
            IntPtr brain, float[] features, uint numFeatures,
            StringBuilder outLabel, ref float outConfidence);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_save(IntPtr brain,
            [MarshalAs(UnmanagedType.LPStr)] string filepath);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_brain_load(
            [MarshalAs(UnmanagedType.LPStr)] string filepath);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_network_create(
            uint numInputs, uint numOutputs, uint numHidden, float learningRate);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nimcp_network_destroy(IntPtr network);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_network_forward(
            IntPtr network, float[] inputs, uint numInputs,
            float[] outputs, uint numOutputs);
    }

    // Main class
    public static class Library
    {
        public static void Init()
        {
            if (Native.nimcp_init() != 0)
            {
                throw new NIMCPException("Failed to initialize NIMCP");
            }
        }

        public static void Shutdown()
        {
            Native.nimcp_shutdown();
        }

        public static string Version()
        {
            return Marshal.PtrToStringAnsi(Native.nimcp_version());
        }

        internal static string GetError()
        {
            return Marshal.PtrToStringAnsi(Native.nimcp_get_error());
        }
    }

    // Brain class
    public class Brain : IDisposable
    {
        private IntPtr handle;
        private bool disposed = false;

        public Brain(string name, BrainSize size, BrainTask task, uint numInputs, uint numOutputs)
        {
            handle = Native.nimcp_brain_create(name, (int)size, (int)task, numInputs, numOutputs);
            if (handle == IntPtr.Zero)
            {
                throw new NIMCPException(Library.GetError());
            }
        }

        public void Learn(float[] features, string label, float confidence = 1.0f)
        {
            if (disposed) throw new ObjectDisposedException(nameof(Brain));

            int status = Native.nimcp_brain_learn_example(
                handle, features, (uint)features.Length, label, confidence);

            if (status != 0)
            {
                throw new NIMCPException(Library.GetError());
            }
        }

        public (string Label, float Confidence) Predict(float[] features)
        {
            if (disposed) throw new ObjectDisposedException(nameof(Brain));

            StringBuilder label = new StringBuilder(64);
            float confidence = 0.0f;

            int status = Native.nimcp_brain_predict(
                handle, features, (uint)features.Length, label, ref confidence);

            if (status != 0)
            {
                throw new NIMCPException(Library.GetError());
            }

            return (label.ToString(), confidence);
        }

        public void Save(string filepath)
        {
            if (disposed) throw new ObjectDisposedException(nameof(Brain));

            int status = Native.nimcp_brain_save(handle, filepath);
            if (status != 0)
            {
                throw new NIMCPException(Library.GetError());
            }
        }

        public static Brain Load(string filepath)
        {
            IntPtr handle = Native.nimcp_brain_load(filepath);
            if (handle == IntPtr.Zero)
            {
                throw new NIMCPException(Library.GetError());
            }

            return new Brain(handle);
        }

        private Brain(IntPtr handle) { this.handle = handle; }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (handle != IntPtr.Zero)
                {
                    Native.nimcp_brain_destroy(handle);
                    handle = IntPtr.Zero;
                }
                disposed = true;
            }
        }

        ~Brain()
        {
            Dispose(false);
        }
    }

    // Network class
    public class Network : IDisposable
    {
        private IntPtr handle;
        private uint numOutputs;
        private bool disposed = false;

        public Network(uint numInputs, uint numOutputs, uint numHidden, float learningRate = 0.01f)
        {
            handle = Native.nimcp_network_create(numInputs, numOutputs, numHidden, learningRate);
            this.numOutputs = numOutputs;

            if (handle == IntPtr.Zero)
            {
                throw new NIMCPException(Library.GetError());
            }
        }

        public float[] Forward(float[] inputs)
        {
            if (disposed) throw new ObjectDisposedException(nameof(Network));

            float[] outputs = new float[numOutputs];

            int status = Native.nimcp_network_forward(
                handle, inputs, (uint)inputs.Length, outputs, numOutputs);

            if (status != 0)
            {
                throw new NIMCPException(Library.GetError());
            }

            return outputs;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (handle != IntPtr.Zero)
                {
                    Native.nimcp_network_destroy(handle);
                    handle = IntPtr.Zero;
                }
                disposed = true;
            }
        }

        ~Network()
        {
            Dispose(false);
        }
    }

    // Exception class
    public class NIMCPException : Exception
    {
        public NIMCPException(string message) : base(message) { }
    }
}
