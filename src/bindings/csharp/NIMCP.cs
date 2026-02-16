/**
 * @file NIMCP.cs
 * @brief Complete C# bindings for NIMCP via P/Invoke
 * @version 2.6.3
 *
 * Wraps the entire nimcp.h public C API with idiomatic C# classes.
 * Uses IDisposable for deterministic resource cleanup,
 * typed exceptions, enums, and delegates for callbacks.
 *
 * Usage:
 *   NimcpLibrary.Init();
 *   using (var brain = new Brain("test", BrainSize.Tiny,
 *              BrainTask.Classification, 4, 2))
 *   {
 *       brain.Learn(new float[]{1,0,0.5f,0.3f}, "cat", 0.9f);
 *       var (label, conf) = brain.Predict(new float[]{1,0,0.5f,0.3f});
 *   }
 *   NimcpLibrary.Shutdown();
 */

using System;
using System.Runtime.InteropServices;
using System.Text;

namespace NIMCP
{
    // ========================================================================
    // Enums
    // ========================================================================

    public enum BrainSize { Tiny = 0, Small = 1, Medium = 2, Large = 3 }

    public enum BrainTask
    {
        Classification = 0, Regression = 1, PatternMatching = 2,
        Sequence = 3, Association = 4
    }

    public enum NetworkType
    {
        Adaptive = 0, SNN = 1, LNN = 2, CNN = 3, Hybrid = 4
    }

    public enum SNNTrainMethod
    {
        STDP = 0, RSTDP = 1, EProp = 2, Surrogate = 3, Homeostatic = 4
    }

    public enum LNNTrainMethod
    {
        Adjoint = 0, BPTT = 1, RTRL = 2, EProp = 3
    }

    public enum LossType
    {
        MSE = 0, CrossEntropy = 1, BinaryCE = 2, Huber = 3,
        MAE = 4, Focal = 5, KLDiv = 6
    }

    public enum OptimizerType
    {
        SGD = 0, Momentum = 1, Adam = 2, AdamW = 3, RMSprop = 4, Adagrad = 5
    }

    public enum SchedulerType
    {
        Constant = 0, Step = 1, Exponential = 2, Cosine = 3,
        WarmupCosine = 4, ReduceOnPlateau = 5, Cyclic = 6
    }

    public enum CallbackEvent
    {
        StepComplete = 0, EpochComplete = 1, LossComputed = 2,
        WeightsUpdated = 3, LRChanged = 4, Convergence = 5,
        Divergence = 6, Checkpoint = 7
    }

    public enum CallbackAction
    {
        Continue = 0, Stop = 1, Skip = 2, Rollback = 3,
        ReduceLR = 4, IncreaseLR = 5
    }

    public enum CognitiveModule
    {
        None = 0, Perception = 1, WorkingMemory = 2, Executive = 3,
        TheoryOfMind = 4, Ethics = 5, Attention = 6, Emotion = 7,
        Salience = 8, Motor = 9, Language = 10, Metacognition = 11,
        Curiosity = 12, Introspection = 13, Predictive = 14,
        Consolidation = 15, EpisodicMemory = 16, SemanticMemory = 17,
        Wellbeing = 18, MentalHealth = 19, GoalMotivation = 20,
        CognitiveControl = 21, CustomStart = 100
    }

    public enum StatusCode
    {
        Ok = 0, Error = 1000, ErrorNullArg = 1003,
        ErrorInvalid = 1004, ErrorMemory = 2000, ErrorIO = 4000
    }

    // ========================================================================
    // Exception Hierarchy
    // ========================================================================

    public class NIMCPException : Exception
    {
        public StatusCode Code { get; }
        public NIMCPException(string message) : base(message) { Code = StatusCode.Error; }
        public NIMCPException(StatusCode code, string message) : base(message) { Code = code; }
    }

    public class NullArgException : NIMCPException
    {
        public NullArgException(string msg) : base(StatusCode.ErrorNullArg, msg) {}
    }

    public class InvalidException : NIMCPException
    {
        public InvalidException(string msg) : base(StatusCode.ErrorInvalid, msg) {}
    }

    public class MemoryException : NIMCPException
    {
        public MemoryException(string msg) : base(StatusCode.ErrorMemory, msg) {}
    }

    public class IOException : NIMCPException
    {
        public IOException(string msg) : base(StatusCode.ErrorIO, msg) {}
    }

    // ========================================================================
    // Data Structures
    // ========================================================================

    public class Prediction
    {
        public string Label { get; }
        public float Confidence { get; }
        public Prediction(string label, float confidence)
        { Label = label; Confidence = confidence; }
    }

    public class TrainingConfig
    {
        public LossType LossType = LossType.CrossEntropy;
        public OptimizerType OptimizerType = OptimizerType.Adam;
        public SchedulerType SchedulerType = SchedulerType.Cosine;
        public float LearningRate = 0.001f;
        public float WeightDecay = 0.0f;
        public float Momentum = 0.9f;
        public float Beta1 = 0.9f;
        public float Beta2 = 0.999f;
        public float Epsilon = 1e-8f;
        public uint SchedulerStepSize = 100;
        public float SchedulerGamma = 0.1f;
        public uint WarmupSteps = 0;
        public bool EnableGradientClipping = false;
        public float GradientClipValue = 1.0f;
        public bool EnableBiologicalModulation = true;
        public float BiologicalBlend = 0.5f;
        public NetworkType NetworkType = NetworkType.Adaptive;
        public SNNTrainMethod SNNMethod = SNNTrainMethod.STDP;
        public float SNNEligibilityTau = 20.0f;
        public float SNNRewardTau = 100.0f;
        public float SNNSurrogateBeta = 5.0f;
        public LNNTrainMethod LNNMethod = LNNTrainMethod.Adjoint;
        public uint LNNBpttTruncation = 100;
        public bool LNNUseAdjointCheckpointing = true;
    }

    public class TrainingResult
    {
        public float Loss;
        public float LearningRate;
        public uint Step;
        public bool EarlyStopped;
        public float GradientNorm;
    }

    public class CallbackConfig
    {
        public bool EnableAutoCheckpoint = false;
        public uint CheckpointInterval = 100;
        public bool EnableEarlyStopping = false;
        public uint Patience = 10;
        public float MinDelta = 0.0001f;
        public float DivergenceThreshold = 10.0f;
        public uint LogInterval = 0;
    }

    public class CallbackMetrics
    {
        public ulong Step;
        public ulong Epoch;
        public float Loss;
        public float LossEma;
        public float LearningRate;
        public float GradientNorm;
        public ulong StepTimeUs;
        public bool IsConverging;
        public bool IsDiverging;
    }

    public class SnapshotInfo
    {
        public string Name;
        public string Description;
        public ulong Timestamp;
        public uint FileSize;
        public bool IsCompressed;
        public bool IsEncrypted;
    }

    public class BrainProbe
    {
        public string TaskName;
        public BrainSize Size;
        public BrainTask Task;
        public uint NumNeurons;
        public uint NumSynapses;
        public uint NumActiveSynapses;
        public ulong TotalInferences;
        public ulong TotalLearningSteps;
        public float AvgSparsity;
        public float AvgInferenceTimeUs;
        public float CurrentLearningRate;
        public float Accuracy;
        public ulong MemoryBytes;
        public uint NumInputs;
        public uint NumOutputs;
        public bool IsCowClone;
        public uint CowRefCount;
        public ulong CowSharedBytes;
        public ulong CowPrivateBytes;
    }

    public struct Phasor
    {
        public float Amplitude;
        public float Phase;
        public Phasor(float amplitude, float phase)
        { Amplitude = amplitude; Phase = phase; }
    }

    public class TrainingStats
    {
        public ulong TotalSteps;
        public float TotalLoss;
        public float CurrentLR;
    }

    public class UtilizationMetrics
    {
        public float Utilization;
        public float Saturation;
    }

    public class WorkspaceReadResult
    {
        public float[] Content;
        public uint ActualDim;
        public CognitiveModule SourceModule;
    }

    public class WorkspaceStats
    {
        public uint TotalBroadcasts;
        public uint TotalCompetitions;
        public float AvgStrength;
    }

    public class CallbackStats
    {
        public ulong TotalFired;
        public float AvgTimeUs;
        public uint EarlyStops;
    }

    public class WorkingMemoryStats
    {
        public uint CurrentSize;
        public uint Capacity;
    }

    // ========================================================================
    // Callback Delegate
    // ========================================================================

    public delegate CallbackAction TrainingCallbackDelegate(
        CallbackEvent evt, CallbackMetrics metrics);

    // ========================================================================
    // Native P/Invoke Declarations
    // ========================================================================

    internal static class Native
    {
        const string Lib = "nimcp";

        // --- Library lifecycle ---
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_init();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nimcp_shutdown();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_version();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_version_int();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_get_error();

        // --- Brain core ---
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_brain_create(
            [MarshalAs(UnmanagedType.LPStr)] string name,
            int size, int task, uint numInputs, uint numOutputs);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nimcp_brain_destroy(IntPtr brain);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_learn_example(
            IntPtr brain, float[] features, uint numFeatures,
            [MarshalAs(UnmanagedType.LPStr)] string label, float confidence);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_predict(
            IntPtr brain, float[] features, uint numFeatures,
            StringBuilder outLabel, ref float outConfidence);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_infer(
            IntPtr brain, float[] features, uint numFeatures,
            float[] outputs, uint numOutputs);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_save(IntPtr brain,
            [MarshalAs(UnmanagedType.LPStr)] string filepath);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_brain_load(
            [MarshalAs(UnmanagedType.LPStr)] string filepath);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_brain_create_from_config(
            [MarshalAs(UnmanagedType.LPStr)] string filepath);

        // --- Brain training ---
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_configure_training(
            IntPtr brain, ref NativeTrainingConfig config);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_train_step(
            IntPtr brain, float[] features, uint numFeatures,
            float[] targets, uint numTargets, ref NativeTrainingResult result);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_train_batch(
            IntPtr brain, float[] features, float[] targets,
            uint batchSize, uint numFeatures, uint numTargets,
            ref NativeTrainingResult result);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_get_training_stats(
            IntPtr brain, ref ulong totalSteps, ref float totalLoss,
            ref float currentLR);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern float nimcp_brain_step_scheduler(
            IntPtr brain, float validationMetric);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NativeTrainingConfig nimcp_training_config_default();

        // --- Brain callbacks ---
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_enable_callbacks(
            IntPtr brain, ref NativeCallbackConfig config);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_disable_callbacks(IntPtr brain);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate int NativeCallbackFn(int evt, IntPtr metrics, IntPtr userData);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint nimcp_brain_register_callback(
            IntPtr brain, int evt, NativeCallbackFn callback,
            IntPtr userData,
            [MarshalAs(UnmanagedType.LPStr)] string name);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_unregister_callback(
            IntPtr brain, uint callbackId);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_get_callback_stats(
            IntPtr brain, ref ulong totalFired, ref float avgTimeUs,
            ref uint earlyStops);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NativeCallbackConfig nimcp_callback_config_default();

        // --- Brain resize ---
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool nimcp_brain_resize(IntPtr brain, uint count);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool nimcp_brain_auto_resize(IntPtr brain);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint nimcp_brain_get_neuron_count(IntPtr brain);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool nimcp_brain_get_utilization_metrics(
            IntPtr brain, ref float utilization, ref float saturation);

        // --- Brain named snapshots ---
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_snapshot_save(
            IntPtr brain,
            [MarshalAs(UnmanagedType.LPStr)] string name,
            [MarshalAs(UnmanagedType.LPStr)] string description);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_brain_snapshot_restore(
            IntPtr brain,
            [MarshalAs(UnmanagedType.LPStr)] string name);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_snapshot_list(
            IntPtr brain, IntPtr infos, uint maxCount, ref uint outCount);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_snapshot_delete(
            IntPtr brain,
            [MarshalAs(UnmanagedType.LPStr)] string name);

        // --- Brain COW ---
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_brain_clone_cow(IntPtr brain);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_brain_snapshot_cow(IntPtr brain);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_restore_cow(
            IntPtr brain, IntPtr snapshot);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nimcp_brain_snapshot_destroy(IntPtr snapshot);

        // --- Brain working memory ---
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_working_memory_add(
            IntPtr brain, float[] data, uint size, float salience);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_brain_working_memory_get(
            IntPtr brain, uint index, ref uint sizeOut);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_working_memory_stats(
            IntPtr brain, ref uint currentSize, ref uint capacity);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_working_memory_refresh(
            IntPtr brain, uint index);

        // --- Brain workspace ---
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_workspace_compete(
            IntPtr brain, int module, float[] content,
            uint contentDim, float strength);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_workspace_read(
            IntPtr brain, float[] content, uint maxDim,
            ref uint actualDim, ref int sourceModule);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_workspace_subscribe(
            IntPtr brain, int module);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_workspace_unsubscribe(
            IntPtr brain, int module);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_workspace_has_broadcast(
            IntPtr brain, [MarshalAs(UnmanagedType.I1)] ref bool hasBroadcast);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_workspace_stats(
            IntPtr brain, ref uint totalBroadcasts,
            ref uint totalCompetitions, ref float avgStrength);

        // --- Brain oscillations ---
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool nimcp_enable_complex_oscillations(
            IntPtr brain, [MarshalAs(UnmanagedType.I1)] bool enable);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool nimcp_is_complex_oscillations_enabled(
            IntPtr brain);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern NativePhasor nimcp_get_oscillation_phasor(
            IntPtr brain, uint neuronId);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern float nimcp_get_phase_coherence(
            IntPtr brain, uint[] neuronIds, uint count);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern float nimcp_get_pac_modulation(
            IntPtr brain, float thetaFreq, float gammaFreq);

        // --- Brain probe ---
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_probe(
            IntPtr brain, ref NativeBrainProbe probe);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_brain_broadcast_probe(IntPtr brain);

        // --- Network ---
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_network_create(
            uint numInputs, uint numOutputs, uint numHidden, float lr);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nimcp_network_destroy(IntPtr network);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_network_forward(
            IntPtr network, float[] inputs, uint numInputs,
            float[] outputs, uint numOutputs);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_network_train(
            IntPtr network, float[] inputs, uint numInputs,
            float[] targets, uint numTargets);

        // --- Ethics ---
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_ethics_create();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nimcp_ethics_destroy(IntPtr ethics);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_ethics_check(
            IntPtr ethics, float[] situation, uint numFeatures,
            ref float outScore);

        // --- Knowledge ---
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr nimcp_knowledge_create();

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern void nimcp_knowledge_destroy(IntPtr knowledge);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_knowledge_add_fact(
            IntPtr knowledge,
            [MarshalAs(UnmanagedType.LPStr)] string subject,
            [MarshalAs(UnmanagedType.LPStr)] string predicate,
            [MarshalAs(UnmanagedType.LPStr)] string obj);

        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
        public static extern int nimcp_knowledge_query(
            IntPtr knowledge,
            [MarshalAs(UnmanagedType.LPStr)] string query,
            StringBuilder outResult, uint maxResultLen);

        // --- Native structs ---

        [StructLayout(LayoutKind.Sequential)]
        public struct NativeTrainingConfig
        {
            public int loss_type;
            public int optimizer_type;
            public int scheduler_type;
            public float learning_rate;
            public float weight_decay;
            public float momentum;
            public float beta1;
            public float beta2;
            public float epsilon;
            public uint scheduler_step_size;
            public float scheduler_gamma;
            public uint warmup_steps;
            [MarshalAs(UnmanagedType.I1)] public bool enable_gradient_clipping;
            public float gradient_clip_value;
            [MarshalAs(UnmanagedType.I1)] public bool enable_biological_modulation;
            public float biological_blend;
            public int network_type;
            public int snn_method;
            public float snn_eligibility_tau;
            public float snn_reward_tau;
            public float snn_surrogate_beta;
            public int lnn_method;
            public uint lnn_bptt_truncation;
            [MarshalAs(UnmanagedType.I1)] public bool lnn_use_adjoint_checkpointing;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct NativeTrainingResult
        {
            public float loss;
            public float learning_rate;
            public uint step;
            [MarshalAs(UnmanagedType.I1)] public bool early_stopped;
            public float gradient_norm;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct NativeCallbackConfig
        {
            [MarshalAs(UnmanagedType.I1)] public bool enable_auto_checkpoint;
            public uint checkpoint_interval;
            [MarshalAs(UnmanagedType.I1)] public bool enable_early_stopping;
            public uint patience;
            public float min_delta;
            public float divergence_threshold;
            public uint log_interval;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct NativeCallbackMetrics
        {
            public ulong step;
            public ulong epoch;
            public float loss;
            public float loss_ema;
            public float learning_rate;
            public float gradient_norm;
            public ulong step_time_us;
            [MarshalAs(UnmanagedType.I1)] public bool is_converging;
            [MarshalAs(UnmanagedType.I1)] public bool is_diverging;
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
        public struct NativeSnapshotInfo
        {
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)]
            public string name;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 512)]
            public string description;
            public ulong timestamp;
            public uint file_size;
            [MarshalAs(UnmanagedType.I1)] public bool is_compressed;
            [MarshalAs(UnmanagedType.I1)] public bool is_encrypted;
        }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Ansi)]
        public struct NativeBrainProbe
        {
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 64)]
            public string task_name;
            public int size;
            public int task;
            public uint num_neurons;
            public uint num_synapses;
            public uint num_active_synapses;
            public ulong total_inferences;
            public ulong total_learning_steps;
            public float avg_sparsity;
            public float avg_inference_time_us;
            public float current_learning_rate;
            public float accuracy;
            public ulong memory_bytes;
            public uint num_inputs;
            public uint num_outputs;
            [MarshalAs(UnmanagedType.I1)] public bool is_cow_clone;
            public uint cow_ref_count;
            public ulong cow_shared_bytes;
            public ulong cow_private_bytes;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct NativePhasor
        {
            public float amplitude;
            public float phase;
        }
    }

    // ========================================================================
    // Helper
    // ========================================================================

    internal static class Helper
    {
        public static string GetError()
        {
            IntPtr ptr = Native.nimcp_get_error();
            if (ptr == IntPtr.Zero) return "NIMCP error";
            string msg = Marshal.PtrToStringAnsi(ptr);
            return string.IsNullOrEmpty(msg) ? "NIMCP error" : msg;
        }

        public static void CheckStatus(int status)
        {
            if (status == 0) return;
            string msg = GetError();
            switch (status)
            {
                case 1003: throw new NullArgException(msg);
                case 1004: throw new InvalidException(msg);
                case 2000: throw new MemoryException(msg);
                case 4000: throw new IOException(msg);
                default:   throw new NIMCPException((StatusCode)status, msg);
            }
        }

        public static Native.NativeTrainingConfig ToNative(TrainingConfig cfg)
        {
            var n = Native.nimcp_training_config_default();
            n.loss_type = (int)cfg.LossType;
            n.optimizer_type = (int)cfg.OptimizerType;
            n.scheduler_type = (int)cfg.SchedulerType;
            n.learning_rate = cfg.LearningRate;
            n.weight_decay = cfg.WeightDecay;
            n.momentum = cfg.Momentum;
            n.beta1 = cfg.Beta1;
            n.beta2 = cfg.Beta2;
            n.epsilon = cfg.Epsilon;
            n.scheduler_step_size = cfg.SchedulerStepSize;
            n.scheduler_gamma = cfg.SchedulerGamma;
            n.warmup_steps = cfg.WarmupSteps;
            n.enable_gradient_clipping = cfg.EnableGradientClipping;
            n.gradient_clip_value = cfg.GradientClipValue;
            n.enable_biological_modulation = cfg.EnableBiologicalModulation;
            n.biological_blend = cfg.BiologicalBlend;
            n.network_type = (int)cfg.NetworkType;
            n.snn_method = (int)cfg.SNNMethod;
            n.snn_eligibility_tau = cfg.SNNEligibilityTau;
            n.snn_reward_tau = cfg.SNNRewardTau;
            n.snn_surrogate_beta = cfg.SNNSurrogateBeta;
            n.lnn_method = (int)cfg.LNNMethod;
            n.lnn_bptt_truncation = cfg.LNNBpttTruncation;
            n.lnn_use_adjoint_checkpointing = cfg.LNNUseAdjointCheckpointing;
            return n;
        }

        public static Native.NativeCallbackConfig ToNative(CallbackConfig cfg)
        {
            var n = Native.nimcp_callback_config_default();
            n.enable_auto_checkpoint = cfg.EnableAutoCheckpoint;
            n.checkpoint_interval = cfg.CheckpointInterval;
            n.enable_early_stopping = cfg.EnableEarlyStopping;
            n.patience = cfg.Patience;
            n.min_delta = cfg.MinDelta;
            n.divergence_threshold = cfg.DivergenceThreshold;
            n.log_interval = cfg.LogInterval;
            return n;
        }
    }

    // ========================================================================
    // Library Lifecycle
    // ========================================================================

    public static class NimcpLibrary
    {
        public static void Init()
        {
            Helper.CheckStatus(Native.nimcp_init());
        }

        public static void Shutdown() => Native.nimcp_shutdown();

        public static string Version()
        {
            IntPtr ptr = Native.nimcp_version();
            return ptr != IntPtr.Zero ? Marshal.PtrToStringAnsi(ptr) : "unknown";
        }

        public static int VersionInt() => Native.nimcp_version_int();
    }

    // ========================================================================
    // Brain Class
    // ========================================================================

    public class Brain : IDisposable
    {
        private IntPtr handle;
        private bool disposed;
        // prevent GC of callback delegates while registered
        private System.Collections.Generic.List<Native.NativeCallbackFn> callbackRefs
            = new System.Collections.Generic.List<Native.NativeCallbackFn>();

        public Brain(string name, BrainSize size, BrainTask task,
                     uint numInputs, uint numOutputs)
        {
            handle = Native.nimcp_brain_create(
                name, (int)size, (int)task, numInputs, numOutputs);
            if (handle == IntPtr.Zero)
                throw new NIMCPException("Failed to create brain");
        }

        private Brain(IntPtr handle) { this.handle = handle; }

        public static Brain Load(string filepath)
        {
            IntPtr h = Native.nimcp_brain_load(filepath);
            if (h == IntPtr.Zero)
                throw new IOException("Failed to load brain: " + filepath);
            return new Brain(h);
        }

        public static Brain CreateFromConfig(string filepath)
        {
            IntPtr h = Native.nimcp_brain_create_from_config(filepath);
            if (h == IntPtr.Zero)
                throw new NIMCPException("Failed to create brain from config");
            return new Brain(h);
        }

        private void CheckOpen()
        {
            if (disposed) throw new ObjectDisposedException(nameof(Brain));
        }

        // --- Core ---

        public void Learn(float[] features, string label, float confidence = 1.0f)
        {
            CheckOpen();
            Helper.CheckStatus(Native.nimcp_brain_learn_example(
                handle, features, (uint)features.Length, label, confidence));
        }

        public Prediction Predict(float[] features)
        {
            CheckOpen();
            var labelBuf = new StringBuilder(64);
            float conf = 0;
            Helper.CheckStatus(Native.nimcp_brain_predict(
                handle, features, (uint)features.Length, labelBuf, ref conf));
            return new Prediction(labelBuf.ToString(), conf);
        }

        public void Infer(float[] features, float[] outputs)
        {
            CheckOpen();
            Helper.CheckStatus(Native.nimcp_brain_infer(
                handle, features, (uint)features.Length,
                outputs, (uint)outputs.Length));
        }

        public void Save(string filepath)
        {
            CheckOpen();
            Helper.CheckStatus(Native.nimcp_brain_save(handle, filepath));
        }

        // --- Training Pipeline ---

        public void ConfigureTraining(TrainingConfig cfg)
        {
            CheckOpen();
            var nc = Helper.ToNative(cfg);
            Helper.CheckStatus(
                Native.nimcp_brain_configure_training(handle, ref nc));
        }

        public TrainingResult TrainStep(float[] features, float[] targets)
        {
            CheckOpen();
            var nr = new Native.NativeTrainingResult();
            Helper.CheckStatus(Native.nimcp_brain_train_step(
                handle, features, (uint)features.Length,
                targets, (uint)targets.Length, ref nr));
            return new TrainingResult
            {
                Loss = nr.loss, LearningRate = nr.learning_rate,
                Step = nr.step, EarlyStopped = nr.early_stopped,
                GradientNorm = nr.gradient_norm
            };
        }

        public TrainingResult TrainBatch(float[] features, float[] targets,
            uint batchSize, uint numFeatures, uint numTargets)
        {
            CheckOpen();
            var nr = new Native.NativeTrainingResult();
            Helper.CheckStatus(Native.nimcp_brain_train_batch(
                handle, features, targets,
                batchSize, numFeatures, numTargets, ref nr));
            return new TrainingResult
            {
                Loss = nr.loss, LearningRate = nr.learning_rate,
                Step = nr.step, EarlyStopped = nr.early_stopped,
                GradientNorm = nr.gradient_norm
            };
        }

        public TrainingStats GetTrainingStats()
        {
            CheckOpen();
            ulong steps = 0; float loss = 0, lr = 0;
            Helper.CheckStatus(Native.nimcp_brain_get_training_stats(
                handle, ref steps, ref loss, ref lr));
            return new TrainingStats
            { TotalSteps = steps, TotalLoss = loss, CurrentLR = lr };
        }

        public float StepScheduler(float validationMetric)
        {
            CheckOpen();
            return Native.nimcp_brain_step_scheduler(handle, validationMetric);
        }

        // --- Callbacks ---

        public void EnableCallbacks(CallbackConfig cfg)
        {
            CheckOpen();
            var nc = Helper.ToNative(cfg);
            Helper.CheckStatus(
                Native.nimcp_brain_enable_callbacks(handle, ref nc));
        }

        public void DisableCallbacks()
        {
            CheckOpen();
            Helper.CheckStatus(Native.nimcp_brain_disable_callbacks(handle));
        }

        public uint RegisterCallback(CallbackEvent evt,
            TrainingCallbackDelegate callback, string name = null)
        {
            CheckOpen();
            Native.NativeCallbackFn native = (int e, IntPtr metricsPtr, IntPtr ud) =>
            {
                var metrics = new CallbackMetrics();
                if (metricsPtr != IntPtr.Zero)
                {
                    var nm = Marshal.PtrToStructure<Native.NativeCallbackMetrics>(metricsPtr);
                    metrics.Step = nm.step; metrics.Epoch = nm.epoch;
                    metrics.Loss = nm.loss; metrics.LossEma = nm.loss_ema;
                    metrics.LearningRate = nm.learning_rate;
                    metrics.GradientNorm = nm.gradient_norm;
                    metrics.StepTimeUs = nm.step_time_us;
                    metrics.IsConverging = nm.is_converging;
                    metrics.IsDiverging = nm.is_diverging;
                }
                var action = callback((CallbackEvent)e, metrics);
                return (int)action;
            };
            callbackRefs.Add(native); // prevent GC

            uint id = Native.nimcp_brain_register_callback(
                handle, (int)evt, native, IntPtr.Zero, name);
            if (id == 0) throw new NIMCPException("Failed to register callback");
            return id;
        }

        public void UnregisterCallback(uint callbackId)
        {
            CheckOpen();
            Helper.CheckStatus(
                Native.nimcp_brain_unregister_callback(handle, callbackId));
        }

        public CallbackStats GetCallbackStats()
        {
            CheckOpen();
            ulong fired = 0; float avg = 0; uint early = 0;
            Helper.CheckStatus(Native.nimcp_brain_get_callback_stats(
                handle, ref fired, ref avg, ref early));
            return new CallbackStats
            { TotalFired = fired, AvgTimeUs = avg, EarlyStops = early };
        }

        // --- Resize ---

        public bool Resize(uint neuronCount)
        {
            CheckOpen();
            return Native.nimcp_brain_resize(handle, neuronCount);
        }

        public bool AutoResize()
        {
            CheckOpen();
            return Native.nimcp_brain_auto_resize(handle);
        }

        public uint GetNeuronCount()
        {
            CheckOpen();
            return Native.nimcp_brain_get_neuron_count(handle);
        }

        public UtilizationMetrics GetUtilizationMetrics()
        {
            CheckOpen();
            float util = 0, sat = 0;
            Native.nimcp_brain_get_utilization_metrics(
                handle, ref util, ref sat);
            return new UtilizationMetrics
            { Utilization = util, Saturation = sat };
        }

        // --- Named Snapshots ---

        public void SnapshotSave(string name, string description = null)
        {
            CheckOpen();
            Helper.CheckStatus(
                Native.nimcp_brain_snapshot_save(handle, name, description));
        }

        public Brain SnapshotRestore(string name)
        {
            CheckOpen();
            IntPtr h = Native.nimcp_brain_snapshot_restore(handle, name);
            if (h == IntPtr.Zero)
                throw new NIMCPException("Failed to restore snapshot");
            return new Brain(h);
        }

        public void SnapshotDelete(string name)
        {
            CheckOpen();
            Helper.CheckStatus(
                Native.nimcp_brain_snapshot_delete(handle, name));
        }

        // --- COW ---

        public Brain CloneCow()
        {
            CheckOpen();
            IntPtr h = Native.nimcp_brain_clone_cow(handle);
            if (h == IntPtr.Zero)
                throw new NIMCPException("Failed to clone brain");
            return new Brain(h);
        }

        public BrainSnapshot SnapshotCow()
        {
            CheckOpen();
            IntPtr h = Native.nimcp_brain_snapshot_cow(handle);
            if (h == IntPtr.Zero)
                throw new NIMCPException("Failed to create snapshot");
            return new BrainSnapshot(h);
        }

        public void RestoreCow(BrainSnapshot snapshot)
        {
            CheckOpen();
            Helper.CheckStatus(
                Native.nimcp_brain_restore_cow(handle, snapshot.Handle));
        }

        // --- Working Memory ---

        public void WorkingMemoryAdd(float[] data, float salience)
        {
            CheckOpen();
            Helper.CheckStatus(Native.nimcp_brain_working_memory_add(
                handle, data, (uint)data.Length, salience));
        }

        public float[] WorkingMemoryGet(uint index)
        {
            CheckOpen();
            uint size = 0;
            IntPtr ptr = Native.nimcp_brain_working_memory_get(
                handle, index, ref size);
            if (ptr == IntPtr.Zero || size == 0) return null;
            float[] result = new float[size];
            Marshal.Copy(ptr, result, 0, (int)size);
            return result;
        }

        public WorkingMemoryStats GetWorkingMemoryStats()
        {
            CheckOpen();
            uint cur = 0, cap = 0;
            Helper.CheckStatus(Native.nimcp_brain_working_memory_stats(
                handle, ref cur, ref cap));
            return new WorkingMemoryStats
            { CurrentSize = cur, Capacity = cap };
        }

        public void WorkingMemoryRefresh(uint index)
        {
            CheckOpen();
            Helper.CheckStatus(
                Native.nimcp_brain_working_memory_refresh(handle, index));
        }

        // --- Workspace ---

        public void WorkspaceCompete(CognitiveModule module,
            float[] content, float strength)
        {
            CheckOpen();
            Helper.CheckStatus(Native.nimcp_brain_workspace_compete(
                handle, (int)module, content,
                (uint)content.Length, strength));
        }

        public WorkspaceReadResult WorkspaceRead(uint maxDim)
        {
            CheckOpen();
            float[] buf = new float[maxDim];
            uint actualDim = 0; int source = 0;
            Helper.CheckStatus(Native.nimcp_brain_workspace_read(
                handle, buf, maxDim, ref actualDim, ref source));
            float[] content = new float[actualDim];
            Array.Copy(buf, content, actualDim);
            return new WorkspaceReadResult
            {
                Content = content, ActualDim = actualDim,
                SourceModule = (CognitiveModule)source
            };
        }

        public void WorkspaceSubscribe(CognitiveModule module)
        {
            CheckOpen();
            Helper.CheckStatus(Native.nimcp_brain_workspace_subscribe(
                handle, (int)module));
        }

        public void WorkspaceUnsubscribe(CognitiveModule module)
        {
            CheckOpen();
            Helper.CheckStatus(Native.nimcp_brain_workspace_unsubscribe(
                handle, (int)module));
        }

        public bool WorkspaceHasBroadcast()
        {
            CheckOpen();
            bool has = false;
            Helper.CheckStatus(Native.nimcp_brain_workspace_has_broadcast(
                handle, ref has));
            return has;
        }

        public WorkspaceStats GetWorkspaceStats()
        {
            CheckOpen();
            uint bc = 0, comp = 0; float avg = 0;
            Helper.CheckStatus(Native.nimcp_brain_workspace_stats(
                handle, ref bc, ref comp, ref avg));
            return new WorkspaceStats
            {
                TotalBroadcasts = bc, TotalCompetitions = comp,
                AvgStrength = avg
            };
        }

        // --- Oscillations ---

        public bool EnableOscillations(bool enable)
        {
            CheckOpen();
            return Native.nimcp_enable_complex_oscillations(handle, enable);
        }

        public bool IsOscillationsEnabled()
        {
            CheckOpen();
            return Native.nimcp_is_complex_oscillations_enabled(handle);
        }

        public Phasor GetPhasor(uint neuronId)
        {
            CheckOpen();
            var np = Native.nimcp_get_oscillation_phasor(handle, neuronId);
            return new Phasor(np.amplitude, np.phase);
        }

        public float GetPhaseCoherence(uint[] neuronIds)
        {
            CheckOpen();
            return Native.nimcp_get_phase_coherence(
                handle, neuronIds, (uint)neuronIds.Length);
        }

        public float GetPacModulation(float thetaFreq, float gammaFreq)
        {
            CheckOpen();
            return Native.nimcp_get_pac_modulation(
                handle, thetaFreq, gammaFreq);
        }

        // --- Probe ---

        public BrainProbe Probe()
        {
            CheckOpen();
            var np = new Native.NativeBrainProbe();
            Helper.CheckStatus(Native.nimcp_brain_probe(handle, ref np));
            return new BrainProbe
            {
                TaskName = np.task_name, Size = (BrainSize)np.size,
                Task = (BrainTask)np.task, NumNeurons = np.num_neurons,
                NumSynapses = np.num_synapses,
                NumActiveSynapses = np.num_active_synapses,
                TotalInferences = np.total_inferences,
                TotalLearningSteps = np.total_learning_steps,
                AvgSparsity = np.avg_sparsity,
                AvgInferenceTimeUs = np.avg_inference_time_us,
                CurrentLearningRate = np.current_learning_rate,
                Accuracy = np.accuracy, MemoryBytes = np.memory_bytes,
                NumInputs = np.num_inputs, NumOutputs = np.num_outputs,
                IsCowClone = np.is_cow_clone, CowRefCount = np.cow_ref_count,
                CowSharedBytes = np.cow_shared_bytes,
                CowPrivateBytes = np.cow_private_bytes
            };
        }

        public void BroadcastProbe()
        {
            CheckOpen();
            Helper.CheckStatus(Native.nimcp_brain_broadcast_probe(handle));
        }

        // --- IDisposable ---

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

        ~Brain() { Dispose(false); }
    }

    // ========================================================================
    // BrainSnapshot (COW)
    // ========================================================================

    public class BrainSnapshot : IDisposable
    {
        internal IntPtr Handle { get; private set; }
        private bool disposed;

        internal BrainSnapshot(IntPtr handle) { Handle = handle; }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (Handle != IntPtr.Zero)
                {
                    Native.nimcp_brain_snapshot_destroy(Handle);
                    Handle = IntPtr.Zero;
                }
                disposed = true;
            }
        }

        ~BrainSnapshot() { Dispose(false); }
    }

    // ========================================================================
    // Network Class
    // ========================================================================

    public class Network : IDisposable
    {
        private IntPtr handle;
        private bool disposed;

        public Network(uint numInputs, uint numOutputs,
                       uint numHidden, float learningRate = 0.01f)
        {
            handle = Native.nimcp_network_create(
                numInputs, numOutputs, numHidden, learningRate);
            if (handle == IntPtr.Zero)
                throw new NIMCPException("Failed to create network");
        }

        public void Forward(float[] inputs, float[] outputs)
        {
            if (disposed) throw new ObjectDisposedException(nameof(Network));
            Helper.CheckStatus(Native.nimcp_network_forward(
                handle, inputs, (uint)inputs.Length,
                outputs, (uint)outputs.Length));
        }

        public void Train(float[] inputs, float[] targets)
        {
            if (disposed) throw new ObjectDisposedException(nameof(Network));
            Helper.CheckStatus(Native.nimcp_network_train(
                handle, inputs, (uint)inputs.Length,
                targets, (uint)targets.Length));
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

        ~Network() { Dispose(false); }
    }

    // ========================================================================
    // Ethics Class
    // ========================================================================

    public class Ethics : IDisposable
    {
        private IntPtr handle;
        private bool disposed;

        public Ethics()
        {
            handle = Native.nimcp_ethics_create();
            if (handle == IntPtr.Zero)
                throw new NIMCPException("Failed to create ethics module");
        }

        public float Check(float[] situation)
        {
            if (disposed) throw new ObjectDisposedException(nameof(Ethics));
            float score = 0;
            Helper.CheckStatus(Native.nimcp_ethics_check(
                handle, situation, (uint)situation.Length, ref score));
            return score;
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
                    Native.nimcp_ethics_destroy(handle);
                    handle = IntPtr.Zero;
                }
                disposed = true;
            }
        }

        ~Ethics() { Dispose(false); }
    }

    // ========================================================================
    // KnowledgeGraph Class
    // ========================================================================

    public class KnowledgeGraph : IDisposable
    {
        private IntPtr handle;
        private bool disposed;

        public KnowledgeGraph()
        {
            handle = Native.nimcp_knowledge_create();
            if (handle == IntPtr.Zero)
                throw new NIMCPException("Failed to create knowledge graph");
        }

        public void AddFact(string subject, string predicate, string obj)
        {
            if (disposed) throw new ObjectDisposedException(nameof(KnowledgeGraph));
            Helper.CheckStatus(Native.nimcp_knowledge_add_fact(
                handle, subject, predicate, obj));
        }

        public string Query(string query, uint maxLen = 1024)
        {
            if (disposed) throw new ObjectDisposedException(nameof(KnowledgeGraph));
            var buf = new StringBuilder((int)maxLen);
            Helper.CheckStatus(Native.nimcp_knowledge_query(
                handle, query, buf, maxLen));
            return buf.ToString();
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
                    Native.nimcp_knowledge_destroy(handle);
                    handle = IntPtr.Zero;
                }
                disposed = true;
            }
        }

        ~KnowledgeGraph() { Dispose(false); }
    }
}
