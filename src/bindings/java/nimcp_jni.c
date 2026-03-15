/**
 * @file nimcp_jni.c
 * @brief JNI bridge implementation for NIMCP Java bindings
 * @version 2.6.3
 *
 * Maps all Java native methods in com.nimcp.NIMCP to the NIMCP C API.
 * Uses JNI conventions: Java_com_nimcp_NIMCP_00024ClassName_methodName
 * where 00024 encodes the '$' separator for inner classes.
 */

#include <jni.h>
#include <nimcp.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

// ============================================================================
// Helper Functions
// ============================================================================

static void throw_nimcp_exception(JNIEnv *env, int code, const char *msg) {
    const char *class_name;
    switch (code) {
        case 1003: class_name = "com/nimcp/NIMCP$NullArgException"; break;
        case 1004: class_name = "com/nimcp/NIMCP$InvalidException"; break;
        case 2000: class_name = "com/nimcp/NIMCP$MemoryException"; break;
        case 4000: class_name = "com/nimcp/NIMCP$IOExceptionNIMCP"; break;
        default:   class_name = "com/nimcp/NIMCP$NIMCPException"; break;
    }
    jclass cls = (*env)->FindClass(env, class_name);
    if (cls) {
        (*env)->ThrowNew(env, cls, msg ? msg : "NIMCP error");
    }
}

static int check_status(JNIEnv *env, nimcp_status_t s) {
    if (s == NIMCP_OK) return 0;
    const char *msg = nimcp_get_error();
    throw_nimcp_exception(env, (int)s, msg && msg[0] ? msg : "NIMCP error");
    return -1;
}

// ============================================================================
// Library Lifecycle
// ============================================================================

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_nativeInit(JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    return (jint)nimcp_init();
}

JNIEXPORT void JNICALL Java_com_nimcp_NIMCP_nativeShutdown(JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    nimcp_shutdown();
}

JNIEXPORT jstring JNICALL Java_com_nimcp_NIMCP_nativeVersion(JNIEnv *env, jclass cls) {
    (void)cls;
    const char *v = nimcp_version();
    return (*env)->NewStringUTF(env, v ? v : "unknown");
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_nativeVersionInt(JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    return (jint)nimcp_version_int();
}

// ============================================================================
// Brain - Core
// ============================================================================

JNIEXPORT jlong JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeCreate(
    JNIEnv *env, jclass cls, jstring name, jint size, jint task,
    jint numInputs, jint numOutputs)
{
    (void)cls;
    const char *c_name = (*env)->GetStringUTFChars(env, name, NULL);
    if (!c_name) return 0; /* OOM — JVM already threw OutOfMemoryError */
    nimcp_brain_t brain = nimcp_brain_create(
        c_name, (nimcp_brain_size_t)size, (nimcp_brain_task_t)task,
        (uint32_t)numInputs, (uint32_t)numOutputs);
    (*env)->ReleaseStringUTFChars(env, name, c_name);
    return (jlong)(uintptr_t)brain;
}

JNIEXPORT void JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeDestroy(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)env; (void)cls;
    if (h) nimcp_brain_destroy((nimcp_brain_t)(uintptr_t)h);
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeLearn(
    JNIEnv *env, jclass cls, jlong h, jfloatArray features,
    jstring label, jfloat confidence)
{
    (void)cls;
    nimcp_brain_t brain = (nimcp_brain_t)(uintptr_t)h;
    jint len = (*env)->GetArrayLength(env, features);
    jfloat *feats = (*env)->GetFloatArrayElements(env, features, NULL);
    if (!feats) return (jint)NIMCP_OK; /* OOM — JVM already threw */
    const char *c_label = (*env)->GetStringUTFChars(env, label, NULL);
    if (!c_label) {
        (*env)->ReleaseFloatArrayElements(env, features, feats, JNI_ABORT);
        return (jint)NIMCP_OK; /* OOM — JVM already threw */
    }

    nimcp_status_t rc = nimcp_brain_learn_example(
        brain, feats, (uint32_t)len, c_label, confidence);

    (*env)->ReleaseStringUTFChars(env, label, c_label);
    (*env)->ReleaseFloatArrayElements(env, features, feats, JNI_ABORT);
    return (jint)rc;
}

JNIEXPORT jstring JNICALL Java_com_nimcp_NIMCP_00024Brain_nativePredict(
    JNIEnv *env, jclass cls, jlong h, jfloatArray features,
    jfloatArray outConf)
{
    (void)cls;
    nimcp_brain_t brain = (nimcp_brain_t)(uintptr_t)h;
    jint len = (*env)->GetArrayLength(env, features);
    jfloat *feats = (*env)->GetFloatArrayElements(env, features, NULL);
    if (!feats) return NULL; /* OOM — JVM already threw */

    char label_buf[NIMCP_MAX_LABEL_SIZE];
    float conf = 0.0f;
    nimcp_status_t rc = nimcp_brain_predict(
        brain, feats, (uint32_t)len, label_buf, &conf);

    (*env)->ReleaseFloatArrayElements(env, features, feats, JNI_ABORT);

    if (rc != NIMCP_OK) return NULL;

    jfloat c = conf;
    (*env)->SetFloatArrayRegion(env, outConf, 0, 1, &c);
    return (*env)->NewStringUTF(env, label_buf);
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeInfer(
    JNIEnv *env, jclass cls, jlong h, jfloatArray features,
    jfloatArray outputs)
{
    (void)cls;
    nimcp_brain_t brain = (nimcp_brain_t)(uintptr_t)h;
    jint flen = (*env)->GetArrayLength(env, features);
    jint olen = (*env)->GetArrayLength(env, outputs);
    jfloat *feats = (*env)->GetFloatArrayElements(env, features, NULL);
    if (!feats) return (jint)NIMCP_OK; /* OOM — JVM already threw */
    jfloat *outs = (*env)->GetFloatArrayElements(env, outputs, NULL);
    if (!outs) {
        (*env)->ReleaseFloatArrayElements(env, features, feats, JNI_ABORT);
        return (jint)NIMCP_OK; /* OOM — JVM already threw */
    }

    nimcp_status_t rc = nimcp_brain_infer(
        brain, feats, (uint32_t)flen, outs, (uint32_t)olen);

    (*env)->ReleaseFloatArrayElements(env, features, feats, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, outputs, outs, 0); // copy back
    return (jint)rc;
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeSave(
    JNIEnv *env, jclass cls, jlong h, jstring path)
{
    (void)cls;
    const char *c_path = (*env)->GetStringUTFChars(env, path, NULL);
    if (!c_path) return (jint)NIMCP_OK; /* OOM — JVM already threw */
    nimcp_status_t rc = nimcp_brain_save(
        (nimcp_brain_t)(uintptr_t)h, c_path);
    (*env)->ReleaseStringUTFChars(env, path, c_path);
    return (jint)rc;
}

JNIEXPORT jlong JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeLoad(
    JNIEnv *env, jclass cls, jstring path)
{
    (void)cls;
    const char *c_path = (*env)->GetStringUTFChars(env, path, NULL);
    if (!c_path) return 0; /* OOM — JVM already threw */
    nimcp_brain_t brain = nimcp_brain_load(c_path);
    (*env)->ReleaseStringUTFChars(env, path, c_path);
    return (jlong)(uintptr_t)brain;
}

JNIEXPORT jlong JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeCreateFromConfig(
    JNIEnv *env, jclass cls, jstring path)
{
    (void)cls;
    const char *c_path = (*env)->GetStringUTFChars(env, path, NULL);
    if (!c_path) return 0; /* OOM — JVM already threw */
    nimcp_brain_t brain = nimcp_brain_create_from_config(c_path);
    (*env)->ReleaseStringUTFChars(env, path, c_path);
    return (jlong)(uintptr_t)brain;
}

// ============================================================================
// Brain - Training Pipeline
// ============================================================================

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeConfigureTraining(
    JNIEnv *env, jclass cls, jlong h,
    jint loss, jint opt, jint sched,
    jfloat lr, jfloat wd, jfloat mom, jfloat b1, jfloat b2, jfloat eps,
    jint schedStep, jfloat schedGamma, jint warmup,
    jboolean gradClip, jfloat gradClipVal,
    jboolean bioMod, jfloat bioBlend,
    jint netType, jint snnMethod, jfloat snnEligTau, jfloat snnRewTau,
    jfloat snnSurrBeta, jint lnnMethod, jint lnnBptt, jboolean lnnAdjCheck)
{
    (void)cls;
    nimcp_training_config_t cfg = nimcp_training_config_default();
    cfg.loss_type      = (nimcp_api_loss_t)loss;
    cfg.optimizer_type = (nimcp_api_optimizer_t)opt;
    cfg.scheduler_type = (nimcp_api_scheduler_t)sched;
    cfg.learning_rate  = lr;
    cfg.weight_decay   = wd;
    cfg.momentum       = mom;
    cfg.beta1          = b1;
    cfg.beta2          = b2;
    cfg.epsilon        = eps;
    cfg.scheduler_step_size = (uint32_t)schedStep;
    cfg.scheduler_gamma     = schedGamma;
    cfg.warmup_steps        = (uint32_t)warmup;
    cfg.enable_gradient_clipping  = gradClip;
    cfg.gradient_clip_value       = gradClipVal;
    cfg.enable_biological_modulation = bioMod;
    cfg.biological_blend             = bioBlend;
    cfg.network_type    = (nimcp_network_type_t)netType;
    cfg.snn_method      = (nimcp_snn_train_method_t)snnMethod;
    cfg.snn_eligibility_tau = snnEligTau;
    cfg.snn_reward_tau      = snnRewTau;
    cfg.snn_surrogate_beta  = snnSurrBeta;
    cfg.lnn_method          = (nimcp_lnn_train_method_t)lnnMethod;
    cfg.lnn_bptt_truncation = (uint32_t)lnnBptt;
    cfg.lnn_use_adjoint_checkpointing = lnnAdjCheck;

    return (jint)nimcp_brain_configure_training(
        (nimcp_brain_t)(uintptr_t)h, &cfg);
}

JNIEXPORT jfloatArray JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeTrainStep(
    JNIEnv *env, jclass cls, jlong h, jfloatArray features,
    jfloatArray targets)
{
    (void)cls;
    nimcp_brain_t brain = (nimcp_brain_t)(uintptr_t)h;
    jint flen = (*env)->GetArrayLength(env, features);
    jint tlen = (*env)->GetArrayLength(env, targets);
    jfloat *feats = (*env)->GetFloatArrayElements(env, features, NULL);
    if (!feats) return NULL; /* OOM — JVM already threw */
    jfloat *tgts = (*env)->GetFloatArrayElements(env, targets, NULL);
    if (!tgts) {
        (*env)->ReleaseFloatArrayElements(env, features, feats, JNI_ABORT);
        return NULL; /* OOM — JVM already threw */
    }

    nimcp_training_result_t result = {0};
    nimcp_status_t rc = nimcp_brain_train_step(
        brain, feats, (uint32_t)flen, tgts, (uint32_t)tlen, &result);

    (*env)->ReleaseFloatArrayElements(env, features, feats, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, targets, tgts, JNI_ABORT);

    if (rc != NIMCP_OK) return NULL;

    jfloatArray out = (*env)->NewFloatArray(env, 5);
    jfloat vals[5] = { result.loss, result.learning_rate,
                       (float)result.step, result.early_stopped ? 1.0f : 0.0f,
                       result.gradient_norm };
    (*env)->SetFloatArrayRegion(env, out, 0, 5, vals);
    return out;
}

JNIEXPORT jfloatArray JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeTrainBatch(
    JNIEnv *env, jclass cls, jlong h, jfloatArray features,
    jfloatArray targets, jint batchSize, jint numFeatures, jint numTargets)
{
    (void)cls;
    nimcp_brain_t brain = (nimcp_brain_t)(uintptr_t)h;
    jfloat *feats = (*env)->GetFloatArrayElements(env, features, NULL);
    if (!feats) return NULL; /* OOM — JVM already threw */
    jfloat *tgts = (*env)->GetFloatArrayElements(env, targets, NULL);
    if (!tgts) {
        (*env)->ReleaseFloatArrayElements(env, features, feats, JNI_ABORT);
        return NULL; /* OOM — JVM already threw */
    }

    nimcp_training_result_t result = {0};
    nimcp_status_t rc = nimcp_brain_train_batch(
        brain, feats, tgts, (uint32_t)batchSize,
        (uint32_t)numFeatures, (uint32_t)numTargets, &result);

    (*env)->ReleaseFloatArrayElements(env, features, feats, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, targets, tgts, JNI_ABORT);

    if (rc != NIMCP_OK) return NULL;

    jfloatArray out = (*env)->NewFloatArray(env, 5);
    jfloat vals[5] = { result.loss, result.learning_rate,
                       (float)result.step, result.early_stopped ? 1.0f : 0.0f,
                       result.gradient_norm };
    (*env)->SetFloatArrayRegion(env, out, 0, 5, vals);
    return out;
}

JNIEXPORT jfloatArray JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeGetTrainingStats(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)cls;
    uint64_t steps = 0; float loss = 0, lr = 0;
    nimcp_status_t rc = nimcp_brain_get_training_stats(
        (nimcp_brain_t)(uintptr_t)h, &steps, &loss, &lr);
    if (rc != NIMCP_OK) return NULL;

    jfloatArray out = (*env)->NewFloatArray(env, 3);
    jfloat vals[3] = { (float)steps, loss, lr };
    (*env)->SetFloatArrayRegion(env, out, 0, 3, vals);
    return out;
}

JNIEXPORT jfloat JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeStepScheduler(
    JNIEnv *env, jclass cls, jlong h, jfloat valMetric)
{
    (void)env; (void)cls;
    return nimcp_brain_step_scheduler(
        (nimcp_brain_t)(uintptr_t)h, valMetric);
}

// ============================================================================
// Brain - Callbacks
// ============================================================================

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeEnableCallbacks(
    JNIEnv *env, jclass cls, jlong h, jboolean autoCP, jint cpInterval,
    jboolean earlyStop, jint patience, jfloat minDelta,
    jfloat divThresh, jint logInterval)
{
    (void)cls;
    nimcp_callback_config_t cfg = nimcp_callback_config_default();
    cfg.enable_auto_checkpoint = autoCP;
    cfg.checkpoint_interval = (uint32_t)cpInterval;
    cfg.enable_early_stopping = earlyStop;
    cfg.patience = (uint32_t)patience;
    cfg.min_delta = minDelta;
    cfg.divergence_threshold = divThresh;
    cfg.log_interval = (uint32_t)logInterval;
    return (jint)nimcp_brain_enable_callbacks(
        (nimcp_brain_t)(uintptr_t)h, &cfg);
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeDisableCallbacks(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)env; (void)cls;
    return (jint)nimcp_brain_disable_callbacks(
        (nimcp_brain_t)(uintptr_t)h);
}

// Callback trampoline data
typedef struct {
    JavaVM *jvm;
    jobject callback_ref; // Global ref to TrainingCallback
} jni_callback_data_t;

static nimcp_callback_action_t jni_callback_trampoline(
    nimcp_callback_event_t event,
    const nimcp_callback_metrics_t *metrics,
    void *user_data)
{
    jni_callback_data_t *data = (jni_callback_data_t *)user_data;
    if (!data) return NIMCP_CB_ACTION_CONTINUE;

    JNIEnv *env = NULL;
    int attached = 0;
    if ((*data->jvm)->GetEnv(data->jvm, (void **)&env, JNI_VERSION_1_8) != JNI_OK) {
        if ((*data->jvm)->AttachCurrentThread(data->jvm, (void **)&env, NULL) != JNI_OK)
            return NIMCP_CB_ACTION_CONTINUE;
        attached = 1;
    }

    // Build CallbackMetrics object
    jclass metricsCls = (*env)->FindClass(env, "com/nimcp/NIMCP$CallbackMetrics");
    if (!metricsCls) goto cleanup;
    jobject metricsObj = (*env)->AllocObject(env, metricsCls);
    if (!metricsObj) goto cleanup;

    // Set fields
    (*env)->SetLongField(env, metricsObj,
        (*env)->GetFieldID(env, metricsCls, "step", "J"), (jlong)metrics->step);
    (*env)->SetLongField(env, metricsObj,
        (*env)->GetFieldID(env, metricsCls, "epoch", "J"), (jlong)metrics->epoch);
    (*env)->SetFloatField(env, metricsObj,
        (*env)->GetFieldID(env, metricsCls, "loss", "F"), metrics->loss);
    (*env)->SetFloatField(env, metricsObj,
        (*env)->GetFieldID(env, metricsCls, "lossEma", "F"), metrics->loss_ema);
    (*env)->SetFloatField(env, metricsObj,
        (*env)->GetFieldID(env, metricsCls, "learningRate", "F"), metrics->learning_rate);
    (*env)->SetFloatField(env, metricsObj,
        (*env)->GetFieldID(env, metricsCls, "gradientNorm", "F"), metrics->gradient_norm);
    (*env)->SetLongField(env, metricsObj,
        (*env)->GetFieldID(env, metricsCls, "stepTimeUs", "J"), (jlong)metrics->step_time_us);
    (*env)->SetBooleanField(env, metricsObj,
        (*env)->GetFieldID(env, metricsCls, "isConverging", "Z"), metrics->is_converging);
    (*env)->SetBooleanField(env, metricsObj,
        (*env)->GetFieldID(env, metricsCls, "isDiverging", "Z"), metrics->is_diverging);

    // Build CallbackEvent enum value
    jclass eventCls = (*env)->FindClass(env, "com/nimcp/NIMCP$CallbackEvent");
    if (!eventCls) goto cleanup;
    jmethodID fromInt = (*env)->GetStaticMethodID(env, eventCls, "fromInt",
        "(I)Lcom/nimcp/NIMCP$CallbackEvent;");
    if (!fromInt) goto cleanup;
    jobject eventObj = (*env)->CallStaticObjectMethod(env, eventCls, fromInt, (jint)event);

    // Call the callback
    jclass cbCls = (*env)->GetObjectClass(env, data->callback_ref);
    jmethodID onEvent = (*env)->GetMethodID(env, cbCls, "onEvent",
        "(Lcom/nimcp/NIMCP$CallbackEvent;Lcom/nimcp/NIMCP$CallbackMetrics;)"
        "Lcom/nimcp/NIMCP$CallbackAction;");
    if (!onEvent) goto cleanup;

    jobject actionObj = (*env)->CallObjectMethod(env, data->callback_ref,
                                                 onEvent, eventObj, metricsObj);
    if (!actionObj) goto cleanup;

    jclass actionCls = (*env)->GetObjectClass(env, actionObj);
    jfieldID valField = (*env)->GetFieldID(env, actionCls, "value", "I");
    jint actionVal = (*env)->GetIntField(env, actionObj, valField);

    if (attached) (*data->jvm)->DetachCurrentThread(data->jvm);
    return (nimcp_callback_action_t)actionVal;

cleanup:
    if (attached) (*data->jvm)->DetachCurrentThread(data->jvm);
    return NIMCP_CB_ACTION_CONTINUE;
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeRegisterCallback(
    JNIEnv *env, jclass cls, jlong h, jint event,
    jobject callback, jstring name)
{
    (void)cls;
    // Allocate trampoline data
    jni_callback_data_t *data = (jni_callback_data_t *)nimcp_malloc(sizeof(jni_callback_data_t));
    if (!data) return 0;
    memset(data, 0, sizeof(jni_callback_data_t));
    (*env)->GetJavaVM(env, &data->jvm);
    data->callback_ref = (*env)->NewGlobalRef(env, callback);
    if (!data->callback_ref) {
        nimcp_free(data);
        return 0;
    }

    const char *c_name = name ? (*env)->GetStringUTFChars(env, name, NULL) : NULL;
    uint32_t id = nimcp_brain_register_callback(
        (nimcp_brain_t)(uintptr_t)h, (nimcp_callback_event_t)event,
        jni_callback_trampoline, data, c_name);
    if (c_name) (*env)->ReleaseStringUTFChars(env, name, c_name);

    if (id == 0) {
        (*env)->DeleteGlobalRef(env, data->callback_ref);
        nimcp_free(data);
    }
    return (jint)id;
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeUnregisterCallback(
    JNIEnv *env, jclass cls, jlong h, jint cbId)
{
    (void)env; (void)cls;
    return (jint)nimcp_brain_unregister_callback(
        (nimcp_brain_t)(uintptr_t)h, (uint32_t)cbId);
}

JNIEXPORT jfloatArray JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeGetCallbackStats(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)cls;
    uint64_t fired = 0; float avg_time = 0; uint32_t early = 0;
    nimcp_status_t rc = nimcp_brain_get_callback_stats(
        (nimcp_brain_t)(uintptr_t)h, &fired, &avg_time, &early);
    if (rc != NIMCP_OK) return NULL;

    jfloatArray out = (*env)->NewFloatArray(env, 3);
    jfloat vals[3] = { (float)fired, avg_time, (float)early };
    (*env)->SetFloatArrayRegion(env, out, 0, 3, vals);
    return out;
}

// ============================================================================
// Brain - Resize
// ============================================================================

JNIEXPORT jboolean JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeResize(
    JNIEnv *env, jclass cls, jlong h, jint count)
{
    (void)env; (void)cls;
    return nimcp_brain_resize(
        (nimcp_brain_t)(uintptr_t)h, (uint32_t)count);
}

JNIEXPORT jboolean JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeAutoResize(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)env; (void)cls;
    return nimcp_brain_auto_resize((nimcp_brain_t)(uintptr_t)h);
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeGetNeuronCount(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)env; (void)cls;
    return (jint)nimcp_brain_get_neuron_count(
        (nimcp_brain_t)(uintptr_t)h);
}

JNIEXPORT jfloatArray JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeGetUtilizationMetrics(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)cls;
    float util = 0, sat = 0;
    bool ok = nimcp_brain_get_utilization_metrics(
        (nimcp_brain_t)(uintptr_t)h, &util, &sat);
    if (!ok) return NULL;

    jfloatArray out = (*env)->NewFloatArray(env, 2);
    jfloat vals[2] = { util, sat };
    (*env)->SetFloatArrayRegion(env, out, 0, 2, vals);
    return out;
}

// ============================================================================
// Brain - Named Snapshots
// ============================================================================

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeSnapshotSave(
    JNIEnv *env, jclass cls, jlong h, jstring name, jstring desc)
{
    (void)cls;
    const char *c_name = (*env)->GetStringUTFChars(env, name, NULL);
    if (!c_name) return (jint)NIMCP_OK; /* OOM — JVM already threw */
    const char *c_desc = desc ? (*env)->GetStringUTFChars(env, desc, NULL) : NULL;
    /* c_desc NULL is OK when desc is NULL (optional param) */

    nimcp_status_t rc = nimcp_brain_snapshot_save(
        (nimcp_brain_t)(uintptr_t)h, c_name, c_desc);

    (*env)->ReleaseStringUTFChars(env, name, c_name);
    if (c_desc) (*env)->ReleaseStringUTFChars(env, desc, c_desc);
    return (jint)rc;
}

JNIEXPORT jlong JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeSnapshotRestore(
    JNIEnv *env, jclass cls, jlong h, jstring name)
{
    (void)cls;
    const char *c_name = (*env)->GetStringUTFChars(env, name, NULL);
    if (!c_name) return 0; /* OOM — JVM already threw */
    nimcp_brain_t brain = nimcp_brain_snapshot_restore(
        (nimcp_brain_t)(uintptr_t)h, c_name);
    (*env)->ReleaseStringUTFChars(env, name, c_name);
    return (jlong)(uintptr_t)brain;
}

JNIEXPORT jobjectArray JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeSnapshotList(
    JNIEnv *env, jclass cls, jlong h, jint maxCount)
{
    (void)cls;
    nimcp_brain_snapshot_info_t *infos = (nimcp_brain_snapshot_info_t *)
        calloc((size_t)maxCount, sizeof(nimcp_brain_snapshot_info_t));
    if (!infos) return NULL;

    uint32_t count = 0;
    nimcp_status_t rc = nimcp_brain_snapshot_list(
        (nimcp_brain_t)(uintptr_t)h, infos, (uint32_t)maxCount, &count);

    if (rc != NIMCP_OK || count == 0) {
        nimcp_free(infos);
        return NULL;
    }

    jclass strCls = (*env)->FindClass(env, "java/lang/String");
    jobjectArray arr = (*env)->NewObjectArray(env, (jsize)count, strCls, NULL);

    for (uint32_t i = 0; i < count; i++) {
        char buf[1024];
        snprintf(buf, sizeof(buf), "%s\t%s\t%lu\t%u\t%d\t%d",
            infos[i].name, infos[i].description,
            (unsigned long)infos[i].timestamp, infos[i].file_size,
            infos[i].is_compressed ? 1 : 0, infos[i].is_encrypted ? 1 : 0);
        (*env)->SetObjectArrayElement(env, arr, (jsize)i,
            (*env)->NewStringUTF(env, buf));
    }

    nimcp_free(infos);
    return arr;
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeSnapshotDelete(
    JNIEnv *env, jclass cls, jlong h, jstring name)
{
    (void)cls;
    const char *c_name = (*env)->GetStringUTFChars(env, name, NULL);
    if (!c_name) return (jint)NIMCP_OK; /* OOM — JVM already threw */
    nimcp_status_t rc = nimcp_brain_snapshot_delete(
        (nimcp_brain_t)(uintptr_t)h, c_name);
    (*env)->ReleaseStringUTFChars(env, name, c_name);
    return (jint)rc;
}

// ============================================================================
// Brain - COW
// ============================================================================

JNIEXPORT jlong JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeCloneCow(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)env; (void)cls;
    return (jlong)(uintptr_t)nimcp_brain_clone_cow(
        (nimcp_brain_t)(uintptr_t)h);
}

JNIEXPORT jlong JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeSnapshotCow(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)env; (void)cls;
    return (jlong)(uintptr_t)nimcp_brain_snapshot_cow(
        (nimcp_brain_t)(uintptr_t)h);
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeRestoreCow(
    JNIEnv *env, jclass cls, jlong h, jlong snapH)
{
    (void)env; (void)cls;
    return (jint)nimcp_brain_restore_cow(
        (nimcp_brain_t)(uintptr_t)h,
        (nimcp_brain_snapshot_t)(uintptr_t)snapH);
}

JNIEXPORT void JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeSnapshotCowDestroy(
    JNIEnv *env, jclass cls, jlong snapH)
{
    (void)env; (void)cls;
    if (snapH) nimcp_brain_snapshot_destroy(
        (nimcp_brain_snapshot_t)(uintptr_t)snapH);
}

// ============================================================================
// Brain - Working Memory
// ============================================================================

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeWorkingMemoryAdd(
    JNIEnv *env, jclass cls, jlong h, jfloatArray data, jfloat salience)
{
    (void)cls;
    jint len = (*env)->GetArrayLength(env, data);
    jfloat *arr = (*env)->GetFloatArrayElements(env, data, NULL);
    if (!arr) return (jint)NIMCP_OK; /* OOM — JVM already threw */
    nimcp_status_t rc = nimcp_brain_working_memory_add(
        (nimcp_brain_t)(uintptr_t)h, arr, (uint32_t)len, salience);
    (*env)->ReleaseFloatArrayElements(env, data, arr, JNI_ABORT);
    return (jint)rc;
}

JNIEXPORT jfloatArray JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeWorkingMemoryGet(
    JNIEnv *env, jclass cls, jlong h, jint index)
{
    (void)cls;
    uint32_t size = 0;
    const float *item = nimcp_brain_working_memory_get(
        (nimcp_brain_t)(uintptr_t)h, (uint32_t)index, &size);
    if (!item || size == 0) return NULL;

    jfloatArray out = (*env)->NewFloatArray(env, (jsize)size);
    (*env)->SetFloatArrayRegion(env, out, 0, (jsize)size, item);
    return out;
}

JNIEXPORT jintArray JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeWorkingMemoryStats(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)cls;
    uint32_t cur = 0, cap = 0;
    nimcp_status_t rc = nimcp_brain_working_memory_stats(
        (nimcp_brain_t)(uintptr_t)h, &cur, &cap);
    if (rc != NIMCP_OK) return NULL;

    jintArray out = (*env)->NewIntArray(env, 2);
    jint vals[2] = { (jint)cur, (jint)cap };
    (*env)->SetIntArrayRegion(env, out, 0, 2, vals);
    return out;
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeWorkingMemoryRefresh(
    JNIEnv *env, jclass cls, jlong h, jint index)
{
    (void)env; (void)cls;
    return (jint)nimcp_brain_working_memory_refresh(
        (nimcp_brain_t)(uintptr_t)h, (uint32_t)index);
}

// ============================================================================
// Brain - Workspace
// ============================================================================

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeWorkspaceCompete(
    JNIEnv *env, jclass cls, jlong h, jint module,
    jfloatArray content, jfloat strength)
{
    (void)cls;
    jint len = (*env)->GetArrayLength(env, content);
    jfloat *arr = (*env)->GetFloatArrayElements(env, content, NULL);
    if (!arr) return (jint)NIMCP_OK; /* OOM — JVM already threw */
    nimcp_status_t rc = nimcp_brain_workspace_compete(
        (nimcp_brain_t)(uintptr_t)h,
        (nimcp_cognitive_module_t)module,
        arr, (uint32_t)len, strength);
    (*env)->ReleaseFloatArrayElements(env, content, arr, JNI_ABORT);
    return (jint)rc;
}

JNIEXPORT jfloatArray JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeWorkspaceRead(
    JNIEnv *env, jclass cls, jlong h, jint maxDim, jintArray outMeta)
{
    (void)cls;
    float *buf = nimcp_calloc((size_t)maxDim, sizeof(float));
    if (!buf) return NULL;

    uint32_t actualDim = 0;
    nimcp_cognitive_module_t source = NIMCP_MODULE_NONE;
    nimcp_status_t rc = nimcp_brain_workspace_read(
        (nimcp_brain_t)(uintptr_t)h,
        buf, (uint32_t)maxDim, &actualDim, &source);

    if (rc != NIMCP_OK) { nimcp_free(buf); return NULL; }

    jfloatArray out = (*env)->NewFloatArray(env, (jsize)actualDim);
    (*env)->SetFloatArrayRegion(env, out, 0, (jsize)actualDim, buf);
    nimcp_free(buf);

    jint meta[2] = { (jint)actualDim, (jint)source };
    (*env)->SetIntArrayRegion(env, outMeta, 0, 2, meta);
    return out;
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeWorkspaceSubscribe(
    JNIEnv *env, jclass cls, jlong h, jint module)
{
    (void)env; (void)cls;
    return (jint)nimcp_brain_workspace_subscribe(
        (nimcp_brain_t)(uintptr_t)h,
        (nimcp_cognitive_module_t)module);
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeWorkspaceUnsubscribe(
    JNIEnv *env, jclass cls, jlong h, jint module)
{
    (void)env; (void)cls;
    return (jint)nimcp_brain_workspace_unsubscribe(
        (nimcp_brain_t)(uintptr_t)h,
        (nimcp_cognitive_module_t)module);
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeWorkspaceHasBroadcast(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)cls;
    bool has = false;
    nimcp_status_t rc = nimcp_brain_workspace_has_broadcast(
        (nimcp_brain_t)(uintptr_t)h, &has);
    if (rc != NIMCP_OK) return 0;
    return has ? 1 : 0;
}

JNIEXPORT jfloatArray JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeWorkspaceStats(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)cls;
    uint32_t broadcasts = 0, competitions = 0;
    float avg = 0;
    nimcp_status_t rc = nimcp_brain_workspace_stats(
        (nimcp_brain_t)(uintptr_t)h, &broadcasts, &competitions, &avg);
    if (rc != NIMCP_OK) return NULL;

    jfloatArray out = (*env)->NewFloatArray(env, 3);
    jfloat vals[3] = { (float)broadcasts, (float)competitions, avg };
    (*env)->SetFloatArrayRegion(env, out, 0, 3, vals);
    return out;
}

// ============================================================================
// Brain - Oscillations
// ============================================================================

JNIEXPORT jboolean JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeEnableOscillations(
    JNIEnv *env, jclass cls, jlong h, jboolean en)
{
    (void)env; (void)cls;
    return nimcp_enable_complex_oscillations(
        (nimcp_brain_t)(uintptr_t)h, en);
}

JNIEXPORT jboolean JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeIsOscillationsEnabled(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)env; (void)cls;
    return nimcp_is_complex_oscillations_enabled(
        (nimcp_brain_t)(uintptr_t)h);
}

JNIEXPORT jfloatArray JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeGetPhasor(
    JNIEnv *env, jclass cls, jlong h, jint neuronId)
{
    (void)cls;
    nimcp_oscillation_phasor_t p = nimcp_get_oscillation_phasor(
        (nimcp_brain_t)(uintptr_t)h, (uint32_t)neuronId);

    jfloatArray out = (*env)->NewFloatArray(env, 2);
    jfloat vals[2] = { p.amplitude, p.phase };
    (*env)->SetFloatArrayRegion(env, out, 0, 2, vals);
    return out;
}

JNIEXPORT jfloat JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeGetPhaseCoherence(
    JNIEnv *env, jclass cls, jlong h, jintArray neuronIds)
{
    (void)cls;
    jint len = (*env)->GetArrayLength(env, neuronIds);
    jint *ids = (*env)->GetIntArrayElements(env, neuronIds, NULL);
    if (!ids) return 0.0f;

    // Convert jint[] to uint32_t[]
    if (len <= 0 || (size_t)len > SIZE_MAX / sizeof(uint32_t)) {
        (*env)->ReleaseIntArrayElements(env, neuronIds, ids, JNI_ABORT);
        return 0.0f;
    }
    uint32_t *uids = nimcp_malloc((size_t)len * sizeof(uint32_t));
    if (!uids) {
        (*env)->ReleaseIntArrayElements(env, neuronIds, ids, JNI_ABORT);
        return 0.0f;
    }
    for (jint i = 0; i < len; i++) uids[i] = (uint32_t)ids[i];

    float result = nimcp_get_phase_coherence(
        (nimcp_brain_t)(uintptr_t)h, uids, (uint32_t)len);

    nimcp_free(uids);
    (*env)->ReleaseIntArrayElements(env, neuronIds, ids, JNI_ABORT);
    return result;
}

JNIEXPORT jfloat JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeGetPacModulation(
    JNIEnv *env, jclass cls, jlong h, jfloat theta, jfloat gamma)
{
    (void)env; (void)cls;
    return nimcp_get_pac_modulation(
        (nimcp_brain_t)(uintptr_t)h, theta, gamma);
}

// ============================================================================
// Brain - Probe
// ============================================================================

JNIEXPORT jfloatArray JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeProbe(
    JNIEnv *env, jclass cls, jlong h, jobjectArray outStrings)
{
    (void)cls;
    nimcp_brain_probe_t probe = {0};
    nimcp_status_t rc = nimcp_brain_probe(
        (nimcp_brain_t)(uintptr_t)h, &probe);
    if (rc != NIMCP_OK) return NULL;

    // Set task name string
    (*env)->SetObjectArrayElement(env, outStrings, 0,
        (*env)->NewStringUTF(env, probe.task_name));

    // Pack numeric fields into float array
    jfloatArray out = (*env)->NewFloatArray(env, 18);
    jfloat vals[18] = {
        (float)probe.size, (float)probe.task,
        (float)probe.num_neurons, (float)probe.num_synapses,
        (float)probe.num_active_synapses,
        (float)probe.total_inferences, (float)probe.total_learning_steps,
        probe.avg_sparsity, probe.avg_inference_time_us,
        probe.current_learning_rate, probe.accuracy,
        (float)probe.memory_bytes,
        (float)probe.num_inputs, (float)probe.num_outputs,
        probe.is_cow_clone ? 1.0f : 0.0f,
        (float)probe.cow_ref_count,
        (float)probe.cow_shared_bytes, (float)probe.cow_private_bytes
    };
    (*env)->SetFloatArrayRegion(env, out, 0, 18, vals);
    return out;
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Brain_nativeBroadcastProbe(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)env; (void)cls;
    return (jint)nimcp_brain_broadcast_probe(
        (nimcp_brain_t)(uintptr_t)h);
}

// ============================================================================
// Network
// ============================================================================

JNIEXPORT jlong JNICALL Java_com_nimcp_NIMCP_00024Network_nativeCreate(
    JNIEnv *env, jclass cls, jint numInputs, jint numOutputs,
    jint numHidden, jfloat lr)
{
    (void)env; (void)cls;
    return (jlong)(uintptr_t)nimcp_network_create(
        (uint32_t)numInputs, (uint32_t)numOutputs,
        (uint32_t)numHidden, lr);
}

JNIEXPORT void JNICALL Java_com_nimcp_NIMCP_00024Network_nativeDestroy(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)env; (void)cls;
    if (h) nimcp_network_destroy((nimcp_network_t)(uintptr_t)h);
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Network_nativeForward(
    JNIEnv *env, jclass cls, jlong h, jfloatArray inputs,
    jfloatArray outputs)
{
    (void)cls;
    jint ilen = (*env)->GetArrayLength(env, inputs);
    jint olen = (*env)->GetArrayLength(env, outputs);
    jfloat *ins = (*env)->GetFloatArrayElements(env, inputs, NULL);
    if (!ins) return (jint)NIMCP_OK; /* OOM — JVM already threw */
    jfloat *outs = (*env)->GetFloatArrayElements(env, outputs, NULL);
    if (!outs) {
        (*env)->ReleaseFloatArrayElements(env, inputs, ins, JNI_ABORT);
        return (jint)NIMCP_OK; /* OOM — JVM already threw */
    }

    nimcp_status_t rc = nimcp_network_forward(
        (nimcp_network_t)(uintptr_t)h,
        ins, (uint32_t)ilen, outs, (uint32_t)olen);

    (*env)->ReleaseFloatArrayElements(env, inputs, ins, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, outputs, outs, 0); // copy back
    return (jint)rc;
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024Network_nativeTrain(
    JNIEnv *env, jclass cls, jlong h, jfloatArray inputs,
    jfloatArray targets)
{
    (void)cls;
    jint ilen = (*env)->GetArrayLength(env, inputs);
    jint tlen = (*env)->GetArrayLength(env, targets);
    jfloat *ins = (*env)->GetFloatArrayElements(env, inputs, NULL);
    if (!ins) return (jint)NIMCP_OK; /* OOM — JVM already threw */
    jfloat *tgts = (*env)->GetFloatArrayElements(env, targets, NULL);
    if (!tgts) {
        (*env)->ReleaseFloatArrayElements(env, inputs, ins, JNI_ABORT);
        return (jint)NIMCP_OK; /* OOM — JVM already threw */
    }

    nimcp_status_t rc = nimcp_network_train(
        (nimcp_network_t)(uintptr_t)h,
        ins, (uint32_t)ilen, tgts, (uint32_t)tlen);

    (*env)->ReleaseFloatArrayElements(env, inputs, ins, JNI_ABORT);
    (*env)->ReleaseFloatArrayElements(env, targets, tgts, JNI_ABORT);
    return (jint)rc;
}

// ============================================================================
// Ethics
// ============================================================================

JNIEXPORT jlong JNICALL Java_com_nimcp_NIMCP_00024Ethics_nativeCreate(
    JNIEnv *env, jclass cls)
{
    (void)env; (void)cls;
    return (jlong)(uintptr_t)nimcp_ethics_create();
}

JNIEXPORT void JNICALL Java_com_nimcp_NIMCP_00024Ethics_nativeDestroy(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)env; (void)cls;
    if (h) nimcp_ethics_destroy((nimcp_ethics_t)(uintptr_t)h);
}

JNIEXPORT jfloat JNICALL Java_com_nimcp_NIMCP_00024Ethics_nativeCheck(
    JNIEnv *env, jclass cls, jlong h, jfloatArray situation)
{
    (void)cls;
    jint len = (*env)->GetArrayLength(env, situation);
    jfloat *arr = (*env)->GetFloatArrayElements(env, situation, NULL);
    if (!arr) return 0.0f; /* OOM — JVM already threw */

    float score = 0;
    nimcp_status_t rc = nimcp_ethics_check(
        (nimcp_ethics_t)(uintptr_t)h, arr, (uint32_t)len, &score);

    (*env)->ReleaseFloatArrayElements(env, situation, arr, JNI_ABORT);
    if (rc != NIMCP_OK) return 0.0f;
    return score;
}

// ============================================================================
// Knowledge Graph
// ============================================================================

JNIEXPORT jlong JNICALL Java_com_nimcp_NIMCP_00024KnowledgeGraph_nativeCreate(
    JNIEnv *env, jclass cls)
{
    (void)env; (void)cls;
    return (jlong)(uintptr_t)nimcp_knowledge_create();
}

JNIEXPORT void JNICALL Java_com_nimcp_NIMCP_00024KnowledgeGraph_nativeDestroy(
    JNIEnv *env, jclass cls, jlong h)
{
    (void)env; (void)cls;
    if (h) nimcp_knowledge_destroy((nimcp_knowledge_t)(uintptr_t)h);
}

JNIEXPORT jint JNICALL Java_com_nimcp_NIMCP_00024KnowledgeGraph_nativeAddFact(
    JNIEnv *env, jclass cls, jlong h, jstring subject,
    jstring predicate, jstring object)
{
    (void)cls;
    const char *c_subj = (*env)->GetStringUTFChars(env, subject, NULL);
    if (!c_subj) return (jint)NIMCP_OK; /* OOM — JVM already threw */
    const char *c_pred = (*env)->GetStringUTFChars(env, predicate, NULL);
    if (!c_pred) {
        (*env)->ReleaseStringUTFChars(env, subject, c_subj);
        return (jint)NIMCP_OK;
    }
    const char *c_obj  = (*env)->GetStringUTFChars(env, object, NULL);
    if (!c_obj) {
        (*env)->ReleaseStringUTFChars(env, subject, c_subj);
        (*env)->ReleaseStringUTFChars(env, predicate, c_pred);
        return (jint)NIMCP_OK;
    }

    nimcp_status_t rc = nimcp_knowledge_add_fact(
        (nimcp_knowledge_t)(uintptr_t)h, c_subj, c_pred, c_obj);

    (*env)->ReleaseStringUTFChars(env, subject, c_subj);
    (*env)->ReleaseStringUTFChars(env, predicate, c_pred);
    (*env)->ReleaseStringUTFChars(env, object, c_obj);
    return (jint)rc;
}

JNIEXPORT jstring JNICALL Java_com_nimcp_NIMCP_00024KnowledgeGraph_nativeQuery(
    JNIEnv *env, jclass cls, jlong h, jstring query, jint maxLen)
{
    (void)cls;
    const char *c_query = (*env)->GetStringUTFChars(env, query, NULL);
    if (!c_query) return NULL; /* OOM — JVM already threw */
    char *buf = nimcp_calloc((size_t)maxLen + 1, 1);
    if (!buf) {
        (*env)->ReleaseStringUTFChars(env, query, c_query);
        return NULL;
    }

    nimcp_status_t rc = nimcp_knowledge_query(
        (nimcp_knowledge_t)(uintptr_t)h, c_query, buf, (uint32_t)maxLen);

    (*env)->ReleaseStringUTFChars(env, query, c_query);

    jstring result = NULL;
    if (rc == NIMCP_OK) {
        result = (*env)->NewStringUTF(env, buf);
    }
    nimcp_free(buf);
    return result;
}
