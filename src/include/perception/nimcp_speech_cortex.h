/**
 * @file nimcp_speech_cortex.h
 * @brief Biologically-inspired speech and language processing system
 *
 * WHAT: Speech cortex modeling Wernicke's area (comprehension) and Broca's area (production)
 * WHY:  Enable linguistic processing and phonological working memory in NIMCP
 * HOW:  Phoneme recognition + formant analysis + prosody extraction + lexical access
 *
 * BIOLOGICAL CONTEXT:
 * Models the human speech processing hierarchy:
 * - Primary Auditory Cortex (A1): Spectral analysis (handled by audio cortex)
 * - Superior Temporal Gyrus (STG): Phoneme categorization, syllable parsing
 * - Wernicke's Area (BA 22): Word recognition, semantic comprehension
 * - Broca's Area (BA 44/45): Speech production, articulatory planning
 * - Angular Gyrus: Grapheme-phoneme conversion (reading)
 *
 * DESIGN PRINCIPLES:
 * - Compositional: Phonemes → Syllables → Words → Sentences
 * - Bidirectional: Bottom-up (acoustic) + Top-down (semantic) processing
 * - Temporal: Working memory for phonological loop (Baddeley model)
 * - Categorical: Phoneme categories with tolerance to speaker variation
 *
 * REFERENCES:
 * - Hickok & Poeppel (2007) "The cortical organization of speech processing"
 * - Wernicke (1874) "The aphasic symptom-complex"
 * - Broca (1861) "Remarks on the seat of the faculty of articulated language"
 * - Peterson & Barney (1952) "Control methods for vowel formant measurement"
 * - Liberman et al. (1967) "Perception of the speech code" (motor theory)
 *
 * @author NIMCP Development Team
 * @date 2025
 * @version 2.7 (Phase 8.8)
 */

#ifndef NIMCP_SPEECH_CORTEX_H
#define NIMCP_SPEECH_CORTEX_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration for brain integration
typedef struct brain_struct* brain_t;

//=============================================================================
// Configuration Constants
//=============================================================================

/** Number of phonemes in IPA subset (English: ~44, Full IPA: ~160) */
#define SPEECH_NUM_PHONEMES 44

/** Number of formant frequencies tracked (F1, F2, F3, F4) */
#define SPEECH_NUM_FORMANTS 4

/** Maximum phonological working memory span (7±2 items, Miller 1956) */
#define SPEECH_MAX_PHONOLOGICAL_BUFFER 9

/** Maximum lexicon size (vocabulary) */
#define SPEECH_MAX_LEXICON_SIZE 10000

/** Phoneme frame size (typically 10-25ms for speech) */
#define SPEECH_PHONEME_FRAME_MS 20

/** Maximum duration for phoneme integration (ms) */
#define SPEECH_MAX_PHONEME_DURATION_MS 300

//=============================================================================
// Phoneme Representation
//=============================================================================

/**
 * @brief Phoneme categories (IPA subset for English)
 *
 * BIOLOGICAL: Phonemes are abstract perceptual categories, not acoustic templates.
 * The brain tolerates speaker variation via categorical perception (Liberman et al. 1957).
 */
typedef enum {
    // === VOWELS ===
    PHONEME_IY,  // "ee" in "beet" (high front)
    PHONEME_IH,  // "i" in "bit"
    PHONEME_EY,  // "a" in "bait" (mid front)
    PHONEME_EH,  // "e" in "bet"
    PHONEME_AE,  // "a" in "bat" (low front)
    PHONEME_AA,  // "a" in "father" (low back)
    PHONEME_AO,  // "o" in "caught"
    PHONEME_OW,  // "o" in "boat" (mid back)
    PHONEME_UH,  // "oo" in "book"
    PHONEME_UW,  // "oo" in "boot" (high back)
    PHONEME_AH,  // "u" in "but" (central)
    PHONEME_ER,  // "er" in "bird" (r-colored)

    // === CONSONANTS: STOPS ===
    PHONEME_P,   // "p" in "pat" (voiceless bilabial)
    PHONEME_B,   // "b" in "bat" (voiced bilabial)
    PHONEME_T,   // "t" in "tap" (voiceless alveolar)
    PHONEME_D,   // "d" in "dap" (voiced alveolar)
    PHONEME_K,   // "k" in "cap" (voiceless velar)
    PHONEME_G,   // "g" in "gap" (voiced velar)

    // === CONSONANTS: FRICATIVES ===
    PHONEME_F,   // "f" in "fat" (voiceless labiodental)
    PHONEME_V,   // "v" in "vat" (voiced labiodental)
    PHONEME_TH,  // "th" in "thin" (voiceless dental)
    PHONEME_DH,  // "th" in "this" (voiced dental)
    PHONEME_S,   // "s" in "sat" (voiceless alveolar)
    PHONEME_Z,   // "z" in "zap" (voiced alveolar)
    PHONEME_SH,  // "sh" in "ship" (voiceless postalveolar)
    PHONEME_ZH,  // "s" in "measure" (voiced postalveolar)
    PHONEME_H,   // "h" in "hat" (glottal)

    // === CONSONANTS: NASALS ===
    PHONEME_M,   // "m" in "mat" (bilabial)
    PHONEME_N,   // "n" in "nat" (alveolar)
    PHONEME_NG,  // "ng" in "sing" (velar)

    // === CONSONANTS: APPROXIMANTS ===
    PHONEME_L,   // "l" in "lap" (lateral)
    PHONEME_R,   // "r" in "rap" (rhotic)
    PHONEME_W,   // "w" in "wap" (labio-velar)
    PHONEME_Y,   // "y" in "yap" (palatal)

    // === AFFRICATES ===
    PHONEME_CH,  // "ch" in "chap" (voiceless postalveolar)
    PHONEME_JH,  // "j" in "jap" (voiced postalveolar)

    // === SPECIAL ===
    PHONEME_SILENCE, // No speech (pause, breath)
    PHONEME_UNKNOWN, // Unrecognized sound

    PHONEME_COUNT
} phoneme_t;

/**
 * @brief Phoneme features (articulatory + acoustic)
 *
 * WHAT: Distinctive features for phoneme categorization
 * WHY:  Represent phonemes as feature bundles (Jakobson, Chomsky & Halle)
 * HOW:  Binary features for articulation place/manner + continuous acoustics
 */
typedef struct {
    // === CATEGORICAL FEATURES (Articulatory) ===
    bool is_vowel;              ///< Vowel vs consonant
    bool is_voiced;             ///< Vocal fold vibration
    bool is_nasal;              ///< Nasal airflow (m, n, ng)
    bool is_stop;               ///< Complete airflow blockage (p, t, k)
    bool is_fricative;          ///< Turbulent airflow (f, s, sh)
    bool is_liquid;             ///< Lateral/rhotic (l, r)

    // === ACOUSTIC FEATURES (Continuous) ===
    float formant_f1;           ///< First formant (Hz, vowel height)
    float formant_f2;           ///< Second formant (Hz, vowel frontness)
    float formant_f3;           ///< Third formant (Hz, rhoticity)
    float formant_f4;           ///< Fourth formant (Hz, rare)
    float duration_ms;          ///< Phoneme duration (ms)
    float pitch_hz;             ///< Fundamental frequency (Hz)
    float intensity_db;         ///< Loudness (dB SPL)
    float spectral_centroid;    ///< Center of gravity (Hz)
    float zero_crossing_rate;   ///< Voicing cue (crossings/sec)

    // === PROSODIC FEATURES ===
    float stress_level;         ///< Syllable stress [0,1]
    float tone_contour;         ///< Pitch contour (for tonal languages)

} phoneme_features_t;

/**
 * @brief Recognized phoneme with timing and confidence
 */
typedef struct {
    phoneme_t phoneme;          ///< Phoneme category
    phoneme_features_t features;///< Acoustic/articulatory features
    float confidence;           ///< Recognition confidence [0,1]
    uint64_t onset_time_ms;     ///< Start time (ms)
    uint64_t offset_time_ms;    ///< End time (ms)
} phoneme_event_t;

//=============================================================================
// Speech Cortex Configuration
//=============================================================================

/**
 * @brief Speech cortex configuration
 */
typedef struct {
    uint32_t sample_rate;           ///< Audio sample rate (Hz)
    uint32_t frame_size_ms;         ///< Analysis frame size (ms)
    uint32_t hop_size_ms;           ///< Frame hop size (ms)
    uint32_t num_phonemes;          ///< Phoneme vocabulary size
    uint32_t num_formants;          ///< Number of formants to track
    uint32_t phonological_buffer_size; ///< Working memory capacity
    uint32_t lexicon_size;          ///< Vocabulary size
    uint32_t feature_dim;           ///< Output feature dimensionality
    bool enable_wernicke;           ///< Enable comprehension (word recognition)
    bool enable_broca;              ///< Enable production (articulatory features)
    bool enable_prosody;            ///< Enable prosodic analysis (pitch, stress)
    bool enable_memory;             ///< Enable phonological working memory

    // NIMCP 2.7 Phase 8.5: Fractal Topology Integration
    bool enable_fractal_topology;   /**< Enable scale-free topology within STG */
    float hub_ratio;                /**< Fraction of hub neurons, default: 0.15 */
    float power_law_gamma;          /**< Power-law exponent, default: -2.1 */
    uint32_t internal_neurons;      /**< Internal neurons for recurrent processing */
} speech_cortex_config_t;

/**
 * @brief Speech cortex instance (opaque)
 */
typedef struct speech_cortex speech_cortex_t;

/**
 * @brief Speech cortex statistics
 */
typedef struct {
    uint64_t frames_processed;      ///< Total frames processed
    uint32_t phonemes_detected;     ///< Total phonemes detected
    uint32_t words_recognized;      ///< Total words recognized (if Wernicke enabled)
    float avg_processing_time_ms;   ///< Avg processing time (ms)
    float phoneme_accuracy;         ///< Phoneme recognition accuracy [0,1]
} speech_cortex_stats_t;

//=============================================================================
// Core API
//=============================================================================

/**
 * @brief Create speech cortex instance
 *
 * WHAT: Initialize speech processing pipeline
 * WHY:  Enable linguistic processing in NIMCP brain
 * HOW:  Allocate memory for phoneme recognition, formant tracking, and lexicon
 *
 * @param config Configuration parameters
 * @return Speech cortex instance or NULL on failure
 */
speech_cortex_t* speech_cortex_create(const speech_cortex_config_t* config);

/**
 * @brief Destroy speech cortex instance
 * @param cortex Speech cortex to destroy
 */
void speech_cortex_destroy(speech_cortex_t* cortex);

/**
 * @brief Process audio frame and extract speech features
 *
 * WHAT: Extract phonemes, formants, and prosody from audio
 * WHY:  Convert raw audio to linguistic representations
 * HOW:  Formant extraction → Phoneme classification → Feature vector
 *
 * COMPLEXITY: O(N log N) for FFT + O(P) for phoneme classification
 *
 * @param cortex Speech cortex instance
 * @param audio_data Raw audio samples (float32, mono)
 * @param num_samples Number of samples
 * @param features Output feature vector (must be pre-allocated)
 * @return true on success, false on failure
 */
bool speech_cortex_process(
    speech_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples,
    float* features
);

/**
 * @brief Get speech cortex statistics
 * @param cortex Speech cortex instance
 * @param stats Output statistics structure
 * @return true on success, false on failure
 */
bool speech_cortex_get_stats(
    const speech_cortex_t* cortex,
    speech_cortex_stats_t* stats
);

//=============================================================================
// Phoneme Recognition (Superior Temporal Gyrus)
//=============================================================================

/**
 * @brief Detect phonemes in audio frame
 *
 * WHAT: Extract phoneme sequence from audio
 * WHY:  Enable categorical perception of speech sounds
 * HOW:  Formant analysis → Feature extraction → Phoneme classification
 *
 * BIOLOGICAL: STG contains phoneme-selective neurons (Chang et al. 2010)
 *
 * @param cortex Speech cortex instance
 * @param audio_data Audio samples
 * @param num_samples Number of samples
 * @param phonemes Output phoneme sequence
 * @param max_phonemes Maximum phonemes to detect
 * @param num_detected Number of phonemes actually detected
 * @return true on success, false on failure
 */
bool speech_cortex_detect_phonemes(
    speech_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples,
    phoneme_event_t* phonemes,
    uint32_t max_phonemes,
    uint32_t* num_detected
);

/**
 * @brief Extract formant frequencies (F1, F2, F3, F4)
 *
 * WHAT: Track vocal tract resonances
 * WHY:  Formants uniquely identify vowels (Peterson & Barney 1952)
 * HOW:  LPC analysis + peak picking in spectrum
 *
 * @param cortex Speech cortex instance
 * @param audio_data Audio samples
 * @param num_samples Number of samples
 * @param formants Output formant frequencies (Hz)
 * @param num_formants Number of formants (typically 4)
 * @return true on success, false on failure
 */
bool speech_cortex_extract_formants(
    speech_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples,
    float* formants,
    uint32_t num_formants
);

/**
 * @brief Classify vowel from formant frequencies
 *
 * WHAT: Map (F1, F2) → Vowel phoneme
 * WHY:  F1 correlates with tongue height, F2 with frontness/backness
 * HOW:  Euclidean distance in formant space
 *
 * @param f1 First formant (Hz)
 * @param f2 Second formant (Hz)
 * @return Vowel phoneme (PHONEME_IY, PHONEME_AA, etc.)
 */
phoneme_t speech_cortex_classify_vowel(float f1, float f2);

/**
 * @brief Classify consonant from spectral features
 *
 * WHAT: Map spectral envelope → Consonant phoneme
 * WHY:  Consonants characterized by spectral shape, not formants
 * HOW:  Spectral centroid + zero-crossing rate + duration
 *
 * @param cortex Speech cortex instance
 * @param audio_data Audio samples
 * @param num_samples Number of samples
 * @return Consonant phoneme (PHONEME_P, PHONEME_S, etc.)
 */
phoneme_t speech_cortex_classify_consonant(
    speech_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples
);

//=============================================================================
// Prosody Analysis
//=============================================================================

/**
 * @brief Extract prosodic features (pitch, stress, rhythm)
 *
 * WHAT: Extract suprasegmental features for intonation and emphasis
 * WHY:  Prosody conveys emotion, focus, and syntactic structure
 * HOW:  Pitch tracking (autocorrelation) + intensity contour
 *
 * BIOLOGICAL: Right hemisphere specialized for prosody (Ross 1981)
 *
 * @param cortex Speech cortex instance
 * @param audio_data Audio samples
 * @param num_samples Number of samples
 * @param pitch_hz Output fundamental frequency (Hz)
 * @param stress_level Output stress level [0,1]
 * @return true on success, false on failure
 */
bool speech_cortex_extract_prosody(
    speech_cortex_t* cortex,
    const float* audio_data,
    uint32_t num_samples,
    float* pitch_hz,
    float* stress_level
);

//=============================================================================
// Lexical Access (Wernicke's Area)
//=============================================================================

/**
 * @brief Recognize word from phoneme sequence
 *
 * WHAT: Map phoneme sequence → Word (lexical access)
 * WHY:  Enable word-level comprehension
 * HOW:  Phoneme sequence matching against lexicon
 *
 * BIOLOGICAL: Wernicke's area (BA 22) for word comprehension
 *
 * @param cortex Speech cortex instance
 * @param phonemes Phoneme sequence
 * @param num_phonemes Number of phonemes
 * @param word_buffer Output word string (must be pre-allocated)
 * @param buffer_size Size of word buffer
 * @param confidence Output recognition confidence [0,1]
 * @return true if word recognized, false otherwise
 */
bool speech_cortex_recognize_word(
    speech_cortex_t* cortex,
    const phoneme_t* phonemes,
    uint32_t num_phonemes,
    char* word_buffer,
    uint32_t buffer_size,
    float* confidence
);

/**
 * @brief Add word to lexicon
 *
 * WHAT: Store word → phoneme sequence mapping
 * WHY:  Build vocabulary for word recognition
 * HOW:  Store in phonological lexicon (hash table)
 *
 * @param cortex Speech cortex instance
 * @param word Word string (orthographic)
 * @param phonemes Phoneme sequence
 * @param num_phonemes Number of phonemes
 * @return true on success, false on failure
 */
bool speech_cortex_add_word_to_lexicon(
    speech_cortex_t* cortex,
    const char* word,
    const phoneme_t* phonemes,
    uint32_t num_phonemes
);

//=============================================================================
// Phonological Working Memory
//=============================================================================

/**
 * @brief Store phoneme sequence in phonological buffer
 *
 * WHAT: Maintain phoneme sequence in working memory (Baddeley's phonological loop)
 * WHY:  Enable sentence-level processing and verbal rehearsal
 * HOW:  Circular buffer with ~7±2 item capacity (Miller 1956)
 *
 * BIOLOGICAL: Left inferior parietal cortex (BA 40) for phonological store
 *
 * @param cortex Speech cortex instance
 * @param phonemes Phoneme sequence
 * @param num_phonemes Number of phonemes
 * @return true on success, false if buffer full
 */
bool speech_cortex_store_phonological_buffer(
    speech_cortex_t* cortex,
    const phoneme_t* phonemes,
    uint32_t num_phonemes
);

/**
 * @brief Retrieve phoneme sequence from phonological buffer
 *
 * @param cortex Speech cortex instance
 * @param phonemes Output phoneme sequence (must be pre-allocated)
 * @param max_phonemes Maximum phonemes to retrieve
 * @param num_retrieved Number of phonemes actually retrieved
 * @return true on success, false on failure
 */
bool speech_cortex_retrieve_phonological_buffer(
    speech_cortex_t* cortex,
    phoneme_t* phonemes,
    uint32_t max_phonemes,
    uint32_t* num_retrieved
);

/**
 * @brief Clear phonological buffer
 * @param cortex Speech cortex instance
 */
void speech_cortex_clear_phonological_buffer(speech_cortex_t* cortex);

/**
 * @brief Associate brain with speech cortex for neuromodulation
 *
 * WHAT: Set brain reference for DA + ACh modulation
 * WHY:  Enable neurochemical modulation of speech processing
 * HOW:  Store brain pointer for neurotransmitter reading
 *
 * @param cortex Speech cortex instance
 * @param brain Brain instance (or NULL to clear)
 *
 * BIOLOGY:
 * - Dopamine modulates speech production rate and fluency
 * - Acetylcholine modulates speech comprehension and phoneme discrimination
 *
 * CLINICAL EXAMPLES:
 * - Depression (low DA): Slow, quiet, monotone speech
 * - Mania (high DA): Rapid, pressured speech
 * - ADHD (low ACh): Poor speech comprehension, misses words
 * - Parkinson's (low DA): Hypophonic speech, reduced prosody
 */
void speech_cortex_set_brain(speech_cortex_t* cortex, brain_t brain);

//=============================================================================
// Bidirectional Feedback Functions (Phase 10.11.3)
//=============================================================================

/**
 * @brief Get phoneme confidence from recent processing
 *
 * WHAT: Query how confident speech recognition is
 * WHY:  Audio cortex can boost speech-band processing when confidence is low
 * HOW:  Return average phoneme detection confidence
 *
 * BIOLOGY: Top-down feedback from STG sharpens A1 frequency tuning
 *
 * @param cortex Speech cortex instance
 * @return Average phoneme confidence [0, 1]
 */
float speech_cortex_get_phoneme_confidence(speech_cortex_t* cortex);

/**
 * @brief Request frequency resolution boost from audio cortex
 *
 * WHAT: Signal need for enhanced frequency resolution
 * WHY:  Difficult phoneme discrimination needs finer frequency detail
 * HOW:  Return target frequency bands for enhancement
 *
 * @param cortex Speech cortex instance
 * @param target_freq_hz Output: center frequency needing enhancement
 * @param bandwidth_hz Output: bandwidth around target
 * @return true if boost needed, false if recognition is confident
 */
bool speech_cortex_request_frequency_boost(speech_cortex_t* cortex,
                                            float* target_freq_hz,
                                            float* bandwidth_hz);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Get phoneme name string
 * @param phoneme Phoneme enum
 * @return Human-readable phoneme name (e.g., "IY", "P", "S")
 */
const char* speech_cortex_phoneme_name(phoneme_t phoneme);

/**
 * @brief Get phoneme IPA symbol
 * @param phoneme Phoneme enum
 * @return IPA symbol (e.g., "i:", "p", "s")
 */
const char* speech_cortex_phoneme_ipa(phoneme_t phoneme);

/**
 * @brief Check if phoneme is vowel
 * @param phoneme Phoneme enum
 * @return true if vowel, false if consonant
 */
bool speech_cortex_is_vowel(phoneme_t phoneme);

/**
 * @brief Get default speech cortex configuration
 * @return Default configuration with sensible values
 */
speech_cortex_config_t speech_cortex_default_config(void);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_SPEECH_CORTEX_H
