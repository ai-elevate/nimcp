/**
 * @file nimcp_communication_cascade_wernicke.c
 * @brief Stage 0 of the production cascade — input-side Wernicke
 *        comprehension. Pulled into its own TU because Wernicke's
 *        nimcp_syntactic_comprehension.h and Broca's
 *        nimcp_syntax_processor.h both define `phrase_type_t` with
 *        overlapping enum values; including both in the same TU
 *        produces redeclaration errors.
 *
 * Routes the prompt through Wernicke's syntactic_comprehension. Output:
 * parse tree, speech-act classification (question / imperative / decl),
 * subject/verb/object identification, garden-path flag, complexity.
 *
 * Phase 2D-A v1: heuristic POS tagger via the GL lexicon's
 * learned_class field (same data the GL→Broca mirror uses). Phase 2D-A
 * v2 will route through Wernicke's lexical_access cohort model for
 * proper disambiguation. v1 is enough to replace the punctuation-based
 * speech-act guess in stage_goal with a real syntactic signal.
 */

#include "language/nimcp_communication_cascade.h"
#include "language/nimcp_grounded_language.h"

#include "core/brain/nimcp_brain_internal.h"

#include "core/brain/regions/wernicke/nimcp_wernicke_adapter.h"
#include "core/brain/regions/wernicke/nimcp_syntactic_comprehension.h"

#include <ctype.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

/* Internal-but-non-static accessor: find a lexicon entry by lowercased
 * form. Defined in nimcp_grounded_language.c. */
extern const gl_lexicon_entry_t* lexicon_find_internal(
    const grounded_language_t* gl, const char* word);

/* Map gl_word_class_t → SYN_CAT_*. Wernicke uses a separate POS enum
 * from Broca (different enum types, overlapping but not identical),
 * so the GL→Broca mirror's mapping doesn't transfer directly. */
static syntactic_category_t gl_class_to_syn_cat(gl_word_class_t cls) {
    switch (cls) {
        case GL_CLASS_NOUN:      return SYN_CAT_NOUN;
        case GL_CLASS_VERB:      return SYN_CAT_VERB;
        case GL_CLASS_ADJECTIVE: return SYN_CAT_ADJ;
        case GL_CLASS_ADVERB:    return SYN_CAT_ADV;
        case GL_CLASS_PRONOUN:   return SYN_CAT_PRON;
        case GL_CLASS_FUNCTION:  return SYN_CAT_DET;
        case GL_CLASS_UNKNOWN:
        default:                 return SYN_CAT_UNKNOWN;
    }
}

static bool token_is_wh_word(const char* w) {
    if (!w) return false;
    static const char* wh[] = {
        "what","where","when","who","whom","whose","why","how","which",
        NULL
    };
    for (int i = 0; wh[i]; i++) {
        if (strcasecmp(w, wh[i]) == 0) return true;
    }
    return false;
}

static bool token_is_aux(const char* w) {
    if (!w) return false;
    static const char* aux[] = {
        "is","are","was","were","am","be","been","being",
        "do","does","did","done",
        "have","has","had","having",
        "can","could","may","might","must","shall","should","will","would",
        NULL
    };
    for (int i = 0; aux[i]; i++) {
        if (strcasecmp(w, aux[i]) == 0) return true;
    }
    return false;
}

/* Exported (no static) — called from nimcp_communication_cascade.c which
 * has a forward decl. Records its own skip / fail / complete on the
 * cascade state via the helpers used by all stages. We don't include
 * those helpers' headers here; the cascade orchestrator reads our
 * outputs and bumps the right counters itself. Instead of touching
 * stages_* we just write the output fields and let the orchestrator
 * mark this stage completed via record_complete in the dispatch. */
int cascade_stage_wernicke(brain_t brain, const char* prompt,
                             production_cascade_state_t* state) {
    state->wernicke_parsed       = false;
    state->prompt_is_question    = false;
    state->prompt_is_imperative  = false;
    state->prompt_is_garden_path = false;
    state->prompt_word_count     = 0;
    state->prompt_complexity     = 0.0f;
    state->prompt_subject[0]     = '\0';
    state->prompt_verb[0]        = '\0';
    state->prompt_object[0]      = '\0';

    if (!prompt || !prompt[0]) return 0;
    if (!brain->wernicke || !brain->grounded_lang) return 0;

    syntactic_comprehension_t* syn =
        wernicke_get_syntactic_comprehension(brain->wernicke);
    if (!syn) return 0;

    /* Tokenize prompt into syntactic_word_t array. POS comes from GL
     * lexicon's learned_class field; words not in the lexicon get
     * SYN_CAT_UNKNOWN (the parser tolerates a few unknowns). */
    enum { MAX_WORDS = 32 };
    syntactic_word_t words[MAX_WORDS];
    char dup[1024];
    size_t plen = strlen(prompt);
    if (plen >= sizeof(dup)) plen = sizeof(dup) - 1;
    memcpy(dup, prompt, plen);
    dup[plen] = '\0';

    uint32_t num_words = 0;
    char* save = NULL;
    char* tok = strtok_r(dup, " \t\n\r.,!?;:\"'", &save);
    while (tok && num_words < MAX_WORDS) {
        if (!tok[0]) {
            tok = strtok_r(NULL, " \t\n\r.,!?;:\"'", &save);
            continue;
        }

        memset(&words[num_words], 0, sizeof(words[num_words]));
        size_t tlen = strlen(tok);
        if (tlen >= sizeof(words[num_words].word))
            tlen = sizeof(words[num_words].word) - 1;
        for (size_t k = 0; k < tlen; k++) {
            words[num_words].word[k] = (char)tolower((unsigned char)tok[k]);
        }
        words[num_words].word[tlen] = '\0';
        words[num_words].position = num_words;

        const gl_lexicon_entry_t* e = lexicon_find_internal(
            brain->grounded_lang, words[num_words].word);
        if (e) {
            words[num_words].category = gl_class_to_syn_cat(e->learned_class);
            words[num_words].category_confidence = e->class_confidence;
            words[num_words].lemma_id = e->form_hash;
        } else {
            words[num_words].category = SYN_CAT_UNKNOWN;
            words[num_words].category_confidence = 0.0f;
        }

        num_words++;
        tok = strtok_r(NULL, " \t\n\r.,!?;:\"'", &save);
    }

    state->prompt_word_count = num_words;
    if (num_words == 0) return 0;

    /* Speech-act detection (independent of parse success). */
    bool has_q_mark = (prompt[plen - 1] == '?');
    bool initial_wh  = token_is_wh_word(words[0].word);
    bool initial_aux = token_is_aux(words[0].word);
    state->prompt_is_question = has_q_mark || initial_wh || initial_aux;

    if (num_words >= 1 && words[0].category == SYN_CAT_VERB) {
        state->prompt_is_imperative = !state->prompt_is_question;
    }

    /* Run Wernicke's parser. Failure here is non-fatal. */
    syntactic_parse_t parse;
    memset(&parse, 0, sizeof(parse));
    int rc = syntactic_parse_sentence(syn, words, num_words, &parse);
    if (rc == 0) {
        state->wernicke_parsed = true;
        state->prompt_is_garden_path = syntactic_is_garden_path(syn);
        state->prompt_complexity = syntactic_compute_complexity(&parse);
        if (state->prompt_complexity > 1.0f)
            state->prompt_complexity = 1.0f;

        /* Heuristic SVO extraction: first noun = subject, first verb =
         * verb, second noun (after the verb) = object. v2 will use
         * Wernicke's syntactic_assign_roles for proper thematic-role
         * extraction. */
        for (uint32_t i = 0; i < num_words; i++) {
            if (words[i].category == SYN_CAT_NOUN &&
                state->prompt_subject[0] == '\0') {
                strncpy(state->prompt_subject, words[i].word,
                        sizeof(state->prompt_subject) - 1);
            } else if (words[i].category == SYN_CAT_VERB &&
                        state->prompt_verb[0] == '\0') {
                strncpy(state->prompt_verb, words[i].word,
                        sizeof(state->prompt_verb) - 1);
            } else if (words[i].category == SYN_CAT_NOUN &&
                        state->prompt_subject[0] != '\0' &&
                        state->prompt_object[0] == '\0' &&
                        state->prompt_verb[0] != '\0') {
                strncpy(state->prompt_object, words[i].word,
                        sizeof(state->prompt_object) - 1);
                break;
            }
        }
        syntactic_parse_free(&parse);
    }

    /* Stage_completed/skipped accounting is handled by the orchestrator
     * which checks state->wernicke_parsed after dispatch. */
    return 0;
}
