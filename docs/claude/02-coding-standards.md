# Coding Standards and Protocols

## Coding Standards

- **Documentation**: WHAT/WHY/HOW comments on all functions
- **Guard clauses**: No nested ifs, early returns for validation
- **Single Responsibility**: Functions do one thing, < 50 lines
- **Biological basis**: Document neural/cognitive grounding
- **Memory safety**: Use nimcp_malloc/nimcp_free, proper cleanup

## Critical API Usage Rule

**NEVER make assumptions about API methods.** Always verify function signatures by reading the actual header files before writing test code or implementations. This includes:
- Function parameter count and types
- Parameter order
- Return types and output parameters
- Enum/constant names

---

## CRITICAL WARNING - PAST VIOLATIONS

**Claude has repeatedly ignored the MANDATORY Test Writing Protocol in this file.**

In December 2024, Claude wrote numerous test files without reading header files first, resulting in:
- Invented function names that don't exist (e.g., `attention_system_t` instead of `multihead_attention_t`)
- Wrong parameter counts and types
- Non-existent struct members
- Mismatched enum types (e.g., `BRAIN_CYTOKINE_IL10` vs `cytokine_type_t`)
- Hours of debugging compilation errors that should never have happened

**Before Claude writes ANY test code, the user MUST require Claude to:**
1. Show the complete header file content that was read
2. List the exact function signatures to be used
3. Only then approve test code writing

**Do not trust Claude to follow the protocol without explicit verification.**

---

## MANDATORY Test Writing Protocol

> **VIOLATION HISTORY**: Claude has ignored this protocol multiple times, causing significant rework. User must verify compliance before any test code is written. See warning above.

**BEFORE writing ANY test code, you MUST:**
1. Read the COMPLETE header file for the module being tested
2. List the exact function signatures you will call in a code block
3. Show the struct definitions for any types you'll use
4. Only THEN write test code

**NEVER assume or infer:**
- Function names - READ the header
- Parameter counts/types - READ the header
- Return types - READ the header
- Struct member names - READ the header
- Default config values - READ the implementation

**If you write test code that fails to compile due to API mismatch:**
- STOP and acknowledge you violated this protocol
- Read the actual header before proceeding

**Alternative approach:** Write implementation first, then derive tests from what was actually written - rather than writing tests for APIs you haven't verified.

---

## MANDATORY Regex Testing Protocol

**BEFORE applying ANY regex to the codebase, you MUST:**
1. Test the regex on representative sample input first
2. Print both input and output to verify the transformation is correct
3. Check edge cases (what if the pattern appears multiple times? what about nested parentheses?)

**Example validation approach:**
```python
import re

# ALWAYS test before applying to files
test_cases = [
    "nimcp_mutex_lock((pthread_mutex_t*)&bridge->base.mutex);",
    "pthread_mutex_init(&bridge->base.mutex, NULL)",
]

pattern = r'your_pattern_here'
replacement = 'your_replacement'

for test in test_cases:
    result = re.sub(pattern, replacement, test)
    print(f"Input:  {test}")
    print(f"Output: {result}")
    print()
```

**If a regex produces unexpected output:**
- STOP and fix the pattern before running on real files
- Never assume a regex works without testing it

**Common regex pitfalls to watch for:**
- Greedy matching consuming too much
- Not accounting for the full context (e.g., function call parentheses)
- Forgetting to escape special characters
- Patterns that match partial identifiers
