# NIMCP Codebase Refactoring Plan

## Objectives

1. **Comprehensive Documentation**: Every function, structure, and complex code block must have multiline comments explaining WHAT and WHY
2. **Eliminate Nested Control Flow**: No nested ifs, nested fors, or if-else-if-else chains
3. **Apply Design Patterns**: Strategy, Factory, Observer, Command patterns where appropriate
4. **Performance Optimization**: Eliminate O(n²) operations, prefer O(n) or O(log n) algorithms
5. **Code Quality**: Clean, maintainable, self-documenting code

## Coding Standards

### Documentation Requirements

Every function must have:
```c
/**
 * @brief One-line summary of WHAT the function does
 *
 * @detailed Multi-paragraph explanation of:
 * - WHAT: Detailed description of functionality
 * - WHY: The reasoning behind this approach
 * - HOW: Key algorithmic decisions
 * - PERFORMANCE: Time/space complexity
 * - CONSTRAINTS: Preconditions, postconditions
 *
 * @param name Description and constraints
 * @return Description of return value and error conditions
 *
 * @example
 * // Usage example if complex
 *
 * @note Important considerations
 * @warning Potential pitfalls
 */
```

### Control Flow Standards

**AVOID:**
```c
if (condition1) {
    if (condition2) {
        if (condition3) {
            // nested logic
        }
    }
}

for (i = 0; i < n; i++) {
    for (j = 0; j < m; j++) {
        // nested loop
    }
}

if (a) { }
else if (b) { }
else if (c) { }
else { }
```

**PREFER:**
```c
// Early returns for error conditions
if (!valid_precondition) {
    return ERROR;
}

// Extract nested logic into helper functions
result = process_step1(data);
if (!result) return ERROR;

result = process_step2(result);
if (!result) return ERROR;

// Use function tables for dispatch
typedef result_t (*handler_fn)(data_t*);
handler_fn handlers[] = {handle_a, handle_b, handle_c};
return handlers[type](data);

// Use single-pass algorithms instead of nested loops
// O(n²) → O(n) with hash tables or auxiliary structures
```

### Design Patterns to Apply

1. **Strategy Pattern**: For different learning rules, activation functions
2. **Factory Pattern**: For object creation (networks, brains, engines)
3. **Observer Pattern**: For event notifications
4. **Command Pattern**: For deferred actions
5. **State Pattern**: For learning stages, network states
6. **Builder Pattern**: For complex configuration objects

### Performance Optimization Guidelines

1. **Hash Tables**: Use for O(1) lookups instead of O(n) linear search
2. **Index Structures**: Pre-build indices for frequently accessed data
3. **Caching**: Cache computed results that are accessed repeatedly
4. **Single-Pass**: Convert multi-pass algorithms to single-pass where possible
5. **Early Exit**: Return as soon as result is determined
6. **Avoid Reallocation**: Pre-allocate buffers, use capacity planning

## File Refactoring Order

### Phase 1: Core 2.5 Systems (Most Recently Added)
1. **nimcp_ethics.c** (600 lines)
   - Apply Strategy pattern for violation detection
   - Replace nested if-else chains with function tables
   - Optimize policy evaluation (currently O(n) per check)
   - Add comprehensive documentation

2. **nimcp_curiosity.c** (1100 lines)
   - Apply State pattern for learning stages
   - Optimize knowledge gap detection (currently O(n))
   - Replace nested question generation logic
   - Hash table for concept lookups

3. **nimcp_knowledge.c** (1500 lines)
   - Implement search indices for O(log n) lookups
   - Strategy pattern for different knowledge types
   - Flatten nested domain traversal
   - Optimize text learning (currently O(n²) in some paths)

### Phase 2: Brain and Network Infrastructure
4. **nimcp_brain.c** (600 lines)
   - Factory pattern for brain creation
   - Strategy pattern for task types
   - Clean up configuration logic

5. **nimcp_adaptive.c** (700 lines)
   - Optimize forward pass
   - Remove nested loops in threshold computation
   - Strategy pattern for spike encoding

6. **nimcp_neuralnet.c** (1167 lines)
   - Major optimization: synapse lookup O(n²) → O(n) with adjacency lists
   - Strategy pattern for learning rules
   - Strategy pattern for activation functions
   - Clean up compute_step nested loops

### Phase 3: Core Infrastructure
7. **nimcp_p2pnode.c**
   - Observer pattern for message handling
   - Clean up connection management

8. **nimcp_protocol.c**
   - Strategy pattern for message types

9. **nimcp_events.c**
   - Observer/Command pattern for event dispatch

### Phase 4: Examples and Tests
10. Example programs - comprehensive inline documentation

## Refactoring Techniques

### Technique 1: Extract Method
```c
// BEFORE: Nested logic
if (engine && action) {
    if (action->num_affected > 0) {
        for (uint32_t i = 0; i < action->num_affected; i++) {
            if (affected[i] < max) {
                // complex logic
            }
        }
    }
}

// AFTER: Flat, documented
if (!validate_action_context(engine, action)) {
    return create_error_result("Invalid context");
}

return evaluate_each_affected_agent(engine, action);
```

### Technique 2: Replace Conditional with Polymorphism
```c
// BEFORE: if-else-if chain
if (type == TYPE_A) {
    // handle A
} else if (type == TYPE_B) {
    // handle B
} else if (type == TYPE_C) {
    // handle C
}

// AFTER: Function table
typedef result_t (*type_handler_t)(data_t*);
static const type_handler_t handlers[] = {
    [TYPE_A] = handle_type_a,
    [TYPE_B] = handle_type_b,
    [TYPE_C] = handle_type_c
};

return handlers[type](data);
```

### Technique 3: Replace Nested Loops with Hash/Index
```c
// BEFORE: O(n × m)
for (int i = 0; i < n_items; i++) {
    for (int j = 0; j < m_targets; j++) {
        if (items[i].id == targets[j].id) {
            // match found
        }
    }
}

// AFTER: O(n + m) with hash table
hash_table_t* index = create_target_index(targets, m_targets);
for (int i = 0; i < n_items; i++) {
    target_t* match = hash_table_lookup(index, items[i].id);
    if (match) {
        // match found in O(1)
    }
}
```

## Success Criteria

- [ ] Zero nested if statements > 1 level deep
- [ ] Zero nested for loops
- [ ] Zero if-else-if chains > 2 conditions
- [ ] All functions < 50 lines (extract if longer)
- [ ] All functions have comprehensive documentation
- [ ] No O(n²) operations in hot paths
- [ ] All tests passing
- [ ] No performance regression (ideally 2-5x faster)

## Timeline

- **Phase 1**: Core 2.5 systems (~3 files, ~3200 lines)
- **Phase 2**: Brain/Network infrastructure (~3 files, ~2500 lines)
- **Phase 3**: Core infrastructure (~3 files, ~1500 lines)
- **Phase 4**: Examples and polish (~5 files, ~1500 lines)

**Total**: ~8700 lines to refactor

## Notes

- Maintain backward API compatibility where possible
- Add performance benchmarks before/after refactoring
- Document all breaking changes
- Keep git commits atomic (one refactoring technique per commit)
