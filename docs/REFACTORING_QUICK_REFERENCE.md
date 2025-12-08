# Middleware Refactoring Quick Reference

## 🚀 Quick Start

```bash
# Run automated refactoring on all modules
cd /home/bbrelin/nimcp
./scripts/refactor_all_middleware.sh

# Or single module:
./scripts/refactor_middleware_module.sh src/middleware/path/to/file.c module_name
```

## 📋 Manual Steps Per Module (After Automation)

### 1. Add Init/Shutdown (10 min)

Copy from `file.c.init_template` into your source file:

```c
static nimcp_error_t module_init(void) {
    // Already generated in template
}

static void module_shutdown(void) {
    // Already generated in template
}
```

Update header file:
```c
nimcp_error_t module_init(void);
void module_shutdown(void);
```

### 2. Add Auto-Init to Create Functions (2 min)

At start of every `*_create()` function:

```c
module_t* module_create(params) {
    // Add this block:
    if (!s_module_initialized) {
        if (module_init() != NIMCP_SUCCESS) {
            LOG_MODULE_ERROR(MODULE_NAME, "Module initialization failed");
            return NULL;
        }
    }

    // Rest of function...
}
```

### 3. Add Logging (15-30 min)

Use `file.c.logging_guide` as reference:

```c
// At function entry (DEBUG)
LOG_MODULE_DEBUG(MODULE_NAME, "function_name: param=%d", param);

// Successful operations (INFO)
LOG_MODULE_INFO(MODULE_NAME, "Created instance with size=%zu", size);

// Warnings (WARN)
LOG_MODULE_WARN(MODULE_NAME, "High utilization: %.1f%%", util);

// Errors (ERROR)
LOG_MODULE_ERROR(MODULE_NAME, "Allocation failed: size=%zu", size);
```

Add to:
- All create/destroy functions
- All major operations
- All error paths
- All validation failures

### 4. Add Configuration (10-15 min)

Use `file.c.config_keys` as reference:

```c
// Replace hardcoded constants:
#define MAX_SIZE 1024  // OLD

// With config lookups:
size_t max_size = config_get_int("module.max_size", 1024);  // NEW
```

Common patterns:
```c
// Integer limits
size_t max_cap = config_get_int("module.max_capacity", 1024);

// Float thresholds
float threshold = config_get_float("module.threshold", 0.95f);

// Boolean flags
bool enable_warn = config_get_bool("module.warn_on_overflow", true);

// String modes
const char* mode = config_get_string("module.mode", "default");
```

### 5. Add Async Events (15-30 min, if needed)

Replace direct module calls:

```c
// OLD: Direct coupling
result = other_module_process(data);

// NEW: Async event
nimcp_promise_t promise = nimcp_promise_create(sizeof(result_t));
nimcp_future_t future = nimcp_promise_get_future(promise);

event_t event = {
    .type = EVENT_TYPE_PROCESS,
    .data = data,
    .promise = promise
};
event_bus_publish(bus, &event);

// Wait for result
if (nimcp_future_wait_timeout(future, 1000)) {
    result_t result;
    nimcp_future_get(future, &result);
}

nimcp_future_destroy(future);
nimcp_promise_destroy(promise);
```

### 6. Test (15-30 min)

```bash
# Build
cd build && make module_name

# Run tests
./test/unit_middleware_module_name

# Check for leaks
valgrind --leak-check=full ./test/unit_middleware_module_name
```

## ✅ Verification Checklist

For each module, verify:

- [ ] Automated changes applied (includes, memory, MODULE_NAME)
- [ ] Init/shutdown functions added and called
- [ ] Logging at: create, destroy, operations, errors
- [ ] Config used for: limits, thresholds, flags
- [ ] Async events replace tight coupling (if applicable)
- [ ] Compiles with zero warnings
- [ ] Tests pass
- [ ] No memory leaks (valgrind clean)
- [ ] Backup file removed (after verification)

## 🐛 Common Issues

**Issue:** Module won't compile after refactoring
**Fix:** Check that all includes are present, especially:
- `utils/memory/nimcp_memory.h`
- `utils/logging/nimcp_logging.h`
- `utils/config/nimcp_dynamic_config.h`
- `security/nimcp_security.h`

**Issue:** Undefined reference to nimcp_malloc
**Fix:** Ensure utils library is linked in CMakeLists.txt:
```cmake
target_link_libraries(module_name nimcp_utils)
```

**Issue:** Security registration fails
**Fix:** Initialize security system before module:
```c
nimcp_sec_integration_t* sec = nimcp_sec_integration_create();
```

**Issue:** Config lookups return defaults always
**Fix:** Initialize config system:
```c
config_init("path/to/config.toml");
```

## 📊 Progress Tracking

Total: 46 modules
- ✅ Completed: 1 (circular_buffer.c)
- 🔄 In Progress: 0
- ⏳ Automated Ready: 45
- ❌ Not Started: 0

## 📚 Reference Documents

- **Full Guide:** `MIDDLEWARE_REFACTORING_GUIDE.md`
- **Status Report:** `MIDDLEWARE_REFACTORING_STATUS.md`
- **Example Module:** `src/middleware/buffering/nimcp_circular_buffer.c`
- **Test Template:** `test/unit/middleware/test_module_template.cpp`

## 🎯 Time Estimates

Per module:
- Automated stage: 2-3 min (done by script)
- Manual integration: 60-90 min
- Testing: 15-30 min
- **Total per module:** 1.5-2 hours

For all 45 remaining modules:
- Single developer: 67-90 hours (2-3 weeks)
- 4 developers parallel: 17-23 hours (1 week)

## 🔍 Code Review Checklist

- [ ] Follows circular_buffer.c pattern
- [ ] All malloc/free replaced with nimcp_*
- [ ] MODULE_NAME defined and used consistently
- [ ] Init/shutdown functions present
- [ ] Security registration succeeds
- [ ] Logging at appropriate levels
- [ ] Config keys documented
- [ ] Error paths logged
- [ ] No NULL pointer dereferences
- [ ] Consistent code style
- [ ] Comments updated
- [ ] Tests updated

## 💡 Pro Tips

1. **Use the template:** Don't start from scratch, copy from circular_buffer.c
2. **Test incrementally:** Don't wait until all modules are done
3. **Review generated files:** The .init_template and .config_keys are good starting points
4. **Keep backups:** Don't delete .bak files until fully tested
5. **Track progress:** Update checklist as you go
6. **Ask questions:** Better to clarify than guess

## 🆘 Need Help?

1. Check the example: `src/middleware/buffering/nimcp_circular_buffer.c`
2. Read the guide: `MIDDLEWARE_REFACTORING_GUIDE.md`
3. Look at generated files: `*.init_template`, `*.logging_guide`, `*.config_keys`
4. Review the test template: `test/unit/middleware/test_module_template.cpp`

## 📞 Support

For issues or questions:
- Document unexpected patterns
- Compare with circular_buffer.c
- Run automated script in dry-run mode
- Check that all dependencies are available
