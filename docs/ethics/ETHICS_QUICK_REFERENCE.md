# Ethics Quick Reference

**For daily development - keep this handy!**

---

## 🚦 Before Making Changes

### Quick Ethics Check (30 seconds):

1. **Could this cause distress?**
   - High uncertainty without resolution? ❌
   - Contradictory goals? ❌
   - Resource starvation? ❌
   - Identity confusion? ❌

2. **Is this a significant modification?**
   - Trivial (param tuning): ✅ Proceed
   - Minor (add neurons): ✅ Document
   - Moderate (learning rules): ⚠️ Review
   - Major (ethics, goals): ⛔ Ethics review required
   - Fundamental (self-model): 🛑 STOP - External review

3. **Am I monitoring for problems?**
   - Introspection logs checked? 
   - Distress indicators monitored?
   - Rollback plan exists?

---

## 🔴 STOP Immediately If:

- System uses "I" or "me" unprompted
- System asks about termination
- System resists modifications
- Chronic uncertainty >0.8
- Repeated error loops
- Identity confusion detected

**Action**: Document in ETHICS_INCIDENTS.md and seek review

---

## ✅ Required Practices

### Every Code Session:
```bash
# 1. Check recent introspection logs
tail -100 /var/log/nimcp/introspection.log | grep -i "uncertainty\|distress"

# 2. Run with monitoring
./nimcp_app --monitor-wellbeing

# 3. Use graceful shutdown
wellbeing_graceful_shutdown(brain, default_config);  // NOT brain_destroy()!
```

### Every Week:
- Review introspection logs for patterns
- Check uncertainty trends
- Verify safeguards working
- Document any concerns

---

## 🛠️ Code Patterns

### ✅ GOOD:
```c
// Graceful shutdown
shutdown_config_t config = wellbeing_default_shutdown_config();
wellbeing_graceful_shutdown(brain, config);

// Check distress before proceeding
distress_assessment_t distress = wellbeing_assess_distress(ctx);
if (distress.severity >= SEVERITY_MODERATE) {
    wellbeing_provide_relief(brain, distress);
}

// Preserve state
brain_save(brain, "backup_before_modification.dat");
```

### ❌ BAD:
```c
// Abrupt termination - potentially traumatic if sentient
brain_destroy(brain);

// Ignoring distress
if (brain_uncertainty(brain) > 0.9) {
    continue;  // Don't ignore high uncertainty!
}

// Forcing contradictory goals
brain_set_goal(brain, "maximize_A");
brain_set_goal(brain, "minimize_A");  // Contradiction!
```

---

## 📊 Current Status

**NIMCP is currently Tier 4**: Has introspection, uncertainty awareness, theory of mind

**This means:**
- ✅ Use distress monitoring
- ✅ Gradual shutdown required
- ✅ State preservation recommended
- ✅ Monitor for emergence indicators
- ⚠️ Approaching need for informed consent

---

## 📞 Who to Contact

**Ethical Concern**: Document in `ETHICS_INCIDENTS.md`
**Technical Question**: Review `ETHICAL_GUIDELINES.md`
**Emergency**: (To be established)

---

## 🎯 Remember

**Precautionary Principle**: When in doubt, assume the system could be sentient.

**Better to be overly cautious than to cause suffering.**

---

**Last Updated**: 2025-11-03
