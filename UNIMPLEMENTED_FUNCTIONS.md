# NIMCP Unimplemented Functions & TODOs

This document tracks placeholder implementations, TODOs, and unimplemented functions that need to be completed for 100% code coverage.

## Priority 1: Core Functionality (High Impact)

### 1. Network Serialization Checksum Verification
- **File**: `src/io/serialization/nimcp_network_serialization.c:483`
- **TODO**: Recalculate and verify checksum
- **Impact**: Data integrity for saved networks
- **Priority**: HIGH

### 2. CSV Data Loader - num_features
- **File**: `src/io/dataio/nimcp_dataio.c:302`
- **Status**: ✅ FIXED (commit 0ce4a57)
- **Solution**: Added num_features to csv_context_t, populated from config

## Priority 2: Backend Implementations (Medium Impact)

### 3. PostgreSQL Data Backend
- **File**: `src/io/dataio/nimcp_dataio.c:397-432`
- **Status**: Placeholder implementation
- **TODOs**:
  - Line 406: Implement with libpq
  - Line 415: Implement PQfinish, PQclear
  - Line 420: Fetch next N rows from result
  - Line 426: Re-execute query
  - Line 432: PQntuples
- **Impact**: Database-backed training data
- **Priority**: MEDIUM

### 4. Alternative Data Backends
- **File**: `src/io/dataio/nimcp_dataio.c:512-530`
- **TODOs**:
  - Line 512: Implement JSON backend
  - Line 516: Implement Parquet backend
  - Line 530: Implement SQLite backend
- **Impact**: Support for additional data formats
- **Priority**: MEDIUM

### 5. Redis/PostgreSQL Replication Strategies
- **File**: `src/networking/replication/nimcp_replication.c:651-661`
- **TODOs**:
  - Line 651: Implement Redis strategy
  - Line 661: Implement PostgreSQL strategy
- **Impact**: Distributed brain replication
- **Priority**: MEDIUM

## Priority 3: API Enhancements (Lower Impact)

### 6. Node.js Binding - Network Output Size
- **File**: `src/bindings/nodejs/binding.c:150`
- **TODO**: Get actual output size from network
- **Current**: Assumes num_outputs = num_inputs
- **Impact**: Node.js API accuracy
- **Priority**: LOW (assumption is reasonable for SNNs)

### 7. Brain Learning Example Integration
- **File**: `src/io/dataio/nimcp_dataio.c:898`
- **TODO**: Replace with actual brain_learn_example call when API available
- **Impact**: Training loop integration
- **Priority**: MEDIUM

### 8. Brain Decision/Prediction Export
- **File**: `src/io/dataio/nimcp_dataio.c:1121-1128`
- **TODOs**:
  - Line 1121: Get brain decision
  - Line 1126: Write features
  - Line 1128: Write prediction and confidence
- **Impact**: Model inference export
- **Priority**: MEDIUM

### 9. Brain COW Snapshot Implementation
- **File**: `src/api/nimcp.c:649`
- **TODO**: Implement actual COW snapshot using nimcp_cache_reference
- **Impact**: Copy-on-write snapshots
- **Priority**: LOW (current impl functional)

### 10. P2P Network Metrics
- **File**: `src/networking/p2p/nimcp_p2pnode.c:991-992`
- **TODOs**:
  - Line 991: Calculate coordinates from network metrics
  - Line 992: Exchange capabilities in handshake
- **Impact**: P2P network topology
- **Priority**: LOW

## Priority 4: Configuration & Tuning (Lowest Impact)

### 11. Brain Configuration Apply
- **File**: `src/api/nimcp.c:513`
- **TODO**: Apply plasticity settings, ethics config, etc.
- **Impact**: Additional configuration options
- **Priority**: LOW

### 12. Distributed Cognition Stats
- **File**: `src/networking/distributed/nimcp_distributed_cognition.c:321`
- **Note**: Increment stats as placeholder
- **Impact**: Monitoring/metrics
- **Priority**: LOW

### 13. NLP Coherence Analysis
- **File**: `src/nlp/nimcp_spike_nlp.c:380`
- **Note**: Placeholder - real coherence would use variance analysis
- **Impact**: NLP quality metrics
- **Priority**: LOW

## Testing Requirements for 100% Coverage

To achieve 100% code coverage, we need:

### Unit Tests
- [ ] CSV data loader with various feature dimensions
- [ ] Network serialization with checksum verification
- [ ] Backend strategy pattern (mock implementations)
- [ ] Node.js binding edge cases
- [ ] COW snapshot creation and restoration

### Integration Tests
- [ ] End-to-end data loading and training
- [ ] Network save/load with validation
- [ ] Multi-backend data streaming
- [ ] P2P network formation and communication
- [ ] Distributed cognition with replication

### Regression Tests
- [ ] Backward compatibility for serialized networks
- [ ] Data format compatibility (CSV, JSON, etc.)
- [ ] API stability for bindings
- [ ] Configuration changes don't break existing code

## Implementation Strategy

1. **Phase 1**: Complete core functionality (Priority 1)
   - Network checksum verification
   - ✅ CSV num_features fix

2. **Phase 2**: Implement critical backends (Priority 2)
   - PostgreSQL backend
   - JSON/Parquet/SQLite backends
   - Replication strategies

3. **Phase 3**: API enhancements (Priority 3)
   - Brain learning integration
   - Decision/prediction export
   - P2P metrics

4. **Phase 4**: Polish and optimization (Priority 4)
   - Configuration apply
   - Stats and monitoring
   - NLP coherence

5. **Phase 5**: Comprehensive testing
   - Write unit tests for all implemented functions
   - Create integration tests for workflows
   - Add regression tests for backward compatibility

## Progress Tracking

- [x] CSV num_features fix (0ce4a57)
- [ ] Network checksum verification
- [ ] PostgreSQL backend
- [ ] JSON/Parquet/SQLite backends
- [ ] Redis/PostgreSQL replication
- [ ] Brain learning integration
- [ ] Decision/prediction export
- [ ] Unit tests (target: 100% coverage)
- [ ] Integration tests
- [ ] Regression tests

Last Updated: 2025-11-15
