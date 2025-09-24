# Requirements Specification: Actuarial Cash Flow Calculation Optimization System

## Document Information
- **Project**: ACFC Optimization System (Implementation Requirements)
- **Version**: 2.0
- **Date**: 2025-09-09
- **Purpose**: Requirements specification for implementing the optimized actuarial cash flow calculation system
- **Target**: Support full implementation of [sas_code_optimized.md](sas_code_optimized.md) approach

---

## 1. EXECUTIVE SUMMARY

### 1.1 Project Overview
This system transforms the existing 17-minute actuarial cash flow calculation (ACFC) into a 2.5-minute parallel processing engine through infrastructure-aware optimization while maintaining complete mathematical accuracy of the underlying Monte Carlo stochastic modeling.

### 1.2 Business Objectives
- **Performance**: Deliver 6.8x improvement (production) to 13.6x improvement (test environment)
- **Mathematical Integrity**: Preserve all actuarial formulas and regulatory compliance
- **Resource Optimization**: Utilize full 12-core/96GB production capacity or 24-core/192GB test capacity
- **Implementation Feasibility**: Build directly upon existing ACFC algorithm without algorithmic changes

### 1.3 Key Performance Targets
| Environment | Core Allocation | Target Runtime | Performance Improvement |
|-------------|----------------|----------------|------------------------|
| **Production** | 12 cores | 2.5 minutes | 6.8x (from 17 minutes) |
| **Test** | 24 cores | 1.25 minutes | 13.6x (from 17 minutes) |

---

## 2. ACTUARIAL ALGORITHM REQUIREMENTS

### 2.1 Mathematical Foundation Preservation

#### 2.1.1 Core Actuarial Formulas (Unchanged)
- **REQ-MATH-001**: System MUST preserve fund value projection formula
  ```
  MT_VM(t+1) = MT_VM(t) × [1 + RENDEMENT(s,t) - PC_REVENU_FDS - FRAIS_ADJ(t)]
  ```

- **REQ-MATH-002**: System MUST preserve survival probability modeling
  ```
  TX_SURVIE(t+1) = TX_SURVIE(t) × [1 - Qx(age+t)] × [1 - WX(t)]
  ```

- **REQ-MATH-003**: System MUST preserve death benefit guarantee mechanism
  ```
  MT_GAR_DECES(t) = MAX(MT_GAR_DECES(t-1), MT_VM(t)) [if reset conditions met]
  Death_Claim(t) = MAX(0, MT_GAR_DECES(t) - MT_VM(t)) × Qx(age+t) × TX_SURVIE(t-1)
  ```

- **REQ-MATH-004**: System MUST preserve cash flow component calculations
  ```
  FLUX_NET(t) = REVENUS(t) + FRAIS_GEST(t) + COMMISSIONS(t) + FRAIS_GEN(t) + PMT_GARANTIE(t)
  VP_FLUX_NET(t) = FLUX_NET(t) × TX_ACTU(t)
  ```

#### 2.1.2 Three-Tier Computational Structure
- **REQ-STRUCT-001**: System MUST implement external calculation tier
  - Scale: 100 accounts × 100 scenarios × 101 years = 1.01M projections
  - Purpose: Primary economic scenario processing
  - Output: External cash flows and population states

- **REQ-STRUCT-002**: System MUST implement reserve calculation tier  
  - Scale: 1.01M external results × 100 internal scenarios × 101 years = 10.2B calculations
  - Purpose: Standard actuarial assumptions (no shocks)
  - Aggregation: Mean across internal scenarios
  - Output: Required reserve values

- **REQ-STRUCT-003**: System MUST implement capital calculation tier
  - Scale: 1.01M external results × 100 internal scenarios × 101 years = 10.2B calculations  
  - Purpose: Stressed assumptions with 35% capital shock
  - Shock application: Fund values reduced by CHOC_CAPITAL (0.35)
  - Aggregation: Mean across internal scenarios
  - Output: Required capital values (net of reserves)

#### 2.1.3 Distributable Cash Flow Integration
- **REQ-DIST-001**: System MUST calculate profit emergence
  ```
  PROFIT(t) = FLUX_NET(t) + [RESERVE(t) - RESERVE(t-1)]
  ```

- **REQ-DIST-002**: System MUST calculate distributable cash flows
  ```  
  FLUX_DISTRIBUABLES(t) = PROFIT(t) + [CAPITAL(t) - CAPITAL(t-1)]
  VP_FLUX_DISTRIBUABLES = FLUX_DISTRIBUABLES(t) / (1 + HURDLE_RT)^t
  ```

### 2.2 Data Structure Requirements

#### 2.2.1 Input Dataset Specifications
- **REQ-DATA-001**: System MUST process POPULATION dataset
  - Required fields: ID_COMPTE, age_deb, MT_VM, MT_GAR_DECES, product parameters
  - Auto-detection: NBCPT = COUNT(*) FROM POPULATION
  - Validation: Ensure data completeness and range validation

- **REQ-DATA-002**: System MUST process rate table datasets
  - TX_DECES: Mortality rates (AGE, Qx) for age-based lookups
  - TX_RETRAIT: Lapse rates (an_proj, WX) for duration-based lookups
  - TX_INTERET: External discount rates (an_proj, TX_ACTU)
  - TX_INTERET_INT: Internal discount rates (an_eval, TX_ACTU_INT)

- **REQ-DATA-003**: System MUST process investment scenario dataset
  - RENDEMENT: Investment returns (scn_proj, an_proj, TYPE, RENDEMENT)
  - Type detection: EXTERNE vs INTERNE scenario identification
  - Auto-detection: NB_SC = MAX(scn_proj WHERE TYPE='EXTERNE')
  - Auto-detection: NB_SC_INT = MAX(scn_proj WHERE TYPE='INTERNE')

#### 2.2.2 Output Dataset Requirements
- **REQ-OUTPUT-001**: System MUST produce SERVEUR.CALCULS_SOMMAIRE
  - Structure: 10,000 rows (100 accounts × 100 scenarios)
  - Fields: ID_COMPTE, scn_eval, VP_FLUX_DISTRIBUABLES
  - Validation: Results identical to sequential baseline within precision tolerance

---

## 3. PERFORMANCE OPTIMIZATION REQUIREMENTS

### 3.1 Infrastructure Discovery and Configuration

#### 3.1.1 Dynamic Hardware Detection
- **REQ-INFRA-001**: System MUST auto-detect hardware capacity from configuration
  - Read HARDWARE_CPU_CORES, HARDWARE_MEMORY_PER_CPU_GB from seed parameters
  - Calculate total available memory: HARDWARE_CPU_CORES × HARDWARE_MEMORY_PER_CPU_GB
  - Determine current allocation: MIN(HARDWARE_CPU_CORES, MAX_USER_CPU_LIMIT)

- **REQ-INFRA-002**: System MUST calculate optimal batch sizing
  - OPTIMAL_BATCHES = MAX_USER_CPU_LIMIT (12 production, up to 24 test)
  - ACCOUNTS_PER_BATCH = NBCPT_ACTUAL / OPTIMAL_BATCHES
  - MEMORY_PER_BATCH = 8GB per batch for stable processing

#### 3.1.2 Resource Allocation Strategy
- **REQ-RESOURCE-001**: System MUST implement organizational policy compliance
  - Respect MAX_USER_CPU_LIMIT for shared resource environments
  - Support BUSINESS_HOURS_CONSERVATIVE mode with reduced resource usage
  - Provide TEST_MODE override for full 24-core capacity validation

- **REQ-RESOURCE-002**: System MUST optimize memory utilization
  - Target: 50% capacity (96GB/192GB) in production
  - Target: 90% capacity (172GB/192GB) in test mode
  - Implement memory pool allocation per processing batch

### 3.2 Parallel Processing Engine

#### 3.2.1 Account-Level Parallelization
- **REQ-PARALLEL-001**: System MUST implement batch parallel processing
  ```sas
  %do batch_id = 1 %to &OPTIMAL_BATCHES.;
      /* Process account range in parallel */
      %batch_calculate_external(batch_id=&batch_id.);
      %batch_calculate_internal(batch_id=&batch_id.);
  %end;
  ```

- **REQ-PARALLEL-002**: System MUST distribute workload evenly
  - Equal account distribution across all available cores
  - Handle remainder accounts in final batches
  - Minimize processing time variance between batches

#### 3.2.2 Data Access Optimization
- **REQ-ACCESS-001**: System MUST implement vectorized rate lookups
  - Replace hash table operations with pre-loaded arrays
  - Mortality rates: `array qx_rates{0:99} _temporary_;`
  - Lapse rates: `array wx_rates{0:100} _temporary_;`
  - Investment returns: `array rend_matrix{1:NB_SC, 0:NB_AN} _temporary_;`
  - Target: 3x improvement in lookup performance

- **REQ-ACCESS-002**: System MUST implement bulk data operations
  - Replace nested DO loops with SQL Cartesian products for scenario expansion
  - Eliminate individual dataset creation/deletion cycles
  - Process entire batches with single data step operations
  - Target: 2x improvement in data processing efficiency

### 3.3 Memory Management Architecture

#### 3.3.1 Memory Pool Implementation
- **REQ-MEMORY-001**: System MUST implement batch-specific memory libraries
  ```sas
  %do i = 1 %to &OPTIMAL_BATCHES.;
      libname BATCH&i. "&PATH.\Memlib\Batch&i." MEMLIB MEMSIZE=8G;
  %end;
  ```

- **REQ-MEMORY-002**: System MUST implement intelligent dataset lifecycle
  - Pre-allocate datasets with required variable structures
  - Use views for zero-copy operations where possible
  - Implement systematic cleanup after batch completion

#### 3.3.2 Large-Scale Data Handling
- **REQ-SCALE-001**: System MUST handle 20.4 billion calculation matrix
  - External calculations: 1.01M projections per batch
  - Internal calculations: 10.2B reserve + 10.2B capital calculations
  - Intermediate result caching for nested scenario processing
  - Graceful memory management under constraint conditions

---

## 4. IMPLEMENTATION ARCHITECTURE REQUIREMENTS

### 4.1 Core System Components

#### 4.1.1 Dynamic Configuration Engine
- **REQ-CONFIG-001**: System MUST implement `%initialize_dynamic_config` macro
  - Auto-detect dataset dimensions from actual input files
  - Calculate optimal processing parameters based on infrastructure
  - Generate configuration report with performance expectations
  - Provide infrastructure utilization summary

#### 4.1.2 Parallel Processing Engine  
- **REQ-PROC-001**: System MUST implement `%parallel_account_processor` macro
  - Distribute accounts across available processing batches
  - Execute external and internal calculations in parallel
  - Merge results from all batches into unified dataset
  - Handle batch failures with appropriate error recovery

#### 4.1.3 Vectorized Calculation Engine
- **REQ-CALC-001**: System MUST implement `%batch_calculate_external` macro
  - Pre-load all rate tables into memory arrays during initialization
  - Process entire account batches with single data step operations
  - Implement identical mathematical logic to sequential version
  - Apply survival thresholds for computational efficiency

- **REQ-CALC-002**: System MUST implement `%batch_calculate_internal` macro
  - Handle both reserve (Type 1) and capital (Type 2) calculations
  - Apply 35% capital shock for stressed scenarios
  - Aggregate internal scenarios using mean calculations
  - Merge external and internal results for distributable cash flow calculation

### 4.2 Quality Assurance Framework

#### 4.2.1 Mathematical Equivalence Validation
- **REQ-VALID-001**: System MUST implement result comparison framework
  ```sas
  proc compare base=SERVEUR.calculs_sommaire_original
               compare=SERVEUR.calculs_sommaire_optimized
               criterion=0.01;
  run;
  ```

- **REQ-VALID-002**: System MUST validate intermediate calculations
  - Compare fund value projections at key projection points
  - Validate survival probability calculations
  - Verify death benefit guarantee applications
  - Confirm present value calculation accuracy

#### 4.2.2 Performance Monitoring
- **REQ-MONITOR-001**: System MUST implement `%performance_summary` macro
  - Track execution time by processing phase
  - Monitor resource utilization (CPU, memory) during execution
  - Calculate and report performance improvement ratios
  - Generate detailed performance metrics for optimization validation

---

## 5. NON-FUNCTIONAL REQUIREMENTS

### 5.1 Performance Requirements

#### 5.1.1 Execution Performance Targets
- **REQ-PERF-001**: System MUST achieve minimum performance improvements
  - Production environment (12 cores): 6.8x improvement (17 min → 2.5 min)
  - Test environment (24 cores): 13.6x improvement (17 min → 1.25 min)
  - Configuration overhead: ≤30 seconds
  - Total optimization overhead: ≤2% of execution time

#### 5.1.2 Resource Utilization Targets
- **REQ-UTIL-001**: System MUST optimize infrastructure utilization
  - CPU utilization: ≥85% on allocated cores during processing
  - Memory utilization: 50-90% of allocated memory capacity
  - Processing efficiency: Minimize idle time between calculation phases

### 5.2 Scalability and Reliability

#### 5.2.1 Infrastructure Scalability
- **REQ-SCALE-002**: System MUST scale across infrastructure configurations
  - Core allocation: Support 1-24 cores based on MAX_USER_CPU_LIMIT
  - Memory scaling: Support 8GB-192GB total memory configurations
  - Account portfolios: Support 10-1000+ accounts efficiently
  - Scenario counts: Support 10-1000+ economic scenarios

#### 5.2.2 Error Handling and Recovery
- **REQ-ERROR-001**: System MUST implement comprehensive error handling
  - Batch failure recovery: Continue processing with remaining batches
  - Memory constraint handling: Graceful degradation with reduced batch sizes
  - Data validation: Detect and report data quality issues
  - Fallback mechanism: Sequential processing if parallel processing fails

### 5.3 Maintainability and Documentation

#### 5.3.1 Code Organization
- **REQ-MAINT-001**: System MUST implement modular architecture
  - Separate macros for configuration, processing, and reporting
  - Clear separation between actuarial logic and optimization logic
  - Parameterized configuration through external macro variables
  - Comprehensive inline documentation for all optimization techniques

#### 5.3.2 Configuration Management
- **REQ-MAINT-002**: System MUST support flexible configuration
  - Integration with seed_use_case.md parameter structure
  - Support for production vs. test environment configurations
  - Business hours and resource policy compliance
  - Override capabilities for testing and validation scenarios

---

## 6. TESTING AND VALIDATION REQUIREMENTS

### 6.1 Functional Testing

#### 6.1.1 Mathematical Accuracy Testing
- **REQ-TEST-001**: System MUST validate mathematical consistency
  - Compare all intermediate calculations with sequential baseline
  - Verify actuarial formula implementation accuracy
  - Test regulatory compliance (35% capital shock application)
  - Validate distributable cash flow calculation logic

#### 6.1.2 Performance Testing
- **REQ-TEST-002**: System MUST validate performance improvements
  - Measure execution time across different infrastructure configurations
  - Validate resource utilization efficiency (CPU, memory, I/O)
  - Test scalability with varying account and scenario counts
  - Verify performance targets (6.8x and 13.6x improvements)

### 6.2 Integration Testing

#### 6.2.1 Infrastructure Testing
- **REQ-TEST-003**: System MUST be tested across environment configurations
  - Production environment: 12 cores, 96GB memory, organizational limits
  - Test environment: 24 cores, 192GB memory, full capacity
  - Business hours vs. off-hours resource allocation testing
  - Multi-tenant environment compatibility verification

#### 6.2.2 Data Variation Testing
- **REQ-TEST-004**: System MUST handle diverse input data scenarios
  - Portfolio sizes: 10, 50, 100, 500+ accounts
  - Scenario variations: 10, 50, 100, 500+ economic scenarios
  - Age distributions: uniform vs. highly variable populations
  - Complex vs. simple actuarial product configurations

### 6.3 Regression Testing

#### 6.3.1 Business Logic Validation
- **REQ-TEST-005**: System MUST maintain business rule compliance
  - Validate all actuarial calculation accuracy against known benchmarks
  - Test regulatory capital calculation compliance
  - Verify financial reporting calculation accuracy
  - Confirm audit trail and calculation transparency

---

## 7. CONSTRAINTS AND ASSUMPTIONS

### 7.1 Technical Constraints

#### 7.1.1 SAS Platform Constraints
- **CONSTRAINT-001**: Limited by SAS 9.4+ platform capabilities
- **CONSTRAINT-002**: Threading limited by available SAS license features
- **CONSTRAINT-003**: Memory allocation limited by 64-bit SAS installation
- **CONSTRAINT-004**: I/O performance dependent on network storage infrastructure

#### 7.1.2 Algorithmic Constraints
- **CONSTRAINT-005**: Core actuarial mathematics cannot be modified
- **CONSTRAINT-006**: Regulatory compliance requirements must be preserved
- **CONSTRAINT-007**: Existing validation and testing frameworks must remain valid
- **CONSTRAINT-008**: Output format and precision must match sequential baseline

### 7.2 Environmental Assumptions

#### 7.2.1 Infrastructure Assumptions
- **ASSUMPTION-001**: Hardware specifications provided in seed_use_case.md are accurate
- **ASSUMPTION-002**: SAS installation supports required memory allocation (MEMLIB)
- **ASSUMPTION-003**: Network storage can support parallel I/O operations
- **ASSUMPTION-004**: Organizational resource limits (MAX_USER_CPU_LIMIT) are correctly specified

#### 7.2.2 Data Quality Assumptions
- **ASSUMPTION-005**: Input datasets follow standard actuarial data structure
- **ASSUMPTION-006**: Rate tables contain complete age and duration coverage
- **ASSUMPTION-007**: Economic scenarios provide complete scenario/year matrix
- **ASSUMPTION-008**: Population data contains no missing critical values

---

## 8. SUCCESS CRITERIA AND DELIVERABLES

### 8.1 Primary Success Metrics

#### 8.1.1 Performance Achievement
- **SUCCESS-001**: Achieve documented performance improvements
  - Production target: 17 minutes → 2.5 minutes (6.8x improvement)
  - Test capacity target: 17 minutes → 1.25 minutes (13.6x improvement)
  - Resource utilization: ≥85% of allocated CPU capacity

#### 8.1.2 Mathematical Integrity
- **SUCCESS-002**: Maintain complete actuarial calculation accuracy
  - Results identical to sequential processing within 0.01% tolerance
  - All regulatory compliance requirements preserved
  - Audit trail and calculation transparency maintained

### 8.2 Implementation Deliverables

#### 8.2.1 Code Deliverables
- **DELIVERABLE-001**: Complete optimized SAS program implementing all requirements
- **DELIVERABLE-002**: Configuration macros for dynamic infrastructure adaptation
- **DELIVERABLE-003**: Validation and testing framework for quality assurance
- **DELIVERABLE-004**: Performance monitoring and reporting utilities

#### 8.2.2 Documentation Deliverables
- **DELIVERABLE-005**: Implementation guide with configuration examples
- **DELIVERABLE-006**: Performance benchmarking report with detailed analysis
- **DELIVERABLE-007**: Mathematical validation report confirming calculation accuracy
- **DELIVERABLE-008**: Operational procedures for production deployment

### 8.3 Validation Deliverables
- **DELIVERABLE-009**: Test results across all infrastructure configurations
- **DELIVERABLE-010**: Performance comparison analysis (baseline vs. optimized)
- **DELIVERABLE-011**: Resource utilization efficiency analysis
- **DELIVERABLE-012**: Production deployment readiness assessment

---

## 9. APPENDICES

### Appendix A: Performance Targets by Infrastructure

| Infrastructure | Cores | Memory | Target Runtime | Improvement | Resource Utilization |
|---------------|-------|--------|----------------|-------------|---------------------|
| **Production** | 12 | 96GB | 2.5 minutes | 6.8x | 50% of hardware |
| **Test** | 24 | 192GB | 1.25 minutes | 13.6x | 90% of hardware |

### Appendix B: Computational Scale Reference

| Calculation Tier | Scale | Purpose | Implementation |
|------------------|-------|---------|----------------|
| **External** | 1.01M projections | Economic scenarios | Parallel batches |
| **Reserve** | 10.2B calculations | Standard assumptions | Vectorized processing |
| **Capital** | 10.2B calculations | 35% shocked scenarios | Vectorized processing |
| **Total** | ~20.4B calculations | Complete ACFC model | Distributed processing |

### Appendix C: Key Mathematical Formulas

**Fund Value Evolution:**
```
MT_VM(t+1) = MT_VM(t) × [1 + RENDEMENT(s,t) - PC_REVENU_FDS - FRAIS_ADJ(t)]
```

**Survival Probability:**
```
TX_SURVIE(t+1) = TX_SURVIE(t) × [1 - Qx(age+t)] × [1 - WX(t)]
```

**Distributable Cash Flows:**
```
VP_FLUX_DISTRIBUABLES = Σ [PROFIT(t) + ΔCAPITAL(t)] / (1 + HURDLE_RT)^t
```

---

**Document Status**: Final v2.0  
**Implementation Target**: Complete support for [sas_code_optimized.md](sas_code_optimized.md)  
**Next Phase**: Implementation and testing execution