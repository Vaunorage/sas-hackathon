# Project Inception: Actuarial Cash Flow Optimization

## Background

Once upon a time (just joking)...

Beneva is running actuarial cash flow calculations that are heavy on the SAS compute infrastructure. A proof of concept (POC) is undergoing to move this calculation load from the SAS compute infrastructure to a more modern and performant one.

## Platform Options Under Evaluation

Among the options being considered are:

### 1. **SAS Viya Platform** 
- **Provider**: SAS Institute
- **Technology**: Modern cloud-native analytics platform
- **Status**: Being tested during SAS Institute hackathon

### 2. **sas2c Project**
- **Technology**: C-based project using CUDA cores
- **Performance**: Already proved its value by running a test set **30 times faster** than the current SAS compute architecture
- **Test Set Limitation**: Was tested with a different, simpler test set than the new hackathon version
- **Status**: Requires significant upgrades to handle the new complex hackathon test set

### 3. **PathWise Platform**
- **Provider**: Third-party platform
- **Technology**: Specialized actuarial computation platform
- **Status**: Under evaluation

## Hackathon Opportunity

SAS Institute is hosting a hackathon giving Beneva the opportunity to test its actuarial cash flow calculations on **SAS Viya**. For the hackathon, a brand new test set has been created.

### Test Set Complexity
The hackathon test set is **significantly more complex** than the original test set used by sas2c. This increased complexity means that:
- Previous performance benchmarks (like sas2c's 30x improvement) may not directly apply
- All platform options will need to demonstrate capabilities on this more challenging calculation set
- The sas2c project will require substantial upgrades to handle the new complexity level

### Baseline Establishment

Running the test set on the actual SAS compute infrastructure will be the **baseline for comparisons** with the results of all platform options.

## Project Goals

This project is an attempt to **optimize the baseline SAS program** so that it is at its best to exploit the current SAS compute infrastructure.

### Key Objectives

1. **Performance Baseline**: Establish the optimal performance benchmark for the current SAS infrastructure
2. **Fair Comparison**: Having the test set optimized will allow observation of how much more performant each of the platform options truly are
3. **Infrastructure Maximization**: Ensure the existing SAS environment is utilized to its fullest potential before migration decisions

## Success Metrics

- **Performance Improvement**: Measure speed gains from optimization efforts
- **Resource Utilization**: Maximize CPU, memory, and I/O efficiency on current infrastructure  
- **Data Accuracy**: Ensure mathematical precision and numerical consistency across all optimizations
- **Data Integrity**: Maintain complete data validation and regulatory compliance throughout processing
- **Comparative Analysis**: Provide fair baseline for evaluating alternative platforms
- **Migration Readiness**: Prepare optimized calculations for potential platform migration

## Strategic Impact

By optimizing the baseline SAS implementation:
- Beneva can make informed decisions about infrastructure investments
- The true performance gap between current and future platforms becomes clear
- Migration costs and benefits can be accurately assessed
- Current infrastructure value is maximized during transition period

---

**Project Status**: Active Development  
**Target Platform**: SAS Compute Infrastructure (Current)  
**Future Platforms**: SAS Viya, sas2c, PathWise  
**Performance Benchmark**: 30x improvement demonstrated by sas2c