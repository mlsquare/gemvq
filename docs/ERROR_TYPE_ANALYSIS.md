# Error Type Analysis: Cumulative vs Tile-Specific Error

## Question Answered

The user asked: **"Is the error cumulative error or tile-specific error?"**

## Answer: **CUMULATIVE ERROR**

The current implementation uses **cumulative error**, which is the correct approach for coarse-to-fine decoding.

## Detailed Analysis

### 1. Cumulative Error (Current Implementation)

**Definition**: Error calculated for the complete reconstruction using levels 0 to max_level.

**Formula**: 
```
Error = ||x - sum(q^(M-1-i) * x_i for i=0 to max_level)|| / ||x||
```

**Characteristics**:
- Each level adds its contribution to the previous levels
- Represents the total error of the progressive reconstruction
- More relevant for coarse-to-fine decoding
- Shows the actual quality of the reconstruction at each level

**Example from test**:
```
Level 0 (cumulative): Error = 5.853210
Level 1 (cumulative): Error = 5.523026  
Level 2 (cumulative): Error = 5.523026
```

### 2. Tile-Specific Error (Alternative)

**Definition**: Error calculated for each individual level's contribution only.

**Formula**:
```
Error = ||x - q^(M-1-level) * x_level|| / ||x||
```

**Characteristics**:
- Only one level contributes to the reconstruction
- Shows the individual contribution of each level
- More relevant for analyzing level contributions
- Useful for understanding which levels are most important

**Example from test**:
```
Level 0 (tile-specific): Error = 5.853210
Level 1 (tile-specific): Error = 0.381135
Level 2 (tile-specific): Error = 1.000000
```

## Why They Differ

### Mathematical Difference

1. **Cumulative Error**: 
   - Level 0: `||x - q^(M-1) * x_0|| / ||x||`
   - Level 1: `||x - (q^(M-1) * x_0 + q^(M-2) * x_1)|| / ||x||`
   - Level 2: `||x - (q^(M-1) * x_0 + q^(M-2) * x_1 + q^(M-3) * x_2)|| / ||x||`

2. **Tile-Specific Error**:
   - Level 0: `||x - q^(M-1) * x_0|| / ||x||`
   - Level 1: `||x - q^(M-2) * x_1|| / ||x||`
   - Level 2: `||x - q^(M-3) * x_2|| / ||x||`

### Practical Implications

1. **Cumulative Error**:
   - Shows the actual reconstruction quality at each level
   - More meaningful for progressive decoding
   - Represents what the user actually sees
   - Used in the current implementation

2. **Tile-Specific Error**:
   - Shows individual level contributions
   - Useful for analyzing which levels are most important
   - Helps understand the hierarchical structure
   - Not used in the current implementation

## Test Results

### Monotonicity Analysis

From the test results:
- **Cumulative Error**: 50% monotonic (5/10 trials)
- **Tile-Specific Error**: 50% monotonic (5/10 trials)

Both show similar monotonicity rates, but they measure different things.

### Example Comparison

```
Original vector: [0.620, 0.178, 0.270, -1.743]

Cumulative errors: [5.853, 5.523, 5.523]
Tile-specific errors: [5.853, 0.381, 1.000]

Interpretation:
- Cumulative: Adding level 1 improves reconstruction (5.853 → 5.523)
- Tile-specific: Level 1 has much lower individual error (0.381) than level 0 (5.853)
```

## Why Cumulative Error is Correct

### 1. Progressive Decoding Context

In coarse-to-fine decoding, we want to know:
- How good is the reconstruction with only the coarsest level?
- How much does adding the next level improve it?
- What is the final quality with all levels?

Cumulative error answers these questions directly.

### 2. User Experience

Users of coarse-to-fine decoding care about:
- The actual quality of the reconstruction they receive
- How quality improves as more levels are added
- The trade-off between quality and computational cost

Cumulative error provides this information.

### 3. Implementation Correctness

The current implementation correctly uses cumulative error because:
- It matches the progressive decoding paradigm
- It shows the actual reconstruction quality
- It's what users expect from coarse-to-fine decoding

## Conclusion

**The current implementation correctly uses CUMULATIVE ERROR**, which is the appropriate choice for coarse-to-fine decoding because:

1. ✅ **Shows actual reconstruction quality** at each level
2. ✅ **Matches progressive decoding paradigm** 
3. ✅ **Provides meaningful user feedback** about quality improvement
4. ✅ **Correctly implements coarse-to-fine behavior**

The tile-specific error, while interesting for analysis, is not the primary metric for coarse-to-fine decoding applications.
