# Ops

### Legend

This file lists the TF Lite ops that are used in the three modules post-conversion, along with the notes about the ops' compatiblity with TF Lite's delegates. When TF Lite interpreter is using a delegate and encounters a non-compatible op during runtime, it is forced to switch to CPU, which is quite expensive in terms of performance and memory, so avoiding such ops is preferrable. For this reason, I'm providing small guides to converting certain incompatible ops into series of compatible ops. Unfortunately there are certain ops that are necessary for the model and that cannot realistically be converted in such manner, these being: tensor tiling, gather_nd, batch matrix multiplication, dtype casting, taking a floor, summing, division, square root, logical operations and comparisons on bool tensors, ternary select, tensor transposition, and argmin.

 * -: not supported by delegates
 * ~: not supported by coreml delegate, but supported by others
 * *: but could be replaced with ops that are supported
 * @: would need CAST to be convertible
 * #: would need SQRT to be convertible
 * %: would need DIV to be convertible
 * [n]: explanations of how the replacement can be done
    * [1]: can be replaced by an equivalent reshape
    * [2]: expand shape with reshape, pack with concatenate
    * [3]: only non-static uses are on the batch dimension, can be replaced with a static 1 for fixed batch size of 1
    * [4]: can be replaced with multiplying the arg by zero
    * [5]: can be replaced with multiplying the arg by itself
    * [6]: can be reimplementing with casting, square root, addition, and multiplication, see bottom
    * [7]: with casting, ```tf.floor(x)``` can be converted to ```tf.cast(tf.cast(x, 'int32'), x.dtype)```
    * [8]: with casting, converted tf.floor (see [7]) and division, ```a % b``` can be converted to ```b * (((a / b) - tf.floor((a / b))))```
    * [9]: with casting, ```a & b``` can be converted as ```tf.cast(tf.cast(a, 'int32') * tf.cast(b, 'int32'), 'bool')```
 
## kp_detector

 * EXPAND_DIMS - * [1]
 * TILE -
 * AVERAGE_POOL_2D
 * GATHER_ND -
 * MUL
 * CONV_2D
 * RESHAPE
 * PAD
 * PACK - * [2]
 * TRANSPOSE -
 * CONCATENATION
 * SOFTMAX
 * STRIDED_SLICE
 * SUM -
 * SHAPE * [3]
 * RANGE * [3]

## generator

 * GATHER_ND -
 * CONV_2D
 * EXP ~
 * DIV -
 * BATCH_MATMUL -
 * RANGE * [3]
 * ZEROS_LIKE * [4]
 * AVERAGE_POOL_2D
 * MUL
 * RESHAPE
 * SQUARE - * [5]
 * SUM -
 * SOFTMAX
 * LESS -
 * CAST - @
 * SUB 
 * EXPAND_DIMS - * [1]
 * SELECT_V2 - @ # [6]
 * GREATER -
 * PAD
 * FLOOR - @ [7]
 * TILE -
 * ADD
 * LOGICAL_AND - @ [9]
 * PACK - * [2]
 * TRANSPOSE -
 * CONCATENATION
 * LOGISTIC
 * STRIDED_SLICE
 * SHAPE * [3]

## process_kp_driving

 * GATHER_ND -
 * ARG_MIN -
 * SQRT -
 * DIV -
 * BATCH_MATMUL -
 * MUL
 * RESHAPE
 * RSQRT - # %
 * FLOOR_MOD - @ % [8]
 * LESS -
 * CAST - @
 * SUB
 * SELECT_V2 - @ # [6]
 * GREATER -
 * TILE -
 * ADD
 * SELECT - # @ [6]
 * LOGICAL_AND - @ [9]
 * FULLY_CONNECTED
 * PACK - * [2]
 * CONCATENATION
 * STRIDED_SLICE
 * SHAPE - * [3]

## Arithmetic tf.where

If casting and square root were allowed, ```tf.where(conditional_bool, leftarg, rightarg)``` could be converted as follows (in pseudocode):

 * Let ```cond1 := tf.cast(conditional_bool, leftarg.dtype)```
 * Let ```cond2 := tf.sqrt((cond1 - tf.cast(1, cond1.dtype)) * (cond1 - tf.cast(1, rightarg.dtype)))``` (arithmetic NOT of cond1)
 * ```out=cond1 * leftarg + cond2 * rightarg```
