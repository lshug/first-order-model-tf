# Ops

This file lists the TF Lite ops that are used in the three modules post-conversion assuming static batch size, prediction-only build with process_kp_driving hardcoded with 11, along with the notes about the ops' compatiblity with TF Lite's delegates. When TF Lite interpreter is using a delegate and encounters a non-compatible op during runtime, it is forced to switch to CPU, which is quite expensive in terms of performance and memory, so avoiding such ops is preferrable. For this reason, I'm providing small guides to converting certain incompatible ops into series of compatible ops.

It should be noted that only non-static uses of non-compatible ops need to be converted. For example, using non-compatible ops to generate tensors in keras layers' build functions should be fine. An example of this is the use of ```make_coordinate_grid```, which uses casts and tiling, in some custom layers' build functions.

### Legend

 * -: not supported by delegates
 * ~: not supported by coreml delegate, but supported by others
 * *: but could be replaced with ops that are supported
 * @: would need CAST to be convertible
 * #: would need SQRT to be convertible
 * %: would need DIV to be convertible
 * !: not required when hardcoding with 10
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

 * TILE -
 * AVERAGE_POOL_2D
 * GATHER -
 * MUL
 * CONV_2D
 * RESHAPE
 * PAD
 * TRANSPOSE -
 * CONCATENATION
 * SOFTMAX
 * STRIDED_SLICE
 * SUM -
 
## generator

 * GATHER -
 * CONV_2D
 * EXP ~
 * DIV -
 * FULLY_CONNECTED
 * AVERAGE_POOL_2D
 * MUL
 * RESHAPE
 * SUM -
 * SOFTMAX
 * LESS -
 * CAST - @
 * SUB 
 * GREATER -
 * PAD
 * FLOOR - @ [7]
 * TILE -
 * ADD
 * PACK - * [2]
 * TRANSPOSE -
 * CONCATENATION
 * LOGISTIC
 * STRIDED_SLICE

## process_kp_driving

 * GATHER - !
 * ARG_MIN - !
 * SQRT - !
 * DIV -
 * MUL
 * RESHAPE
 * RSQRT - # %
 * FLOOR_MOD - @ % ! [8]
 * LESS - !
 * SUB
 * SELECT_V2 - @ # ! [6]
 * GREATER ! -
 * TILE ! -
 * ADD
 * SELECT - # @ ! [6]
 * LOGICAL_AND - @ ! [9]
 * FULLY_CONNECTED
 * PACK - * ! [2]
 * CONCATENATION
 * STRIDED_SLICE
