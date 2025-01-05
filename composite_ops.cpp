#include "nn.h"

tensor* log_softmax(tensor* logits){
    // min-max trick for numerical stability, python: "logits -= np.max(logits, axis=1, keepdims=True)"
    int n_repeats = logits->shape[1];
    tensor* maxes = repeat(batched_reduce_max(logits), /*axis = */ 1, /*num_repeats = */ n_repeats);
    set_name(maxes, "maxes"); // sprint(maxes);
    tensor* su = sub(logits, maxes);
    set_name(su, "su"); // sprint(su);

    tensor* ex = exp(su);                           // (B, ?)
    set_name(ex, "ex"); // sprint(ex);
    tensor* re = batched_reduce_sum(ex);            // (B, 1)
    set_name(re, "re"); // sprint(denom);

    // https://github.com/pytorch/pytorch/blob/de484134e4700f95a8a9db5b15daf57d28496a6b/aten/src/ATen/native/vulkan/ops/Softmax.cpp#L196-L203
    //
    // note: this is invisible to the generated code, bc uses k_, and not the add op
    add_k_(re, TensorLikeFill(re, 6e-8), re);

    tensor* denom = log(re);                        // (B, 1)
    set_name(denom, "denom"); // sprint(denom);
    // print(denom);
    n_repeats = ex->shape[1];
    tensor* denom_broadcasted = repeat(denom, /*axis = */ 1, /*num_repeats = */ n_repeats);
    set_name(denom_broadcasted, "denom_broadcasted"); // sprint(denom_broadcasted);

    tensor* log_sm = sub(su, denom_broadcasted);    // (B, ?)
    set_name(log_sm, "log_sm"); // sprint(log_sm);
    return log_sm;
}

// expects log probabilities (output of LOGsoftmax) as input
tensor* NLL(tensor* log_probs, tensor* label){
    int B = label->shape[0];
    set_name(label, "label"); // sprint(label);
    tensor* se = select(log_probs, label);      // (B, 1)
    set_name(se, "se"); // sprint(se);
    tensor* lgsum = reduce_sum(se);         // (, )
    set_name(lgsum, "lgsum");  // sprint(lgsum);
    tensor* nll = neg(lgsum);               // (, )
    set_name(nll, "nll"); // sprint(nll);
    // divide by the batch size
    tensor* nll_normalized = div(nll, TensorScalarFill(B)); // (, )
    set_name(nll_normalized, "nll_normalized"); // sprint(nll_normalized);
    return nll_normalized;
}


// comment: log_sofmtax followed by NLL (which expects log probs) -- is the same as "nn.CrossEntropyLoss"
//  - cross entropy takes in raw logits and internally applies softmax
//  - NLL takes log-softmax’ed values as input (log probabilities)
