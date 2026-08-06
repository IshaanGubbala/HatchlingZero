#![forbid(unsafe_code)]

//! Backend-neutral, fixed-shape MoE dispatch planning for HZ-0E.
//! Expert execution is intentionally separate: this crate defines the stable
//! queue and fallback contract that MLX and Metal implementations consume.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchPlan {
    pub capacity: usize,
    pub expert_index: Vec<usize>,
    pub rank: Vec<usize>,
    pub accepted: Vec<bool>,
    pub overflow: Vec<bool>,
    /// Flattened expert/slot index per token; `usize::MAX` means fallback.
    pub dispatch_slot: Vec<usize>,
    /// Row-major `[expert, slot]`; `usize::MAX` is padding.
    pub grouped_tokens: Vec<usize>,
}

impl DispatchPlan {
    pub fn token_count(&self) -> usize {
        self.expert_index.len()
    }

    pub fn grouped_token(&self, expert: usize, slot: usize) -> usize {
        self.grouped_tokens[expert * self.capacity + slot]
    }

    /// Validate the complete host-side routing invariant before a backend
    /// uploads or executes any plan buffers.
    pub fn validate(&self, num_experts: usize) -> Result<(), String> {
        if num_experts == 0 || self.capacity == 0 {
            return Err("dispatch plan dimensions must be positive".into());
        }
        if self
            .expert_index
            .iter()
            .any(|&expert| expert >= num_experts)
        {
            return Err("dispatch plan contains an out-of-range expert".into());
        }
        let grouped_len = num_experts
            .checked_mul(self.capacity)
            .ok_or_else(|| "dispatch plan queue size overflow".to_string())?;
        if self.grouped_tokens.len() != grouped_len
            || self.dispatch_slot.len() != self.token_count()
            || self.rank.len() != self.token_count()
            || self.accepted.len() != self.token_count()
            || self.overflow.len() != self.token_count()
        {
            return Err("dispatch plan buffer shape mismatch".into());
        }
        for (token, &slot) in self.dispatch_slot.iter().enumerate() {
            if slot == usize::MAX {
                if !self.overflow[token] || self.accepted[token] {
                    return Err("overflow dispatch metadata is inconsistent".into());
                }
                continue;
            }
            if slot >= grouped_len || self.grouped_tokens[slot] != token {
                return Err("dispatch plan token/slot mapping is inconsistent".into());
            }
            let expert = slot / self.capacity;
            let rank = slot % self.capacity;
            if expert != self.expert_index[token] || rank != self.rank[token] {
                return Err("dispatch plan expert/rank metadata is inconsistent".into());
            }
            if !self.accepted[token] || self.overflow[token] {
                return Err("accepted dispatch metadata is inconsistent".into());
            }
        }
        Ok(())
    }
}

pub fn build_dispatch_plan(
    expert_index: &[usize],
    num_experts: usize,
    capacity_factor: f32,
) -> Result<DispatchPlan, String> {
    if num_experts == 0 {
        return Err("num_experts must be positive".into());
    }
    if !capacity_factor.is_finite() || capacity_factor <= 0.0 {
        return Err("capacity_factor must be finite and positive".into());
    }
    if expert_index.iter().any(|&e| e >= num_experts) {
        return Err("expert_index contains an out-of-range expert".into());
    }
    let n = expert_index.len();
    let raw_capacity = capacity_factor * n as f32 / num_experts as f32;
    if !raw_capacity.is_finite() || raw_capacity > usize::MAX as f32 {
        return Err("capacity calculation became non-finite".into());
    }
    let capacity = (raw_capacity.ceil() as usize).max(1);
    let mut counts = vec![0usize; num_experts];
    let mut rank = Vec::with_capacity(n);
    let mut accepted = Vec::with_capacity(n);
    let mut overflow = Vec::with_capacity(n);
    let mut dispatch_slot = Vec::with_capacity(n);
    let queue_size = num_experts
        .checked_mul(capacity)
        .ok_or_else(|| "dispatch queue size overflow".to_string())?;
    let mut grouped_tokens = vec![usize::MAX; queue_size];
    for (token, &expert) in expert_index.iter().enumerate() {
        let position = counts[expert];
        counts[expert] += 1;
        let keep = position < capacity;
        rank.push(position);
        accepted.push(keep);
        overflow.push(!keep);
        dispatch_slot.push(if keep {
            expert * capacity + position
        } else {
            usize::MAX
        });
        if keep {
            grouped_tokens[expert * capacity + position] = token;
        }
    }
    Ok(DispatchPlan {
        capacity,
        expert_index: expert_index.to_vec(),
        rank,
        accepted,
        overflow,
        dispatch_slot,
        grouped_tokens,
    })
}

/// Convert row-major router logits `[tokens, experts]` into the deterministic
/// top-1 dispatch contract. Softmax is evaluated stably so the returned gate
/// weights are suitable for scaling the selected expert output.
pub fn build_top1_dispatch_plan_f32(
    router_logits: &[f32],
    tokens: usize,
    num_experts: usize,
    capacity_factor: f32,
) -> Result<(DispatchPlan, Vec<f32>), String> {
    let expected = tokens
        .checked_mul(num_experts)
        .ok_or("router logits size overflow")?;
    if num_experts == 0 || router_logits.len() != expected {
        return Err("router logits shape mismatch".into());
    }
    if !router_logits.iter().all(|value| value.is_finite()) {
        return Err("router logits contain non-finite values".into());
    }
    let mut experts = Vec::with_capacity(tokens);
    let mut gates = Vec::with_capacity(tokens);
    for row in router_logits.chunks_exact(num_experts) {
        let mut chosen = 0usize;
        let mut max_logit = row[0];
        for (expert, &logit) in row.iter().enumerate().skip(1) {
            if logit > max_logit {
                chosen = expert;
                max_logit = logit;
            }
        }
        let mut denominator = 0.0f32;
        for &logit in row {
            denominator += (logit - max_logit).exp();
        }
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err("router softmax became non-finite".into());
        }
        experts.push(chosen);
        gates.push(1.0 / denominator);
    }
    Ok((
        build_dispatch_plan(&experts, num_experts, capacity_factor)?,
        gates,
    ))
}

/// Scatter `[expert, capacity, width]` outputs back to `[token, width]`.
/// The fallback buffer supplies overflow rows and is also used as the initial
/// result, making padding entries and overflow behavior explicit.
pub fn scatter_expert_outputs_f32(
    plan: &DispatchPlan,
    expert_outputs: &[f32],
    fallback_outputs: &[f32],
    width: usize,
) -> Result<Vec<f32>, String> {
    if width == 0 {
        return Err("width must be positive".into());
    }
    let grouped_len = plan
        .grouped_tokens
        .len()
        .checked_mul(width)
        .ok_or_else(|| "expert output size overflow".to_string())?;
    let fallback_len = plan
        .token_count()
        .checked_mul(width)
        .ok_or_else(|| "fallback output size overflow".to_string())?;
    if expert_outputs.len() != grouped_len || fallback_outputs.len() != fallback_len {
        return Err("output buffer shape does not match dispatch plan".into());
    }
    let mut output = fallback_outputs.to_vec();
    for expert in 0..(plan.grouped_tokens.len() / plan.capacity) {
        for slot in 0..plan.capacity {
            let token = plan.grouped_token(expert, slot);
            if token == usize::MAX {
                continue;
            }
            let source = (expert * plan.capacity + slot) * width;
            let destination = token * width;
            output[destination..destination + width]
                .copy_from_slice(&expert_outputs[source..source + width]);
        }
    }
    Ok(output)
}

/// Row-major SwiGLU expert execution for the fixed dispatch contract. This is
/// the dependency-free reference stage before adding a fused Metal expert
/// kernel. Weights are `[out, in]`; inputs are `[tokens, dim]`.
#[allow(clippy::too_many_arguments)]
pub fn routed_swiglu_f32(
    plan: &DispatchPlan,
    input: &[f32],
    dim: usize,
    expert_d_ff: usize,
    expert_weights: &[f32],
    expert_biases: &[f32],
    fallback_weights: &[f32],
    fallback_biases: &[f32],
) -> Result<Vec<f32>, String> {
    if dim == 0 || expert_d_ff == 0 {
        return Err("expert dimensions must be positive".into());
    }
    let tokens = plan.token_count();
    let input_size = tokens.checked_mul(dim).ok_or("input size overflow")?;
    if input.len() != input_size {
        return Err("input shape does not match dispatch plan".into());
    }
    let finite = |values: &[f32]| values.iter().all(|value| value.is_finite());
    if !finite(input) {
        return Err("input contains non-finite values".into());
    }
    let expert_weight_width = 3 * expert_d_ff * dim;
    let expert_bias_width = 2 * expert_d_ff + dim;
    let fallback_weight_width = 3 * dim * dim;
    let fallback_bias_width = 3 * dim;
    let expert_count = plan.grouped_tokens.len() / plan.capacity;
    let expert_weights_size = expert_count
        .checked_mul(expert_weight_width)
        .ok_or("expert weight size overflow")?;
    let expert_biases_size = expert_count
        .checked_mul(expert_bias_width)
        .ok_or("expert bias size overflow")?;
    if expert_weights.len() != expert_weights_size {
        return Err("expert weight shape mismatch".into());
    }
    if expert_biases.len() != expert_biases_size
        || fallback_weights.len() != fallback_weight_width
        || fallback_biases.len() != fallback_bias_width
    {
        return Err("expert bias or fallback shape mismatch".into());
    }
    if !finite(expert_weights)
        || !finite(expert_biases)
        || !finite(fallback_weights)
        || !finite(fallback_biases)
    {
        return Err("expert parameters contain non-finite values".into());
    }
    let mut output = vec![0.0; tokens * dim];
    let swish = |value: f32| value / (1.0 + (-value).exp());
    let run =
        |x: &[f32], weights: &[f32], biases: &[f32], d_ff: usize| -> Result<Vec<f32>, String> {
            let (gate_w, rest) = weights.split_at(d_ff * dim);
            let (up_w, down_w) = rest.split_at(d_ff * dim);
            let (gate_b, rest_b) = biases.split_at(d_ff);
            let (up_b, down_b) = rest_b.split_at(d_ff);
            let mut hidden = vec![0.0; d_ff];
            for j in 0..d_ff {
                let mut gate = gate_b[j];
                let mut up = up_b[j];
                for i in 0..dim {
                    gate += gate_w[j * dim + i] * x[i];
                    up += up_w[j * dim + i] * x[i];
                }
                hidden[j] = swish(gate) * up;
            }
            let mut y = down_b.to_vec();
            for o in 0..dim {
                for j in 0..d_ff {
                    y[o] += down_w[o * d_ff + j] * hidden[j];
                }
            }
            if !finite(&y) {
                return Err("expert execution produced non-finite values".into());
            }
            Ok(y)
        };
    for token in 0..tokens {
        let x = &input[token * dim..(token + 1) * dim];
        let y = if plan.dispatch_slot[token] == usize::MAX {
            run(x, fallback_weights, fallback_biases, dim)?
        } else {
            let slot = plan.dispatch_slot[token];
            let expert = slot / plan.capacity;
            if expert >= expert_count {
                return Err("dispatch slot references an invalid expert".into());
            }
            let weight_base = expert * expert_weight_width;
            let bias_base = expert * expert_bias_width;
            run(
                x,
                &expert_weights[weight_base..weight_base + expert_weight_width],
                &expert_biases[bias_base..bias_base + expert_bias_width],
                expert_d_ff,
            )?
        };
        output[token * dim..(token + 1) * dim].copy_from_slice(&y);
    }
    Ok(output)
}

/// Complete backend-neutral top-1 MoE forward: router logits, capacity
/// dispatch, expert/fallback SwiGLU execution, and router-gated outputs.
#[allow(clippy::too_many_arguments)]
pub fn routed_moe_swiglu_f32(
    router_logits: &[f32],
    input: &[f32],
    tokens: usize,
    dim: usize,
    num_experts: usize,
    expert_d_ff: usize,
    capacity_factor: f32,
    expert_weights: &[f32],
    expert_biases: &[f32],
    fallback_weights: &[f32],
    fallback_biases: &[f32],
) -> Result<(DispatchPlan, Vec<f32>), String> {
    let (plan, gates) =
        build_top1_dispatch_plan_f32(router_logits, tokens, num_experts, capacity_factor)?;
    let mut output = routed_swiglu_f32(
        &plan,
        input,
        dim,
        expert_d_ff,
        expert_weights,
        expert_biases,
        fallback_weights,
        fallback_biases,
    )?;
    for (token, gate) in gates.iter().enumerate() {
        if !plan.overflow[token] {
            for value in &mut output[token * dim..(token + 1) * dim] {
                *value *= *gate;
            }
        }
    }
    if !output.iter().all(|value| value.is_finite()) {
        return Err("routed MoE output contains non-finite values".into());
    }
    Ok((plan, output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_token_order_and_bounded_queues() {
        let p = build_dispatch_plan(&[1, 0, 1, 1, 0, 2, 1], 3, 1.0).unwrap();
        assert_eq!(p.capacity, 3);
        assert_eq!(p.rank, vec![0, 0, 1, 2, 1, 0, 3]);
        assert_eq!(
            p.overflow,
            vec![false, false, false, false, false, false, true]
        );
        assert_eq!(p.dispatch_slot, vec![3, 0, 4, 5, 1, 6, usize::MAX]);
        assert_eq!(
            p.grouped_tokens,
            vec![1, 4, usize::MAX, 0, 2, 3, 5, usize::MAX, usize::MAX]
        );
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        assert!(build_dispatch_plan(&[0, 2], 2, 1.0).is_err());
        assert!(build_dispatch_plan(&[0], 0, 1.0).is_err());
        assert!(build_dispatch_plan(&[0], 1, 0.0).is_err());
        assert!(build_dispatch_plan(&[0], 1, f32::MAX).is_err());
    }

    #[test]
    fn plan_validation_rejects_expert_and_rank_drift() {
        let mut plan = build_dispatch_plan(&[0, 1], 2, 1.0).unwrap();
        plan.expert_index[0] = 1;
        assert!(plan.validate(2).is_err());
        let mut plan = build_dispatch_plan(&[0, 1], 2, 1.0).unwrap();
        plan.rank[1] = 1;
        assert!(plan.validate(2).is_err());
    }

    #[test]
    fn scatter_preserves_fallback_for_overflow() {
        let plan = build_dispatch_plan(&[0, 0, 0], 1, 2.0 / 3.0).unwrap();
        let expert = [10.0, 20.0];
        let fallback = [1.0, 2.0, 3.0];
        assert_eq!(
            scatter_expert_outputs_f32(&plan, &expert, &fallback, 1).unwrap(),
            vec![10.0, 20.0, 3.0]
        );
        assert!(scatter_expert_outputs_f32(&plan, &expert, &fallback, 2).is_err());
    }

    #[test]
    fn every_expert_has_an_independent_capacity_bound() {
        let plan = build_dispatch_plan(&[0, 0, 0, 1, 1, 1], 2, 0.5).unwrap();
        assert_eq!(plan.capacity, 2);
        assert_eq!(plan.dispatch_slot, vec![0, 1, usize::MAX, 2, 3, usize::MAX]);
        let expert = [10.0, 11.0, 20.0, 21.0];
        let fallback = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(
            scatter_expert_outputs_f32(&plan, &expert, &fallback, 1).unwrap(),
            vec![10.0, 11.0, 3.0, 20.0, 21.0, 6.0]
        );
    }

    #[test]
    fn routed_swiglu_executes_experts_and_fallback_in_token_order() {
        let plan = build_dispatch_plan(&[0, 0, 1], 2, 2.0 / 3.0).unwrap();
        // dim=d_ff=1, zero weights, distinct gate/up/down biases per route.
        let expert_weights = vec![0.0; 2 * 3];
        let expert_biases = vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0];
        let fallback_weights = vec![0.0; 3];
        let fallback_biases = vec![3.0, 4.0, 5.0];
        let output = routed_swiglu_f32(
            &plan,
            &[0.0, 0.0, 0.0],
            1,
            1,
            &expert_weights,
            &expert_biases,
            &fallback_weights,
            &fallback_biases,
        )
        .unwrap();
        assert!(output[0] < output[2] && output[2] < output[1]);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn routed_swiglu_rejects_non_finite_inputs_or_parameters() {
        let plan = build_dispatch_plan(&[0], 1, 1.0).unwrap();
        let weights = vec![0.0; 3];
        let biases = vec![0.0; 3];
        assert!(routed_swiglu_f32(
            &plan,
            &[f32::NAN],
            1,
            1,
            &weights,
            &biases,
            &weights,
            &biases,
        )
        .is_err());
        let mut bad_weights = weights.clone();
        bad_weights[0] = f32::INFINITY;
        assert!(routed_swiglu_f32(
            &plan,
            &[0.0],
            1,
            1,
            &bad_weights,
            &biases,
            &weights,
            &biases,
        )
        .is_err());
        let huge_weights = vec![1.0e38; 3];
        assert!(routed_swiglu_f32(
            &plan,
            &[1.0],
            1,
            1,
            &huge_weights,
            &biases,
            &weights,
            &biases,
        )
        .is_err());
    }

    #[test]
    fn routed_swiglu_rejects_malformed_parameter_shapes() {
        let plan = build_dispatch_plan(&[0], 1, 1.0).unwrap();
        let weights = vec![0.0; 3];
        let biases = vec![0.0; 3];
        assert!(
            routed_swiglu_f32(&plan, &[], 1, 1, &weights, &biases, &weights, &biases,).is_err()
        );
        assert!(
            routed_swiglu_f32(&plan, &[0.0], 1, 1, &[0.0, 0.0], &biases, &weights, &biases,)
                .is_err()
        );
    }

    #[test]
    fn router_logits_flow_through_capacity_gate_and_fallback() {
        let (plan, output) = routed_moe_swiglu_f32(
            &[2.0, 0.0, 2.0, 0.0],
            &[0.0, 0.0],
            2,
            1,
            2,
            1,
            0.5,
            &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
            &[0.0, 0.0, 1.0],
            &[0.0, 0.0, 5.0],
        )
        .unwrap();
        assert_eq!(plan.expert_index, vec![0, 0]);
        assert_eq!(plan.overflow, vec![false, true]);
        let gate = 2.0f32.exp() / (2.0f32.exp() + 1.0);
        assert!((output[0] - 4.462117 * gate).abs() < 1e-5);
        assert!((output[1] - 5.0).abs() < 1e-5);
    }
}
