use hz0e_pmetal_moe::{build_top1_dispatch_plan_f32, DispatchPlan};
use metal::{Device, MTLResourceOptions, MTLSize};

const SOURCE: &str = include_str!("../../../metal/moe_swiglu.metal");

pub struct MetalMoeSwiGlu {
    device: Device,
    pipeline_hidden: metal::ComputePipelineState,
    pipeline_down: metal::ComputePipelineState,
    queue: metal::CommandQueue,
}

impl MetalMoeSwiGlu {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device available")?;
        let library = device
            .new_library_with_source(SOURCE, &metal::CompileOptions::new())
            .map_err(|e| format!("MoE SwiGLU shader compilation failed: {e}"))?;
        let hidden_function = library
            .get_function("hz0e_moe_swiglu_hidden", None)
            .map_err(|e| format!("could not find hz0e_moe_swiglu_hidden: {e}"))?;
        let down_function = library
            .get_function("hz0e_moe_swiglu_down", None)
            .map_err(|e| format!("could not find hz0e_moe_swiglu_down: {e}"))?;
        let pipeline_hidden = device
            .new_compute_pipeline_state_with_function(&hidden_function)
            .map_err(|e| format!("could not build MoE SwiGLU hidden-stage pipeline: {e}"))?;
        let pipeline_down = device
            .new_compute_pipeline_state_with_function(&down_function)
            .map_err(|e| format!("could not build MoE SwiGLU down-stage pipeline: {e}"))?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            pipeline_hidden,
            pipeline_down,
            queue,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input: &[f32],
        dispatch_slot: &[i32],
        expert_weights: &[f32],
        expert_biases: &[f32],
        fallback_weights: &[f32],
        fallback_biases: &[f32],
        experts: usize,
        capacity: usize,
        dim: usize,
        expert_d_ff: usize,
        fallback_d_ff: usize,
    ) -> Result<Vec<f32>, String> {
        if experts == 0 || capacity == 0 || dim == 0 || expert_d_ff == 0 || fallback_d_ff == 0 {
            return Err("MoE SwiGLU dimensions must be positive".into());
        }
        let expert_weights_size = experts
            .checked_mul(
                3usize
                    .checked_mul(expert_d_ff)
                    .and_then(|v| v.checked_mul(dim))
                    .ok_or("expert weight size overflow")?,
            )
            .ok_or("expert weight size overflow")?;
        let expert_biases_size = experts
            .checked_mul(
                2usize
                    .checked_mul(expert_d_ff)
                    .and_then(|v| v.checked_add(dim))
                    .ok_or("expert bias size overflow")?,
            )
            .ok_or("expert bias size overflow")?;
        // `fallback_d_ff` is a REAL, separate hidden width from
        // `expert_d_ff` -- see the Metal shader's own module comment for
        // why this must not be assumed equal to `dim`.
        let fallback_weights_size = 3usize
            .checked_mul(fallback_d_ff)
            .and_then(|v| v.checked_mul(dim))
            .ok_or("fallback weight size overflow")?;
        let fallback_biases_size = 2usize
            .checked_mul(fallback_d_ff)
            .and_then(|v| v.checked_add(dim))
            .ok_or("fallback bias size overflow")?;
        if expert_weights.len() != expert_weights_size
            || expert_biases.len() != expert_biases_size
            || fallback_weights.len() != fallback_weights_size
            || fallback_biases.len() != fallback_biases_size
        {
            return Err("MoE SwiGLU buffer shape mismatch".into());
        }
        let finite = |values: &[f32]| values.iter().all(|value| value.is_finite());
        if !finite(expert_weights)
            || !finite(expert_biases)
            || !finite(fallback_weights)
            || !finite(fallback_biases)
        {
            return Err("MoE SwiGLU input or parameter buffer contains non-finite values".into());
        }
        let opts = MTLResourceOptions::StorageModeShared;
        let f32_buffer = |data: &[f32]| {
            self.device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                (data.len() * 4) as u64,
                opts,
            )
        };
        let expert_weights_buf = f32_buffer(expert_weights);
        let expert_biases_buf = f32_buffer(expert_biases);
        let fallback_weights_buf = f32_buffer(fallback_weights);
        let fallback_biases_buf = f32_buffer(fallback_biases);
        self.dispatch_two_stage(
            input,
            dispatch_slot,
            &expert_weights_buf,
            &expert_biases_buf,
            &fallback_weights_buf,
            &fallback_biases_buf,
            experts,
            capacity,
            dim,
            expert_d_ff,
            fallback_d_ff,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_plan(
        &self,
        plan: &DispatchPlan,
        input: &[f32],
        expert_weights: &[f32],
        expert_biases: &[f32],
        fallback_weights: &[f32],
        fallback_biases: &[f32],
        experts: usize,
        dim: usize,
        expert_d_ff: usize,
        fallback_d_ff: usize,
    ) -> Result<Vec<f32>, String> {
        if dim == 0 || expert_d_ff == 0 || fallback_d_ff == 0 {
            return Err("expert dimensions must be positive".into());
        }
        plan.validate(experts)?;
        let slots = plan
            .dispatch_slot
            .iter()
            .map(|&slot| {
                if slot == usize::MAX {
                    Ok(-1)
                } else {
                    i32::try_from(slot).map_err(|_| "dispatch slot exceeds Metal i32 range")
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.forward(
            input,
            &slots,
            expert_weights,
            expert_biases,
            fallback_weights,
            fallback_biases,
            experts,
            plan.capacity,
            dim,
            expert_d_ff,
            fallback_d_ff,
        )
    }

    /// Execute the native top-1 routing contract with the Metal expert
    /// primitive. Planning remains deterministic and host-side until a
    /// device-side capacity planner is introduced.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_logits(
        &self,
        router_logits: &[f32],
        input: &[f32],
        expert_weights: &[f32],
        expert_biases: &[f32],
        fallback_weights: &[f32],
        fallback_biases: &[f32],
        tokens: usize,
        experts: usize,
        capacity_factor: f32,
        dim: usize,
        expert_d_ff: usize,
        fallback_d_ff: usize,
    ) -> Result<(DispatchPlan, Vec<f32>), String> {
        let (plan, gates) =
            build_top1_dispatch_plan_f32(router_logits, tokens, experts, capacity_factor)?;
        let mut output = self.forward_plan(
            &plan,
            input,
            expert_weights,
            expert_biases,
            fallback_weights,
            fallback_biases,
            experts,
            dim,
            expert_d_ff,
            fallback_d_ff,
        )?;
        for (token, gate) in gates.iter().enumerate() {
            if !plan.overflow[token] {
                for value in &mut output[token * dim..(token + 1) * dim] {
                    *value *= *gate;
                }
            }
        }
        if !output.iter().all(|value| value.is_finite()) {
            return Err("routed Metal MoE output contains non-finite values".into());
        }
        Ok((plan, output))
    }

    /// Uploads expert/fallback weight and bias buffers to the GPU ONCE,
    /// returning a handle that keeps them device-resident across many
    /// forward calls, for callers that dispatch many times against the
    /// same weights (real model integration: same layer's weights, many
    /// token batches). Pair with `forward_logits_cached`/
    /// `forward_plan_cached`.
    ///
    /// NOTE: an earlier version of this module's docs attributed E9's
    /// measured ~40x end-to-end slowdown to this re-upload cost. Direct
    /// isolation testing DISPROVED that: `forward_cached` with weights
    /// already resident measured the SAME per-call cost as the
    /// uncached path. The real cause was the single-stage kernel's
    /// O(dim) redundant recompute (see the Metal shader's own module
    /// comment) -- fixed by the two-stage dispatch both `forward` and
    /// the cached path now share. Weight residency is still a real,
    /// correct, worthwhile improvement (avoids ~42MB of redundant
    /// upload+copy per call at this model's real scale) -- it just was
    /// not, by itself, the dominant cost.
    pub fn upload_weights(
        &self,
        expert_weights: &[f32],
        expert_biases: &[f32],
        fallback_weights: &[f32],
        fallback_biases: &[f32],
        experts: usize,
        dim: usize,
        expert_d_ff: usize,
        fallback_d_ff: usize,
    ) -> Result<CachedMoeWeights, String> {
        if experts == 0 || dim == 0 || expert_d_ff == 0 || fallback_d_ff == 0 {
            return Err("MoE SwiGLU dimensions must be positive".into());
        }
        let expert_weights_size = experts
            .checked_mul(
                3usize
                    .checked_mul(expert_d_ff)
                    .and_then(|v| v.checked_mul(dim))
                    .ok_or("expert weight size overflow")?,
            )
            .ok_or("expert weight size overflow")?;
        let expert_biases_size = experts
            .checked_mul(
                2usize
                    .checked_mul(expert_d_ff)
                    .and_then(|v| v.checked_add(dim))
                    .ok_or("expert bias size overflow")?,
            )
            .ok_or("expert bias size overflow")?;
        let fallback_weights_size = 3usize
            .checked_mul(fallback_d_ff)
            .and_then(|v| v.checked_mul(dim))
            .ok_or("fallback weight size overflow")?;
        let fallback_biases_size = 2usize
            .checked_mul(fallback_d_ff)
            .and_then(|v| v.checked_add(dim))
            .ok_or("fallback bias size overflow")?;
        if expert_weights.len() != expert_weights_size
            || expert_biases.len() != expert_biases_size
            || fallback_weights.len() != fallback_weights_size
            || fallback_biases.len() != fallback_biases_size
        {
            return Err("MoE SwiGLU weight buffer shape mismatch".into());
        }
        let finite = |values: &[f32]| values.iter().all(|value| value.is_finite());
        if !finite(expert_weights)
            || !finite(expert_biases)
            || !finite(fallback_weights)
            || !finite(fallback_biases)
        {
            return Err("MoE SwiGLU weight buffer contains non-finite values".into());
        }
        let opts = MTLResourceOptions::StorageModeShared;
        let f32_buffer = |data: &[f32]| {
            self.device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                (data.len() * 4) as u64,
                opts,
            )
        };
        Ok(CachedMoeWeights {
            expert_weights: f32_buffer(expert_weights),
            expert_biases: f32_buffer(expert_biases),
            fallback_weights: f32_buffer(fallback_weights),
            fallback_biases: f32_buffer(fallback_biases),
            experts,
            dim,
            expert_d_ff,
            fallback_d_ff,
        })
    }

    /// Same dispatch `forward_plan` performs, but reads weight/bias
    /// buffers from an already-resident `CachedMoeWeights` instead of
    /// re-uploading them -- only `input`/`dispatch_slot`/`output`/
    /// `uniform`/`hidden` (all small, tokens-sized) are built per call.
    pub fn forward_plan_cached(
        &self,
        plan: &DispatchPlan,
        input: &[f32],
        weights: &CachedMoeWeights,
    ) -> Result<Vec<f32>, String> {
        plan.validate(weights.experts)?;
        let slots = plan
            .dispatch_slot
            .iter()
            .map(|&slot| {
                if slot == usize::MAX {
                    Ok(-1)
                } else {
                    i32::try_from(slot).map_err(|_| "dispatch slot exceeds Metal i32 range")
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.dispatch_two_stage(
            input,
            &slots,
            &weights.expert_weights,
            &weights.expert_biases,
            &weights.fallback_weights,
            &weights.fallback_biases,
            weights.experts,
            plan.capacity,
            weights.dim,
            weights.expert_d_ff,
            weights.fallback_d_ff,
        )
    }

    /// Same contract as `forward_logits` (native top-1 routing, router
    /// gate scaling, unscaled overflow fallback), but against
    /// already-resident weight buffers.
    pub fn forward_logits_cached(
        &self,
        router_logits: &[f32],
        input: &[f32],
        tokens: usize,
        capacity_factor: f32,
        weights: &CachedMoeWeights,
    ) -> Result<(DispatchPlan, Vec<f32>), String> {
        let (plan, gates) =
            build_top1_dispatch_plan_f32(router_logits, tokens, weights.experts, capacity_factor)?;
        let mut output = self.forward_plan_cached(&plan, input, weights)?;
        for (token, gate) in gates.iter().enumerate() {
            if !plan.overflow[token] {
                for value in &mut output[token * weights.dim..(token + 1) * weights.dim] {
                    *value *= *gate;
                }
            }
        }
        if !output.iter().all(|value| value.is_finite()) {
            return Err("routed Metal MoE output contains non-finite values".into());
        }
        Ok((plan, output))
    }

    /// Real two-dispatch execution shared by every public entry point.
    ///
    /// Stage 1 (`hz0e_moe_swiglu_hidden`) computes each token's SwiGLU
    /// hidden activation ONCE per (token, dff-index) into a `hidden`
    /// scratch buffer. Stage 2 (`hz0e_moe_swiglu_down`) reduces that
    /// hidden activation down to each (token, out) output scalar. Both
    /// run as separate compute-command-encoders within the SAME command
    /// buffer, so Metal's automatic hazard tracking serializes them
    /// (stage 2's reads of `hidden` are guaranteed to see stage 1's
    /// writes) with only ONE `commit`/`wait_until_completed` round trip
    /// -- not two.
    ///
    /// This replaces a prior single-stage kernel that used one thread
    /// per (token, out) output scalar, with each thread independently
    /// recomputing the full per-token hidden activation from scratch --
    /// an O(dim) redundant-recompute blowup (the SAME hidden activation
    /// recomputed once per each of `dim` output threads). Measured
    /// effect of the fix: see `docs/restart/hz0e_e9_pmetal_dispatch_results.md`.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_two_stage(
        &self,
        input: &[f32],
        dispatch_slot: &[i32],
        expert_weights_buf: &metal::Buffer,
        expert_biases_buf: &metal::Buffer,
        fallback_weights_buf: &metal::Buffer,
        fallback_biases_buf: &metal::Buffer,
        experts: usize,
        capacity: usize,
        dim: usize,
        expert_d_ff: usize,
        fallback_d_ff: usize,
    ) -> Result<Vec<f32>, String> {
        if experts == 0 || capacity == 0 || dim == 0 || expert_d_ff == 0 || fallback_d_ff == 0 {
            return Err("MoE SwiGLU dimensions must be positive".into());
        }
        let tokens = dispatch_slot.len();
        let input_size = tokens.checked_mul(dim).ok_or("input size overflow")?;
        if input.len() != input_size {
            return Err("MoE SwiGLU buffer shape mismatch".into());
        }
        let finite = |values: &[f32]| values.iter().all(|value| value.is_finite());
        if !finite(input) {
            return Err("MoE SwiGLU input buffer contains non-finite values".into());
        }
        let queue_size = experts
            .checked_mul(capacity)
            .ok_or("dispatch queue size overflow")?;
        if dispatch_slot
            .iter()
            .any(|&slot| slot >= 0 && slot as usize >= queue_size)
        {
            return Err("MoE SwiGLU dispatch slot is out of range".into());
        }
        let max_d_ff = expert_d_ff.max(fallback_d_ff);
        let hidden_size = tokens.checked_mul(max_d_ff).ok_or("hidden buffer size overflow")?;

        let opts = MTLResourceOptions::StorageModeShared;
        let f32_buffer = |data: &[f32]| {
            self.device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                (data.len() * 4) as u64,
                opts,
            )
        };
        let i32_buffer = |data: &[i32]| {
            self.device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                (data.len() * 4) as u64,
                opts,
            )
        };
        let input_buffer = f32_buffer(input);
        let slot_buffer = i32_buffer(dispatch_slot);
        let hidden_buffer = self.device.new_buffer((hidden_size * 4) as u64, opts);
        let output = self.device.new_buffer((input_size * 4) as u64, opts);
        let uniforms = [
            u32::try_from(capacity).map_err(|_| "expert capacity exceeds Metal u32 range")?,
            u32::try_from(dim).map_err(|_| "expert width exceeds Metal u32 range")?,
            u32::try_from(expert_d_ff)
                .map_err(|_| "expert hidden width exceeds Metal u32 range")?,
            u32::try_from(tokens).map_err(|_| "expert token count exceeds Metal u32 range")?,
            u32::try_from(fallback_d_ff)
                .map_err(|_| "fallback hidden width exceeds Metal u32 range")?,
            u32::try_from(max_d_ff).map_err(|_| "max hidden width exceeds Metal u32 range")?,
        ];
        let uniform = self.device.new_buffer_with_data(
            uniforms.as_ptr() as *const std::ffi::c_void,
            24,
            opts,
        );

        let command = self.queue.new_command_buffer();

        let hidden_encoder = command.new_compute_command_encoder();
        hidden_encoder.set_compute_pipeline_state(&self.pipeline_hidden);
        hidden_encoder.set_buffer(0, Some(&input_buffer), 0);
        hidden_encoder.set_buffer(1, Some(&slot_buffer), 0);
        hidden_encoder.set_buffer(2, Some(expert_weights_buf), 0);
        hidden_encoder.set_buffer(3, Some(expert_biases_buf), 0);
        hidden_encoder.set_buffer(4, Some(fallback_weights_buf), 0);
        hidden_encoder.set_buffer(5, Some(fallback_biases_buf), 0);
        hidden_encoder.set_buffer(6, Some(&hidden_buffer), 0);
        for index in 0..6u64 {
            hidden_encoder.set_buffer(7 + index, Some(&uniform), index * 4);
        }
        hidden_encoder.dispatch_threads(
            MTLSize::new(tokens as u64, max_d_ff as u64, 1),
            MTLSize::new(tokens.min(256).max(1) as u64, 1, 1),
        );
        hidden_encoder.end_encoding();

        let down_encoder = command.new_compute_command_encoder();
        down_encoder.set_compute_pipeline_state(&self.pipeline_down);
        down_encoder.set_buffer(0, Some(&slot_buffer), 0);
        down_encoder.set_buffer(1, Some(expert_weights_buf), 0);
        down_encoder.set_buffer(2, Some(expert_biases_buf), 0);
        down_encoder.set_buffer(3, Some(fallback_weights_buf), 0);
        down_encoder.set_buffer(4, Some(fallback_biases_buf), 0);
        down_encoder.set_buffer(5, Some(&hidden_buffer), 0);
        down_encoder.set_buffer(6, Some(&output), 0);
        for index in 0..6u64 {
            down_encoder.set_buffer(7 + index, Some(&uniform), index * 4);
        }
        down_encoder.dispatch_threads(
            MTLSize::new(tokens as u64, dim as u64, 1),
            MTLSize::new(tokens.min(256).max(1) as u64, 1, 1),
        );
        down_encoder.end_encoding();

        command.commit();
        command.wait_until_completed();
        if command.status() != metal::MTLCommandBufferStatus::Completed {
            return Err(format!("MoE SwiGLU command failed: {:?}", command.status()));
        }
        let result = unsafe {
            std::slice::from_raw_parts(output.contents() as *const f32, input_size).to_vec()
        };
        if !finite(&result) {
            return Err("MoE SwiGLU device output contains non-finite values".into());
        }
        Ok(result)
    }
}

/// Device-resident expert/fallback weight and bias buffers, produced by
/// `MetalMoeSwiGlu::upload_weights` and consumed by
/// `forward_plan_cached`/`forward_logits_cached`. Uploaded once, reused
/// across many forward calls.
pub struct CachedMoeWeights {
    expert_weights: metal::Buffer,
    expert_biases: metal::Buffer,
    fallback_weights: metal::Buffer,
    fallback_biases: metal::Buffer,
    experts: usize,
    dim: usize,
    expert_d_ff: usize,
    fallback_d_ff: usize,
}

impl CachedMoeWeights {
    pub fn experts(&self) -> usize {
        self.experts
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::MetalMoeSwiGlu;

    #[test]
    fn metal_swiglu_matches_expert_fallback_reference() {
        let kernel = MetalMoeSwiGlu::new().expect("Metal device required");
        let output = kernel
            .forward(
                &[0.0, 0.0, 0.0],
                &[0, -1, 1],
                &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
                &[0.0, 0.0, 1.0],
                &[3.0, 4.0, 5.0],
                2,
                1,
                1,
                1,
                1,
            )
            .unwrap();
        assert!((output[0] - 4.462117).abs() < 1e-5);
        assert!((output[1] - 16.430_89).abs() < 1e-5);
        assert!((output[2] - 9.284782).abs() < 1e-5);
    }

    #[test]
    fn metal_swiglu_rejects_non_finite_host_buffers() {
        let kernel = MetalMoeSwiGlu::new().expect("Metal device required");
        let result = kernel.forward(
            &[f32::NAN],
            &[0],
            &[0.0, 0.0, 1.0],
            &[1.0, 2.0, 3.0],
            &[0.0, 0.0, 1.0],
            &[3.0, 4.0, 5.0],
            1,
            1,
            1,
            1,
            1,
        );
        assert!(result.is_err());
    }

    #[test]
    fn metal_swiglu_accepts_canonical_padded_plan() {
        let kernel = MetalMoeSwiGlu::new().expect("Metal device required");
        let plan = hz0e_pmetal_moe::build_dispatch_plan(&[0, 1], 4, 1.5).unwrap();
        let output = kernel
            .forward_plan(
                &plan,
                &[0.0, 0.0],
                &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                &[0.0; 3],
                &[3.0, 4.0, 5.0],
                4,
                1,
                1,
                1,
            )
            .unwrap();
        assert!((output[0] - 4.462117).abs() < 1e-5);
        assert!((output[1] - 9.284782).abs() < 1e-5);
    }

    #[test]
    fn metal_swiglu_accepts_router_logits_and_applies_gate() {
        let kernel = MetalMoeSwiGlu::new().expect("Metal device required");
        let (plan, output) = kernel
            .forward_logits(
                &[2.0, 0.0, 2.0, 0.0],
                &[0.0, 0.0],
                &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
                &[0.0, 0.0, 1.0],
                &[0.0, 0.0, 5.0],
                2,
                2,
                0.5,
                1,
                1,
                1,
            )
            .unwrap();
        assert_eq!(plan.overflow, vec![false, true]);
        let gate = 2.0f32.exp() / (2.0f32.exp() + 1.0);
        assert!((output[0] - 4.462117 * gate).abs() < 1e-5);
        assert!((output[1] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn metal_swiglu_cached_weights_match_uncached_forward_logits_and_are_reusable() {
        // Same fixture as `metal_swiglu_accepts_router_logits_and_applies_gate`,
        // but through the weight-resident path: upload once, call
        // `forward_logits_cached` twice. Locks in both correctness
        // (identical output to the re-upload-every-call path) and the
        // actual point of caching (repeated calls against the SAME
        // uploaded buffers give identical, correct results without
        // re-uploading).
        let kernel = MetalMoeSwiGlu::new().expect("Metal device required");
        let weights = kernel
            .upload_weights(
                &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
                &[0.0, 0.0, 1.0],
                &[0.0, 0.0, 5.0],
                2,
                1,
                1,
                1,
            )
            .unwrap();
        let gate = 2.0f32.exp() / (2.0f32.exp() + 1.0);
        for _ in 0..2 {
            let (plan, output) = kernel
                .forward_logits_cached(&[2.0, 0.0, 2.0, 0.0], &[0.0, 0.0], 2, 0.5, &weights)
                .unwrap();
            assert_eq!(plan.overflow, vec![false, true]);
            assert!((output[0] - 4.462117 * gate).abs() < 1e-5);
            assert!((output[1] - 5.0).abs() < 1e-5);
        }
    }

    #[test]
    fn metal_swiglu_uses_a_real_fallback_hidden_width_distinct_from_expert_d_ff() {
        // Locks in the real fix: a prior version of this kernel hardcoded
        // the fallback's hidden width to `dim`, which does not match
        // `reference/hz0e_moe_contract.py`'s real contract (the fallback
        // is full DENSE-FFN width, not `dim`). Here `expert_d_ff=1` but
        // `fallback_d_ff=2` -- if the fix regressed to the old hardcoded
        // behavior, this test would read past/short of the real fallback
        // buffers and either error or produce the wrong value, not
        // silently pass.
        let kernel = MetalMoeSwiGlu::new().expect("Metal device required");
        let output = kernel
            .forward(
                &[2.0],  // one token, dim=1
                &[-1],   // fallback (no expert)
                &[0.0, 0.0, 0.0], // dummy, unused 1-expert weight buffer (3*expert_d_ff*dim = 3)
                &[0.0, 0.0, 0.0], // dummy, unused 1-expert bias buffer (2*expert_d_ff+dim = 3)
                &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], // gate_w[2], up_w[2], down_w[2] (fallback_d_ff=2, dim=1)
                &[0.0, 0.0, 0.0, 0.0, 0.0], // gate_b[2], up_b[2], down_b[1]
                1,
                1,
                1,
                1,
                2, // fallback_d_ff, distinct from expert_d_ff=1
            )
            .unwrap();
        assert!((output[0] - 7.046_377).abs() < 1e-4, "got {}", output[0]);
    }
}
