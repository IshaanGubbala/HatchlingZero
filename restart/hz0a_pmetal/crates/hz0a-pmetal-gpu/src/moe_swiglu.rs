use hz0e_pmetal_moe::{build_top1_dispatch_plan_f32, DispatchPlan};
use metal::{Device, MTLResourceOptions, MTLSize};

const SOURCE: &str = include_str!("../../../metal/moe_swiglu.metal");

pub struct MetalMoeSwiGlu {
    device: Device,
    pipeline: metal::ComputePipelineState,
    queue: metal::CommandQueue,
}

impl MetalMoeSwiGlu {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device available")?;
        let library = device
            .new_library_with_source(SOURCE, &metal::CompileOptions::new())
            .map_err(|e| format!("MoE SwiGLU shader compilation failed: {e}"))?;
        let function = library
            .get_function("hz0e_moe_swiglu", None)
            .map_err(|e| format!("could not find hz0e_moe_swiglu: {e}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("could not build MoE SwiGLU pipeline: {e}"))?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            pipeline,
            queue,
        })
    }

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
    ) -> Result<Vec<f32>, String> {
        if experts == 0 || capacity == 0 || dim == 0 || expert_d_ff == 0 {
            return Err("MoE SwiGLU dimensions must be positive".into());
        }
        let tokens = dispatch_slot.len();
        let input_size = tokens.checked_mul(dim).ok_or("input size overflow")?;
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
            .checked_mul(dim)
            .and_then(|v| v.checked_mul(dim))
            .ok_or("fallback weight size overflow")?;
        let fallback_biases_size = 3usize
            .checked_mul(dim)
            .ok_or("fallback bias size overflow")?;
        if input.len() != input_size
            || expert_weights.len() != expert_weights_size
            || expert_biases.len() != expert_biases_size
            || fallback_weights.len() != fallback_weights_size
            || fallback_biases.len() != fallback_biases_size
        {
            return Err("MoE SwiGLU buffer shape mismatch".into());
        }
        let finite = |values: &[f32]| values.iter().all(|value| value.is_finite());
        if !finite(input)
            || !finite(expert_weights)
            || !finite(expert_biases)
            || !finite(fallback_weights)
            || !finite(fallback_biases)
        {
            return Err("MoE SwiGLU input or parameter buffer contains non-finite values".into());
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
        let buffers = [
            f32_buffer(input),
            i32_buffer(dispatch_slot),
            f32_buffer(expert_weights),
            f32_buffer(expert_biases),
            f32_buffer(fallback_weights),
            f32_buffer(fallback_biases),
        ];
        let output = self.device.new_buffer((input_size * 4) as u64, opts);
        let uniforms = [
            u32::try_from(capacity).map_err(|_| "expert capacity exceeds Metal u32 range")?,
            u32::try_from(dim).map_err(|_| "expert width exceeds Metal u32 range")?,
            u32::try_from(expert_d_ff)
                .map_err(|_| "expert hidden width exceeds Metal u32 range")?,
            u32::try_from(tokens).map_err(|_| "expert token count exceeds Metal u32 range")?,
        ];
        let uniform = self.device.new_buffer_with_data(
            uniforms.as_ptr() as *const std::ffi::c_void,
            16,
            opts,
        );
        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        for (index, buffer) in buffers.iter().enumerate() {
            encoder.set_buffer(index as u64, Some(buffer), 0);
        }
        encoder.set_buffer(6, Some(&output), 0);
        for index in 0..4 {
            encoder.set_buffer(7 + index as u64, Some(&uniform), (index * 4) as u64);
        }
        encoder.dispatch_threads(
            MTLSize::new(tokens as u64, dim as u64, 1),
            MTLSize::new(tokens.min(256).max(1) as u64, 1, 1),
        );
        encoder.end_encoding();
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
    ) -> Result<Vec<f32>, String> {
        if dim == 0 || expert_d_ff == 0 {
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
            )
            .unwrap();
        assert_eq!(plan.overflow, vec![false, true]);
        let gate = 2.0f32.exp() / (2.0f32.exp() + 1.0);
        assert!((output[0] - 4.462117 * gate).abs() < 1e-5);
        assert!((output[1] - 5.0).abs() < 1e-5);
    }
}
