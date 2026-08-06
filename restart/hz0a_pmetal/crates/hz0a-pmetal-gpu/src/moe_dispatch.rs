use hz0e_pmetal_moe::DispatchPlan;
use metal::{Device, MTLResourceOptions, MTLSize};

const SOURCE: &str = include_str!("../../../metal/moe_dispatch.metal");

pub struct MetalMoeScatter {
    device: Device,
    pipeline: metal::ComputePipelineState,
    queue: metal::CommandQueue,
}

impl MetalMoeScatter {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("no Metal device available")?;
        let library = device
            .new_library_with_source(SOURCE, &metal::CompileOptions::new())
            .map_err(|e| format!("MoE scatter shader compilation failed: {e}"))?;
        let function = library
            .get_function("hz0e_moe_scatter", None)
            .map_err(|e| format!("could not find hz0e_moe_scatter: {e}"))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("could not build MoE scatter pipeline: {e}"))?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            pipeline,
            queue,
        })
    }

    /// Scatter grouped expert rows back to token order. `dispatch_slot` uses
    /// `-1` for overflow rows, which retain their fallback output.
    pub fn scatter(
        &self,
        dispatch_slot: &[i32],
        expert_outputs: &[f32],
        fallback_outputs: &[f32],
        experts: usize,
        capacity: usize,
        width: usize,
    ) -> Result<Vec<f32>, String> {
        if experts == 0 || capacity == 0 || width == 0 {
            return Err("MoE scatter dimensions must be positive".into());
        }
        let tokens = dispatch_slot.len();
        let queue_size = experts
            .checked_mul(capacity)
            .ok_or("MoE queue size overflow")?;
        let expert_size = queue_size
            .checked_mul(width)
            .ok_or("MoE expert buffer size overflow")?;
        let token_size = tokens
            .checked_mul(width)
            .ok_or("MoE token buffer size overflow")?;
        if expert_outputs.len() != expert_size || fallback_outputs.len() != token_size {
            return Err("MoE scatter buffer shape mismatch".into());
        }
        let finite = |values: &[f32]| values.iter().all(|value| value.is_finite());
        if !finite(expert_outputs) || !finite(fallback_outputs) {
            return Err("MoE scatter input contains non-finite values".into());
        }
        if dispatch_slot
            .iter()
            .any(|&slot| slot >= 0 && slot as usize >= queue_size)
        {
            return Err("MoE dispatch slot is out of range".into());
        }
        let opts = MTLResourceOptions::StorageModeShared;
        let f32_buffer = |data: &[f32]| {
            self.device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of_val(data) as u64,
                opts,
            )
        };
        let i32_buffer = |data: &[i32]| {
            self.device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                std::mem::size_of_val(data) as u64,
                opts,
            )
        };
        let slots = i32_buffer(dispatch_slot);
        let expert = f32_buffer(expert_outputs);
        let fallback = f32_buffer(fallback_outputs);
        let output = self
            .device
            .new_buffer((fallback_outputs.len() * 4) as u64, opts);
        let uniforms = [
            u32::try_from(width).map_err(|_| "scatter width exceeds Metal u32 range")?,
            u32::try_from(tokens).map_err(|_| "scatter token count exceeds Metal u32 range")?,
        ];
        let uniform = self.device.new_buffer_with_data(
            uniforms.as_ptr() as *const std::ffi::c_void,
            (uniforms.len() * std::mem::size_of::<u32>()) as u64,
            opts,
        );

        let command = self.queue.new_command_buffer();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(&slots), 0);
        encoder.set_buffer(1, Some(&expert), 0);
        encoder.set_buffer(2, Some(&fallback), 0);
        encoder.set_buffer(3, Some(&output), 0);
        encoder.set_buffer(4, Some(&uniform), 0);
        encoder.set_buffer(5, Some(&uniform), 4);
        encoder.dispatch_threads(
            MTLSize::new(tokens as u64, width as u64, 1),
            MTLSize::new(
                tokens
                    .min(self.pipeline.max_total_threads_per_threadgroup() as usize)
                    .max(1) as u64,
                1,
                1,
            ),
        );
        encoder.end_encoding();
        command.commit();
        command.wait_until_completed();
        if command.status() != metal::MTLCommandBufferStatus::Completed {
            return Err(format!(
                "MoE scatter command failed: {:?}",
                command.status()
            ));
        }
        let result = unsafe {
            std::slice::from_raw_parts(output.contents() as *const f32, fallback_outputs.len())
                .to_vec()
        };
        if !finite(&result) {
            return Err("MoE scatter device output contains non-finite values".into());
        }
        Ok(result)
    }

    /// Dispatch directly from the canonical Rust routing plan. This keeps
    /// queue construction and device buffer construction in one contract.
    pub fn scatter_plan(
        &self,
        plan: &DispatchPlan,
        expert_outputs: &[f32],
        fallback_outputs: &[f32],
        width: usize,
        experts: usize,
    ) -> Result<Vec<f32>, String> {
        if width == 0 {
            return Err("dispatch width must be positive".into());
        }
        plan.validate(experts)?;
        let dispatch_slot = plan
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
        self.scatter(
            &dispatch_slot,
            expert_outputs,
            fallback_outputs,
            experts,
            plan.capacity,
            width,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::MetalMoeScatter;
    use hz0e_pmetal_moe::{build_dispatch_plan, DispatchPlan};

    #[test]
    fn metal_scatter_matches_fallback_contract() {
        let scatter = MetalMoeScatter::new().expect("Metal device required");
        let output = scatter
            .scatter(
                &[0, 1, -1],
                &[10.0, 11.0, 20.0, 21.0],
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                1,
                2,
                2,
            )
            .unwrap();
        assert_eq!(output, vec![10.0, 11.0, 20.0, 21.0, 5.0, 6.0]);
    }

    #[test]
    fn metal_scatter_accepts_canonical_dispatch_plan() {
        let scatter = MetalMoeScatter::new().expect("Metal device required");
        let plan = build_dispatch_plan(&[0, 0, 0], 1, 2.0 / 3.0).unwrap();
        let output = scatter
            .scatter_plan(&plan, &[10.0, 20.0], &[1.0, 2.0, 3.0], 1, 1)
            .unwrap();
        assert_eq!(output, vec![10.0, 20.0, 3.0]);
    }

    #[test]
    fn padded_canonical_plan_dispatches_without_uploading_padding() {
        let scatter = MetalMoeScatter::new().expect("Metal device required");
        let plan = build_dispatch_plan(&[0, 1], 4, 1.5).unwrap();
        let output = scatter
            .scatter_plan(&plan, &[10.0, 20.0, 30.0, 40.0], &[1.0, 2.0], 1, 4)
            .unwrap();
        assert_eq!(output, vec![10.0, 20.0]);
    }

    #[test]
    fn malformed_canonical_plan_is_rejected_before_dispatch() {
        let scatter = MetalMoeScatter::new().expect("Metal device required");
        let mut plan = build_dispatch_plan(&[0, 0, 0], 1, 2.0 / 3.0).unwrap();
        plan.dispatch_slot[0] = 1;
        let result = scatter.scatter_plan(&plan, &[10.0, 20.0], &[1.0, 2.0, 3.0], 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn overflowing_dimensions_are_rejected_without_panicking() {
        let scatter = MetalMoeScatter::new().expect("Metal device required");
        let result = scatter.scatter(&[], &[], &[], usize::MAX, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn non_finite_scatter_buffers_are_rejected_before_dispatch() {
        let scatter = MetalMoeScatter::new().expect("Metal device required");
        let result = scatter.scatter(&[0], &[f32::NAN], &[0.0], 1, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn overflowing_canonical_plan_is_rejected_without_panicking() {
        let scatter = MetalMoeScatter::new().expect("Metal device required");
        let plan = DispatchPlan {
            capacity: usize::MAX,
            expert_index: vec![0],
            rank: vec![0],
            accepted: vec![true],
            overflow: vec![false],
            dispatch_slot: vec![0],
            grouped_tokens: vec![],
        };
        assert!(scatter.scatter_plan(&plan, &[1.0], &[0.0], 1, 2).is_err());
    }
}
