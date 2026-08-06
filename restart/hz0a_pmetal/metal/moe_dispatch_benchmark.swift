import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "hz0e_moe_scatter")!)
let tokens = Int(CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "4096")!
let width = Int(CommandLine.arguments.count > 3 ? CommandLine.arguments[3] : "768")!
let iterations = Int(CommandLine.arguments.count > 4 ? CommandLine.arguments[4] : "100")!
let warmup = 5
let experts = 4
let capacity = Int(ceil(1.5 * Double(tokens) / Double(experts)))

// Deterministic balanced routing exercises the real flattened expert/slot
// layout without requiring a model or a random-number generator.
let slots = (0..<tokens).map { token -> Int32 in
    let expert = token % experts
    let rank = token / experts
    return Int32(expert * capacity + rank)
}
let expert = (0..<(experts * capacity * width)).map { index in
    Float(index / (capacity * width) + 1)
}
let fallback = (0..<(tokens * width)).map { _ in Float(0.002) }
func buffer<T>(_ values: [T]) -> MTLBuffer {
    values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: [])! }
}
let slotsBuffer = buffer(slots)
let expertBuffer = buffer(expert)
let fallbackBuffer = buffer(fallback)
let outputBuffer = device.makeBuffer(length: fallback.count * 4, options: [])!
let uniforms = [UInt32(width), UInt32(tokens)]
let uniformBuffer = buffer(uniforms)
let queue = device.makeCommandQueue()!
var totalGpuSeconds = 0.0
var checksum: Float = 0

for _ in 0..<warmup {
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    for (index, item) in [slotsBuffer, expertBuffer, fallbackBuffer, outputBuffer].enumerated() {
        encoder.setBuffer(item, offset: 0, index: index)
    }
    encoder.setBuffer(uniformBuffer, offset: 0, index: 4)
    encoder.setBuffer(uniformBuffer, offset: 4, index: 5)
    encoder.dispatchThreads(MTLSize(width: tokens, height: width, depth: 1), threadsPerThreadgroup: MTLSize(width: min(tokens, 256), height: 1, depth: 1))
    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
    if command.status != .completed { fatalError("Metal MoE warmup dispatch failed: \(command.status)") }
}

for _ in 0..<iterations {
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    for (index, item) in [slotsBuffer, expertBuffer, fallbackBuffer, outputBuffer].enumerated() {
        encoder.setBuffer(item, offset: 0, index: index)
    }
    encoder.setBuffer(uniformBuffer, offset: 0, index: 4)
    encoder.setBuffer(uniformBuffer, offset: 4, index: 5)
    encoder.dispatchThreads(
        MTLSize(width: tokens, height: width, depth: 1),
        threadsPerThreadgroup: MTLSize(width: min(tokens, 256), height: 1, depth: 1)
    )
    encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
    if command.status != .completed { fatalError("Metal MoE measured dispatch failed: \(command.status)") }
    totalGpuSeconds += command.gpuEndTime - command.gpuStartTime
}
let values = outputBuffer.contents().bindMemory(to: Float.self, capacity: fallback.count)
checksum = values[0] + values[fallback.count - 1]
let firstNonFinite = (0..<fallback.count).first { !values[$0].isFinite }
let firstMismatch = (0..<fallback.count).first { index in
    let token = index / width
    let expectedValue = Float(token % experts + 1)
    return abs(values[index] - expectedValue) > 1e-6
}
if let mismatch = firstMismatch { fatalError("Metal MoE scatter value mismatch at index \(mismatch)") }
let totalTokens = Double(tokens * iterations)
let bufferBytes = slots.count * 4 + expert.count * 4 + fallback.count * 4 + fallback.count * 4 + uniforms.count * 4
let report: [String: Any] = [
    "tokens": tokens, "width": width, "iterations": iterations,
    "warmup_dispatches": warmup,
    "gpu_ms_total": totalGpuSeconds * 1000.0,
    "gpu_ms_per_dispatch": totalGpuSeconds * 1000.0 / Double(iterations),
    "tokens_per_second": totalTokens / totalGpuSeconds,
    "checksum": checksum,
    "finite": firstNonFinite == nil,
    "first_non_finite_index": firstNonFinite.map { $0 } ?? NSNull(),
    "matches_expected": firstMismatch == nil,
    "device_buffer_bytes": bufferBytes,
]
print(String(data: try JSONSerialization.data(withJSONObject: report), encoding: .utf8)!)
