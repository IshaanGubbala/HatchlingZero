import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "hz0e_moe_scatter")!)
let capacity: UInt32 = 2, width: UInt32 = 1, tokens: UInt32 = 6
// Two experts, two accepted rows each, and one overflow row per expert.
let dispatchSlot: [Int32] = [0, 1, -1, 2, 3, -1]
let expert: [Float] = [10, 20, 30, 40]
let fallback: [Float] = [1, 2, 3, 4, 5, 6]
func floatBuffer(_ values: [Float]) -> MTLBuffer {
    values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: [])! }
}
func intBuffer(_ values: [Int32]) -> MTLBuffer {
    values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: [])! }
}
let output = device.makeBuffer(length: fallback.count * MemoryLayout<Float>.stride, options: [])!
let buffers = [intBuffer(dispatchSlot), floatBuffer(expert), floatBuffer(fallback), output]
let command = device.makeCommandQueue()!.makeCommandBuffer()!
let encoder = command.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
for (index, item) in buffers.enumerated() { encoder.setBuffer(item, offset: 0, index: index) }
for (index, value) in [width, tokens].enumerated() {
    var value = value
    encoder.setBytes(&value, length: 4, index: 4 + index)
}
encoder.dispatchThreads(MTLSize(width: Int(tokens), height: Int(width), depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
let values = output.contents().bindMemory(to: Float.self, capacity: fallback.count)
let actual = Array(UnsafeBufferPointer(start: values, count: fallback.count))
let expected: [Float] = [10, 20, 3, 30, 40, 6]
let matchesExpected = actual == expected
let report: [String: Any] = ["output": actual, "expected": expected, "matches_expected": matchesExpected]
print(String(data: try JSONSerialization.data(withJSONObject: report), encoding: .utf8)!)
if !matchesExpected { exit(1) }
