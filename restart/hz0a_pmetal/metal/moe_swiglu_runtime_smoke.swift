import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "hz0e_moe_swiglu")!)
let capacity: UInt32 = 1, dim: UInt32 = 1, dff: UInt32 = 1, tokens: UInt32 = 3
let input: [Float] = [0, 0, 0]
let slots: [Int32] = [0, -1, 1]
// Each expert has zero weights and distinct gate/up/down biases.
let expertWeights: [Float] = [0, 0, 1, 0, 0, 1]
let expertBiases: [Float] = [1, 2, 3, 2, 3, 4]
let fallbackWeights: [Float] = [0, 0, 1]
let fallbackBiases: [Float] = [3, 4, 5]
func buffer<T>(_ values: [T]) -> MTLBuffer {
    values.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: [])! }
}
let output = device.makeBuffer(length: input.count * 4, options: [])!
let uniform = buffer([capacity, dim, dff, tokens])
let buffers = [buffer(input), buffer(slots), buffer(expertWeights), buffer(expertBiases), buffer(fallbackWeights), buffer(fallbackBiases), output]
let command = device.makeCommandQueue()!.makeCommandBuffer()!
let encoder = command.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
for (index, item) in buffers.enumerated() { encoder.setBuffer(item, offset: 0, index: index) }
for index in 0..<4 { encoder.setBuffer(uniform, offset: index * 4, index: 7 + index) }
encoder.dispatchThreads(MTLSize(width: Int(tokens), height: Int(dim), depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
let values = output.contents().bindMemory(to: Float.self, capacity: input.count)
let actual = Array(UnsafeBufferPointer(start: values, count: input.count))
let expected: [Float] = [1.0 / (1.0 + exp(-1.0)) * 2.0 + 3.0, 3.0 / (1.0 + exp(-3.0)) * 4.0 + 5.0, 2.0 / (1.0 + exp(-2.0)) * 3.0 + 4.0]
let matches = zip(actual, expected).allSatisfy { abs($0 - $1) < 1e-5 }
print(String(data: try JSONSerialization.data(withJSONObject: ["output": actual, "matches_expected": matches]), encoding: .utf8)!)
if !matches { exit(1) }
