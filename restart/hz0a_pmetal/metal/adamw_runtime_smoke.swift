import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "hz0a_adamw")!)
let count: UInt32 = 8
let parameters: [Float] = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8]
let gradients: [Float] = [1, -2, 3, -4, 5, -6, 7, -8]
let first = [Float](repeating: 0, count: Int(count)), second = [Float](repeating: 0, count: Int(count))
func buffer(_ values: [Float]) -> MTLBuffer { device.makeBuffer(bytes: values, length: values.count * MemoryLayout<Float>.stride, options: [])! }
let buffers = [buffer(parameters), buffer(gradients), buffer(first), buffer(second), device.makeBuffer(length: Int(count) * 4, options: [])!, device.makeBuffer(length: Int(count) * 4, options: [])!, device.makeBuffer(length: Int(count) * 4, options: [])!]
let command = device.makeCommandQueue()!.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
for (index, item) in buffers.enumerated() { encoder.setBuffer(item, offset: 0, index: index) }
var lr: Float = 1e-4, beta1: Float = 0.9, beta2: Float = 0.999, epsilon: Float = 1e-8, decay: Float = 0.01
var step: UInt32 = 1
encoder.setBytes([count], length: 4, index: 7)
encoder.setBytes([lr], length: 4, index: 8)
encoder.setBytes([beta1], length: 4, index: 9)
encoder.setBytes([beta2], length: 4, index: 10)
encoder.setBytes([epsilon], length: 4, index: 11)
encoder.setBytes([decay], length: 4, index: 12)
encoder.setBytes([step], length: 4, index: 13)
encoder.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
let output = buffers[4].contents().bindMemory(to: Float.self, capacity: Int(count))
let nextFirst = buffers[5].contents().bindMemory(to: Float.self, capacity: Int(count))
let nextSecond = buffers[6].contents().bindMemory(to: Float.self, capacity: Int(count))
let result: [String: Any] = ["parameters": Array(UnsafeBufferPointer(start: output, count: Int(count))), "first": Array(UnsafeBufferPointer(start: nextFirst, count: Int(count))), "second": Array(UnsafeBufferPointer(start: nextSecond, count: Int(count)))]
print(String(data: try JSONSerialization.data(withJSONObject: result), encoding: .utf8)!)
