import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
let airPath = CommandLine.arguments[1]
let iterations = CommandLine.arguments.count > 2 ? Int(CommandLine.arguments[2])! : 1
let library = try device.makeLibrary(URL: URL(fileURLWithPath: airPath))
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "hz0a_gdn2_forward")!)
let queue = device.makeCommandQueue()!

let batch: UInt32 = 1
let steps: UInt32 = 3
let heads: UInt32 = 1
let values: UInt32 = 2
let keys: UInt32 = 2
let q: [Float] = [1, 2, 2, 1, 1, -1]
let k: [Float] = [0.5, -1, 1, 0.25, -0.5, 2]
let v: [Float] = [1, -2, 0.5, 3, -1, 0.25]
let decay: [Float] = [0.8, 0.7, 0.6, 0.9, 0.5, 0.4]
let erase: [Float] = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
let write: [Float] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
let initial = [Float](repeating: 0, count: Int(values * keys))
let output = [Float](repeating: 0, count: Int(steps * heads * values))
let final = [Float](repeating: 0, count: Int(values * keys))
func buffer(_ data: [Float], _ options: MTLResourceOptions = []) -> MTLBuffer { device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: options)! }
let qBuffer = buffer(q), kBuffer = buffer(k), vBuffer = buffer(v), dBuffer = buffer(decay), eBuffer = buffer(erase), wBuffer = buffer(write), iBuffer = buffer(initial)
let oBuffer = device.makeBuffer(bytes: output, length: output.count * MemoryLayout<Float>.stride, options: [])!
let fBuffer = device.makeBuffer(bytes: final, length: final.count * MemoryLayout<Float>.stride, options: [])!
let start = Date().timeIntervalSinceReferenceDate
for _ in 0..<iterations {
    let command = queue.makeCommandBuffer()!
    let encoder = command.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(pipeline)
    for (index, buffer) in [qBuffer, kBuffer, vBuffer, dBuffer, eBuffer, wBuffer, iBuffer, oBuffer, fBuffer].enumerated() { encoder.setBuffer(buffer, offset: 0, index: index) }
    let dimensions = [batch, steps, heads, values, keys]
    for (index, value) in dimensions.enumerated() { encoder.setBytes([value], length: MemoryLayout<UInt32>.stride, index: 9 + index) }
    encoder.dispatchThreads(MTLSize(width: Int(batch), height: Int(heads), depth: Int(values)), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
}
let elapsedMilliseconds = (Date().timeIntervalSinceReferenceDate - start) * 1000.0
let outputValues = oBuffer.contents().bindMemory(to: Float.self, capacity: output.count)
let finalValues = fBuffer.contents().bindMemory(to: Float.self, capacity: final.count)
let result = ["output": Array(UnsafeBufferPointer(start: outputValues, count: output.count)), "final_state": Array(UnsafeBufferPointer(start: finalValues, count: final.count)), "iterations": iterations, "kernel_elapsed_ms": elapsedMilliseconds] as [String: Any]
print(String(data: try JSONSerialization.data(withJSONObject: result), encoding: .utf8)!)
