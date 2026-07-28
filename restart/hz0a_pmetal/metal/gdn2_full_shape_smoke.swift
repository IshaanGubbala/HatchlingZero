import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "hz0a_gdn2_forward")!)
let b: UInt32 = 1, t: UInt32 = 2, h: UInt32 = 12, vdim: UInt32 = 64, kdim: UInt32 = 64
func values(_ count: Int, _ scale: Float, _ offset: Int) -> [Float] { (0..<count).map { Float((($0 * 17 + offset) % 101) - 50) / scale } }
let inputCount = Int(b * t * h * kdim)
let valueCount = Int(b * t * h * vdim)
let stateCount = Int(b * h * vdim * kdim)
let q = values(inputCount, 50, 1), k = values(inputCount, 50, 3), decay = values(inputCount, 100, 20).map { $0 + 0.5 }, erase = values(inputCount, 100, 40).map { $0 + 0.5 }, write = values(valueCount, 100, 60).map { $0 + 0.5 }, vv = values(valueCount, 50, 7)
let initial = [Float](repeating: 0, count: stateCount)
func buffer(_ data: [Float]) -> MTLBuffer { device.makeBuffer(bytes: data, length: data.count * MemoryLayout<Float>.stride, options: [])! }
let buffers = [buffer(q), buffer(k), buffer(vv), buffer(decay), buffer(erase), buffer(write), buffer(initial), device.makeBuffer(length: valueCount * MemoryLayout<Float>.stride, options: [])!, device.makeBuffer(length: stateCount * MemoryLayout<Float>.stride, options: [])!]
let command = device.makeCommandQueue()!.makeCommandBuffer()!
let encoder = command.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
for (index, item) in buffers.enumerated() { encoder.setBuffer(item, offset: 0, index: index) }
for (index, value) in [b, t, h, vdim, kdim].enumerated() { encoder.setBytes([value], length: MemoryLayout<UInt32>.stride, index: 9 + index) }
encoder.dispatchThreads(MTLSize(width: Int(b), height: Int(h), depth: Int(vdim)), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
let output = buffers[7].contents().bindMemory(to: Float.self, capacity: valueCount)
let final = buffers[8].contents().bindMemory(to: Float.self, capacity: stateCount)
let result: [String: Any] = ["output": Array(UnsafeBufferPointer(start: output, count: valueCount)), "final_state": Array(UnsafeBufferPointer(start: final, count: stateCount))]
print(String(data: try JSONSerialization.data(withJSONObject: result), encoding: .utf8)!)
