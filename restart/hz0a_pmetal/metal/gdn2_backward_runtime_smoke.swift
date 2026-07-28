import Foundation
import Metal

let device = MTLCreateSystemDefaultDevice()!
let library = try device.makeLibrary(URL: URL(fileURLWithPath: CommandLine.arguments[1]))
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "hz0a_gdn2_backward")!)
let b: UInt32 = 1, steps: UInt32 = 3, heads: UInt32 = 1, values: UInt32 = 2, keys: UInt32 = 2
let q: [Float] = [1, 2, 2, 1, 1, -1]
let k: [Float] = [0.5, -1, 1, 0.25, -0.5, 2]
let v: [Float] = [1, -2, 0.5, 3, -1, 0.25]
let decay: [Float] = [0.8, 0.7, 0.6, 0.9, 0.5, 0.4]
let erase: [Float] = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
let write: [Float] = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
let initial = [Float](repeating: 0, count: 4)
let gradOutput = [Float](repeating: 1, count: 6)
let gradFinal = [Float](repeating: 0, count: 4)
let inputCount = 6, valueCount = 6, stateCount = 4
func buffer(_ values: [Float]) -> MTLBuffer { device.makeBuffer(bytes: values, length: values.count * 4, options: [])! }
func empty(_ count: Int) -> MTLBuffer { device.makeBuffer(bytes: [Float](repeating: 0, count: count), length: count * 4, options: [])! }
let buffers = [buffer(q), buffer(k), buffer(v), buffer(decay), buffer(erase), buffer(write), buffer(initial), buffer(gradOutput), buffer(gradFinal), empty(inputCount), empty(inputCount), empty(valueCount), empty(inputCount), empty(inputCount), empty(valueCount), empty(stateCount)]
let command = device.makeCommandQueue()!.makeCommandBuffer()!, encoder = command.makeComputeCommandEncoder()!
encoder.setComputePipelineState(pipeline)
for (index, item) in buffers.enumerated() { encoder.setBuffer(item, offset: 0, index: index) }
for (index, value) in [b, steps, heads, values, keys].enumerated() { encoder.setBytes([value], length: 4, index: 16 + index) }
encoder.dispatchThreads(MTLSize(width: 1, height: 1, depth: 2), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
func read(_ buffer: MTLBuffer, _ count: Int) -> [Float] { Array(UnsafeBufferPointer(start: buffer.contents().bindMemory(to: Float.self, capacity: count), count: count)) }
let result: [String: Any] = ["q": read(buffers[9], inputCount), "k": read(buffers[10], inputCount), "v": read(buffers[11], valueCount), "decay": read(buffers[12], inputCount), "erase": read(buffers[13], inputCount), "write": read(buffers[14], valueCount), "initial": read(buffers[15], stateCount)]
print(String(data: try JSONSerialization.data(withJSONObject: result), encoding: .utf8)!)
