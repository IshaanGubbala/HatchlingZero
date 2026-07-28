#include <metal_stdlib>
using namespace metal;

kernel void hz0a_adamw(
    device const float *parameters [[buffer(0)]], device const float *gradients [[buffer(1)]],
    device const float *first_moment [[buffer(2)]], device const float *second_moment [[buffer(3)]],
    device float *next_parameters [[buffer(4)]], device float *next_first [[buffer(5)]],
    device float *next_second [[buffer(6)]], constant uint &count [[buffer(7)]],
    constant float &learning_rate [[buffer(8)]], constant float &beta1 [[buffer(9)]],
    constant float &beta2 [[buffer(10)]], constant float &epsilon [[buffer(11)]],
    constant float &weight_decay [[buffer(12)]], constant uint &step [[buffer(13)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    float g = gradients[index];
    float m = beta1 * first_moment[index] + (1.0f - beta1) * g;
    float v = beta2 * second_moment[index] + (1.0f - beta2) * g * g;
    float m_hat = m / (1.0f - pow(beta1, float(step)));
    float v_hat = v / (1.0f - pow(beta2, float(step)));
    next_parameters[index] = parameters[index] - learning_rate * (m_hat / (sqrt(v_hat) + epsilon) + weight_decay * parameters[index]);
    next_first[index] = m;
    next_second[index] = v;
}
