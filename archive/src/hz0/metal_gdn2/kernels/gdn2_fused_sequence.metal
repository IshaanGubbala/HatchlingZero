#define MAX_DK  64
#define MAX_DV  64
#define CLIP    100.0

constexpr uint B  = {B};
constexpr uint T  = {T};
constexpr uint H  = {H};
constexpr uint Dk = {Dk};
constexpr uint Dv = {Dv};

uint grid_id = threadgroup_position_in_grid.x;
uint lid     = thread_position_in_threadgroup.x;

if (grid_id >= B * H || lid >= Dv) return;
uint b = grid_id / H;
uint h = grid_id - b * H;

thread float state[MAX_DK];
uint state_base = (b * H + h) * Dv * Dk + lid * Dk;
for (uint kk = 0; kk < Dk; ++kk) {{
    state[kk] = state_in[state_base + kk];
}}

#pragma unroll(1)
for (uint t = 0; t < T; ++t) {{
    uint qk_off = ((b * T + t) * H + h) * Dk;
    uint v_off  = ((b * T + t) * H + h) * Dv;
    float write_val = w[v_off + lid] * v[v_off + lid];

    thread float s_decayed[MAX_DK];
    thread float k_row[MAX_DK];
    float erase_value = 0.0f;
    for (uint kk = 0; kk < Dk; ++kk) {{
        float k_kk = k[qk_off + kk];
        k_row[kk]  = k_kk;
        float d_kk = d[qk_off + kk];
        float e_kk = e[qk_off + kk];
        float s_d  = state[kk] * d_kk;
        s_decayed[kk] = s_d;
        erase_value  += s_d * (k_kk * e_kk);
    }}

    float query_value = 0.0f;
    for (uint kk = 0; kk < Dk; ++kk) {{
        float k_kk = k_row[kk];
        float q_kk = q[qk_off + kk];
        float ns   = s_decayed[kk] - erase_value * k_kk + write_val * k_kk;
        ns = clamp(ns, -CLIP, CLIP);
        state[kk] = ns;
        query_value += ns * q_kk;
    }}

    out[v_off + lid] = query_value;
}}

for (uint kk = 0; kk < Dk; ++kk) {{
    state_out[state_base + kk] = state[kk];
}}
