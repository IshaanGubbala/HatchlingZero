# HZ-0I factor rank sweep

A 30-step real packed-corpus sweep at the 96-wide bring-up scale gave:

| Rank | Parameters | Final CE | tok/s |
|---:|---:|---:|---:|
| Dense control | 5.60M | 9.047 | 4,367 |
| 8 | 4.80M | 9.886 | 4,840 |
| 16 | 4.88M | 9.505 | 5,095 |
| 32 | 5.05M | 9.438 | 5,043 |
| 64 | 5.38M | 9.254 | 4,759 |

Rank 16/32 improve throughput but lose short-run CE; rank 64 is the best
quality/size point in this tiny sweep but still trails dense. These are not
long-run quality conclusions. The 0.3B target should use rank 64/128/256
ablations rather than assuming rank 256 is optimal.
