# Performance Notes

Starting this here to collect some notes on inferencing performance on different hardware.
Let's try to collect only the really meaningful data points / sources here.

## 4090 vs H100

**From:** https://www.reddit.com/r/LocalLLaMA/comments/188boew/multiple_4090s_instead_of_h100/

First of all, VRAM capacity aside, the HBM on a H100 is significantly faster than GDDR6X on a 4090. By significant I mean 2-3x the bandwidth! Memory bandwidth is often a significant bottleneck for both inference and training.

Something else you misunderstood is that computational "speed" is not solely dependent on clock speed, far from that in fact. Core architecture, core count, and other design choices massively influence the raw power of a GPU. 4090 is made for playing video games so it has very strong FP32 compute power, nearly 100 TFlops, very impressive, much higher than H100's 64 TFlops shader FP32 performance . However, in the context of machine learning, people often prefer to drop the precision down to FP16 or lower for memory and compute efficiency. H100 on the other hand are purpose built for AI, not even focusing on traditional HPC market anymore. It has a ridiculous 1000 TFlops of FP16 Matrix compute power, a 4090 might have 150~300 TFlops I can't recall correctly, but it is off by a order of magnitude... H100 also support a new data format, FP8, that 4090 does not even support. The FP8 Matrix performance of H100 is about 2000 TFlops... Absolute tremendous power within just 1 card at around 300-700W of TDP.

Something else H100 has that 4090 does not is NVLink. H100 has 600-900GB/s of NVLink bandwidth so when multiple cards are installed, they can all communicate with each other very quickly to transfer data or even access other GPU's memory. Without NVLink data transfer and communication must go through PCIe. Although PCIe is also quite fast, it is not even close to NVLink unfortunately.

Considering the capacity and performance density and efficiency of a H100, it is obvious why big corporations are chasing after it. Of course the cost of making a H100 is not 20x of a 4090, but H100 really has near zero competition now so Jensen can charge others however he want. You are right that you can use a bunch of 4090s to match a H100 in someway, people in China are doing it right now because of the sanctions, they have no chlsoice. But it is in general a terrible alternative.
