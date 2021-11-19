import time
import torch
import pickle


from .. import wave, generators


def measure(gen, n: int, verbose: bool = False):
    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA,
    #     ],
    #     with_stack=True,
    # ) as p:
    #     generators.slice_generator(gen, stop=n)[0].cpu()
    # if verbose:
    #     print(
    #         p.key_averages().table(
    #             sort_by="cuda_time_total", row_limit=10, top_level_events_only=True
    #         )
    #     )
    # ev = p.profiler.total_average()
    # return ev.cuda_time_total / 1e6, ev.cpu_time_total / 1e6

    t = time.time()
    generators.slice_generator(gen, stop=n)[0].cpu()
    return (time.time() - t) / n


@torch.no_grad()
def main():
    dilations = [2 ** i for i in range(30)]
    rs = [
        wave.compute_receptive_field(dilations[: i + 1], [2] * (i + 1))
        for i in range(len(dilations))
    ]
    B = 32
    T = 2 ** 13
    N = 256
    W = 128
    Q = 8

    data = []
    x = torch.randint(0, Q, (B, Q, T)).float().cuda()
    sampler = lambda x: x
    for i in range(len(dilations)):
        try:
            print(dilations[: (i + 1)])
            net = (
                wave.WaveNet(
                    quantization_levels=Q,
                    wave_dilations=dilations[: (i + 1)],
                    wave_channels=W,
                )
                .cuda()
                .eval()
            )
            # burn-in
            for _ in range(10):
                net(x)
            g_slow = generators.generate(net, x, sampler)
            t_slow = measure(g_slow, N, verbose=True)
            del g_slow
            g_fast = generators.generate_fast(net, x, sampler)
            t_fast = measure(g_fast, N, verbose=True)
            del g_fast
            entry = {
                "R": rs[i],
                "L": i,
                "slow": t_slow,
                "fast": t_fast,
                "B": B,
                "W": W,
                "Q": Q,
            }
            print(entry)
            data.append(entry)
        except RuntimeError as e:
            break
        except KeyboardInterrupt:
            print("Stopping...")
            break
    pickle.dump(
        data,
        open(f"tmp/profile_generators_W{W}_B{B}_Q{Q}_T{T}_N{N}.pkl", "wb"),
    )


if __name__ == "__main__":
    # python -m autoregressive.scripts.generate --config models\fseries\config.yaml "models\fseries\wavenet-epoch=16-val_loss_epoch=4.9223.ckpt"
    main()


"""
{'R': 8192, 'L': 12, 'slow': (2.342562, 2.227074), 'fast': (2.436324, 2.310584), 'B': 1, 'W': 128, 'Q': 8}
[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
Receptive field of WaveNet 16384
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
    aten::cudnn_convolution         7.88%     187.529ms        17.86%     425.010ms     137.100us        1.646s        65.77%        1.656s     534.272us          3100
                 aten::add_         4.47%     106.298ms         4.47%     106.298ms      34.290us     290.598ms        11.61%     290.598ms      93.741us          3100
                aten::copy_         1.96%      46.667ms         1.96%      46.667ms      29.167us     154.108ms         6.16%     154.108ms      96.317us          1600
                aten::fill_         1.77%      42.224ms         1.77%      42.224ms      28.131us     117.605ms         4.70%     117.605ms      78.351us          1501
         aten::_convolution         5.91%     140.682ms        57.71%        1.373s     442.949us     111.247ms         4.44%        2.099s     677.225us          3100
           aten::as_strided        17.31%     411.960ms        17.31%     411.960ms      22.785us      20.755ms         0.83%      20.755ms       1.148us         18080
                aten::slice         7.52%     178.880ms        13.86%     329.717ms      45.161us      17.129ms         0.68%      26.174ms       3.585us          7301
            aten::unsqueeze         8.89%     211.599ms        16.52%     393.052ms      51.717us      16.471ms         0.66%      24.734ms       3.254us          7600
              aten::sigmoid         1.70%      40.467ms         1.70%      40.467ms      28.905us      14.236ms         0.57%      14.236ms      10.169us          1400
                  aten::mul         1.72%      40.906ms         1.72%      40.906ms      29.219us      12.243ms         0.49%      12.243ms       8.745us          1400
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 2.379s
Self CUDA time total: 2.503s

fast
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                       Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                 aten::_cat         2.92%      73.718ms         5.82%     146.993ms      91.756us        2.106s        78.42%        2.109s       1.317ms          1602
    aten::cudnn_convolution         7.60%     192.175ms        16.67%     421.381ms     134.670us     359.970ms        13.41%     370.293ms     118.342us          3129
         aten::_convolution         5.38%     135.890ms        51.99%        1.314s     419.931us      25.149ms         0.94%     457.360ms     146.168us          3129
           aten::as_strided        17.14%     433.193ms        17.14%     433.193ms      22.231us      24.789ms         0.92%      24.789ms       1.272us         19486
                aten::slice         8.70%     219.837ms        15.75%     398.175ms      46.833us      18.618ms         0.69%      27.909ms       3.283us          8502
                 aten::add_         3.88%      97.971ms         3.88%      97.971ms      31.311us      17.727ms         0.66%      17.727ms       5.665us          3129
            aten::unsqueeze         8.13%     205.408ms        15.10%     381.750ms      49.850us      16.744ms         0.62%      28.648ms       3.741us          7658
                 aten::roll         1.69%      42.772ms         4.45%     112.562ms      75.041us      16.352ms         0.61%      22.725ms      15.150us          1500
                aten::copy_         1.73%      43.812ms         1.73%      43.812ms      28.635us      10.240ms         0.38%      10.240ms       6.693us          1530
              aten::resize_         7.33%     185.146ms         7.33%     185.146ms      23.555us       8.649ms         0.32%       8.649ms       1.100us          7860
---------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 2.527s
Self CUDA time total: 2.685s
"""