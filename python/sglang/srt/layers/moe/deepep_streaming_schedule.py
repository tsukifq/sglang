"""Fixed single-wave topology; reuse only inside a serialized prepared lease."""
import torch


class _PreparedWaveSchedule:
    def __init__(self, streams, lane_output, wave_size):
        from cuda.bindings import driver as cuda
        lanes = lane_output.shape[0]
        if wave_size != lanes or lanes <= 1 or len(streams) != lanes:
            raise ValueError("prepared schedule requires one full wave")
        self.wave_ranges = ((0, lanes),)
        self.wave_streams = (streams[0],)
        # Context.__enter__ refreshes the caller's current stream every epoch.
        # The plan lease prohibits reentrant use of this mutable context.
        self.compute_context = torch.cuda.stream(streams[0])
        self.wave_output = lane_output
        self.driver_stream = cuda.CUstream(streams[0].cuda_stream)
        self.wait_flag = int(cuda.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ)
