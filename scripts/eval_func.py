import os
from pathlib import Path

import modal

# .resolve() gets the absolute path. .parents[1] goes up exactly two directory levels.
repo_top_dir = Path(__file__).resolve().parents[1]
REMOTE_REPO_TOP_DIR = str(repo_top_dir.parent / "KernelBench_remote")

app = modal.App("eval_single_sample")

cuda_version = "13.0.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

SRC_DIR = os.path.join(REMOTE_REPO_TOP_DIR, "src")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10")
    .apt_install(
        "git",
        "gcc-10",
        "g++-10",
        "clang",  # note i skip a step
    )
    .uv_sync(uv_project_dir=REMOTE_REPO_TOP_DIR, extras=["gpu"])
    .env({"PYTHONPATH": "/root:/root/src"})
    .add_local_dir(SRC_DIR, remote_path="/root/src")  # must be last
)

@app.cls(image=image)
class EvalFunc:

    @modal.method()
    def eval_single_sample_modal(self, ref_arch_src, custom_kernel, verbose, gpu_arch, backend, precision, timing_method):
        # 3. Evaluate Kernel
        # NOTE: no need to wrap around process here as only a single sample
        # see batch eval for examples of process isolation
        import torch
        torch.set_printoptions(precision=4, threshold=10)

        from kernelbench.eval import (
            eval_kernel_against_ref,
            get_torch_dtype_from_string,
        )

        # Use utility function to set the GPU architecture in the modal environment
        from kernelbench.utils import set_gpu_arch as modal_set_gpu_arch

        modal_set_gpu_arch(gpu_arch)
        return eval_kernel_against_ref(
            ref_arch_src,
            custom_kernel,
            verbose=verbose,
            measure_performance=True,
            timing_method=timing_method,
            num_correct_trials=5,
            num_perf_trials=100,
            backend=backend,
            precision=get_torch_dtype_from_string(precision),
        )