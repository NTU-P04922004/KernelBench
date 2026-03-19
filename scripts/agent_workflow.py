import os
from typing import Any, Callable, TypedDict

from kernelbench.utils import extract_first_code
from langgraph.graph import END, StateGraph


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        config: The evaluation configuration.
        custom_prompt: The prompt to be sent to the model.
        custom_kernel: The generated kernel code.
    """
    level: str
    problem_id: int
    problem_name: str
    ref_arch_src: str
    custom_prompt: str

    check_kernel: bool
    backend: str
    precision: str
    gpu: str
    timing_method: str

    log: bool
    logdir: str
    verbose: bool

    custom_kernel: str

    inference_server: Callable
    eval_callback: Callable

    result: dict
    kernel_exec_result: dict


def generate_kernel(state: GraphState):
    """Node to generate kernel code from a prompt."""
    print("---GENERATING KERNEL---")

    backend = state["backend"]
    custom_prompt = state["custom_prompt"]
    inference_server = state["inference_server"]

    # Query server with constructed prompt
    custom_kernel = inference_server(custom_prompt)
    custom_kernel = extract_first_code(custom_kernel, ["python", "cpp"])

    # check LLM is able to generate custom kernel code
    assert custom_kernel is not None, (
        f"Custom {backend} kernel code generation failed"
    )
    return {"custom_kernel": custom_kernel}


def evaluate_kernel(state: GraphState):
    """Node to evaluate the generated kernel."""
    print("---EVALUATING KERNEL---")

    check_kernel = state["check_kernel"]
    backend = state["backend"]
    precision = state["precision"]
    custom_kernel = state["custom_kernel"]
    level = state["level"]
    problem_name = state["problem_name"]
    problem_id = state["problem_id"]
    ref_arch_src = state["ref_arch_src"]
    log = state["log"]
    logdir = state["logdir"]
    gpu = state["gpu"]
    verbose = state["verbose"]
    timing_method = state["timing_method"]

    # Optional: static code checker for kernel code using regex matching
    # NOTE: by no means is this checker complete, but it could help catch some potential hacks
    if check_kernel:
        from kernelbench.kernel_static_checker import validate_kernel_static
        static_check_status, errors, warnings = validate_kernel_static(
            custom_kernel,
            backend=backend,
            precision=precision,
        )
        assert static_check_status, f"Static check failed for level {level} problem {problem_id}. Errors: {errors}. Warnings: {warnings}"
        if warnings:
            print(f"Static check warnings for level {level} problem {problem_id}: {warnings}")

    # this should be optional
    if log:
        with open(os.path.join(logdir, f"generated_kernel_level_{level}_problem_{problem_id}.py"), "w") as f:
            f.write(custom_kernel)

    eval_callback = state["eval_callback"]
    kernel_exec_result = eval_callback(ref_arch_src, custom_kernel, gpu, verbose, timing_method, backend, precision)

    print(
        f"Evaluation result for level {level} problem {problem_id}:\n{kernel_exec_result}"
    )

    if log:
        with open(os.path.join(logdir, f"eval_result_level_{level}_problem_{problem_id}.txt"), "a") as f:
            f.write(f"Problem Name: {problem_name}\n")
            f.write(str(kernel_exec_result))

    return {"kernel_exec_result": kernel_exec_result.model_dump()}


def refine_kernel(state: GraphState):
    """Node to generate kernel code from a prompt."""
    print("---REFINING KERNEL---")
    # config = state["config"]
    backend = state["backend"]
    custom_prompt = state["custom_prompt"]
    inference_server = state["inference_server"]

    # Query server with constructed prompt
    custom_kernel = inference_server(custom_prompt)
    custom_kernel = extract_first_code(custom_kernel, ["python", "cpp"])

    # check LLM is able to generate custom kernel code
    assert custom_kernel is not None, (
        f"Custom {backend} kernel code generation failed"
    )
    return {"custom_kernel": custom_kernel}


def build_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("generate_kernel", generate_kernel)
    workflow.add_node("evaluate_kernel", evaluate_kernel)
    workflow.set_entry_point("generate_kernel")
    workflow.add_edge("generate_kernel", "evaluate_kernel")
    workflow.add_edge("evaluate_kernel", END)
    return workflow.compile()


def build_iterative_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("generate_kernel", generate_kernel)
    workflow.add_node("evaluate_kernel", evaluate_kernel)
    workflow.set_entry_point("generate_kernel")
    workflow.add_edge("generate_kernel", "evaluate_kernel")
    workflow.add_edge("evaluate_kernel", END)
    return workflow.compile()