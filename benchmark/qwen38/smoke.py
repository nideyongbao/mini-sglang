import sys

from minisgl.core import SamplingParams
from minisgl.llm import LLM

llm = LLM(
    "/model",
    max_running_req=8,
    max_seq_len_override=int(sys.argv[1]) if len(sys.argv) > 1 else 2048,
    num_page_override=8192,
    max_extend_tokens=512,
    cuda_graph_bs=[],
    attention_backend="fa",
    cache_type="naive",
    use_pynccl=False,
)
print(
    llm.generate(
        [[100, 200, 300, 400], [1, 2, 3, 4, 5, 6]], SamplingParams(max_tokens=8, ignore_eos=True)
    )
)
llm.shutdown()
