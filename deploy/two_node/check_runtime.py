"""Check the built runtime and small CUDA kernels without loading model weights."""

import argparse
import importlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/workspace/models/Qwen2.5-7B-Instruct")
    args = parser.parse_args()
    import torch
    import vllm
    from transformers import AutoTokenizer

    expected = Path("/opt/ppd/vllm-source/vllm").resolve()
    assert Path(vllm.__file__).resolve().is_relative_to(expected), vllm.__file__
    assert torch.__version__.split("+", 1)[0] == "2.9.1", torch.__version__
    assert torch.cuda.is_available(), "CUDA unavailable"
    assert torch.cuda.device_count() == 1, "Expose only GPU 0"
    assert torch.cuda.get_device_capability(0) == (8, 9), "Expected RTX 4090 SM 8.9"
    native = importlib.import_module("vllm._C")
    flash = importlib.import_module("vllm.vllm_flash_attn._vllm_fa2_C")
    connector = importlib.import_module(
        "vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector")
    assert Path(connector.__file__).resolve().is_relative_to(expected)
    assert hasattr(connector.P2pNcclConnector, "is_pd_request"), "PPD patch missing"
    from vllm import _custom_ops as ops

    torch.manual_seed(42)
    x = torch.randn(8, 128, dtype=torch.bfloat16, device="cuda")
    weight = torch.ones(128, dtype=x.dtype, device=x.device)
    output = torch.empty_like(x)
    ops.rms_norm(output, x, weight, 1e-6)
    reference = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
    torch.testing.assert_close(output, reference, atol=0.02, rtol=0.02)
    product = x @ x.T
    torch.testing.assert_close(product, (x.float() @ x.float().T).to(x.dtype), atol=0.1, rtol=0.02)
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    # Qwen2.5-7B uses head dimension 128 and seven query heads per KV head.
    q = torch.randn(16, 7, 128, dtype=x.dtype, device=x.device)
    k = torch.randn(16, 1, 128, dtype=x.dtype, device=x.device)
    v = torch.randn_like(k)
    lengths = torch.tensor([0, 16], dtype=torch.int32, device=x.device)
    attention = flash_attn_varlen_func(
        q=q, k=k, v=v, max_seqlen_q=16, cu_seqlens_q=lengths,
        max_seqlen_k=16, cu_seqlens_k=lengths, causal=True, fa_version=2)
    expanded_k = k.float().repeat_interleave(7, dim=1)
    expanded_v = v.float().repeat_interleave(7, dim=1)
    scores = torch.einsum("thd,shd->hts", q.float(), expanded_k) * (128 ** -0.5)
    mask = torch.ones(16, 16, dtype=torch.bool, device=x.device).triu(1)
    scores.masked_fill_(mask.unsqueeze(0), float("-inf"))
    reference_attention = torch.einsum(
        "hts,shd->thd", scores.softmax(-1), expanded_v).to(x.dtype)
    torch.testing.assert_close(attention, reference_attention, atol=0.02, rtol=0.02)
    torch.cuda.synchronize()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = tokenizer.encode("你好，请介绍一下北京。")
    assert tokens, "Empty tokenization"
    print(json.dumps({
        "passed": True, "python": sys.version, "torch": torch.__version__,
        "cuda": torch.version.cuda, "nccl": torch.cuda.nccl.version(),
        "gpu": torch.cuda.get_device_name(0), "vllm_version": vllm.__version__,
        "vllm_source": vllm.__file__, "native_extension": native.__file__,
        "flash_attention_extension": flash.__file__, "connector": connector.__file__,
        "bf16_matmul": "passed", "native_rms_norm": "passed",
        "flash_attention_2_gqa": "passed",
        "local_tokenizer_tokens": len(tokens),
        "scope": "No model weights loaded; inference and cross-node KV transfer untested",
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
