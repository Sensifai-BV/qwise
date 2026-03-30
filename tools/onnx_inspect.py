import onnx
import os
import numpy as np

# Complete dtype → byte-size map for all ONNX TensorProto types
_DTYPE_ITEMSIZE: dict[str, int] = {
    "FLOAT":    4,
    "DOUBLE":   8,
    "FLOAT16":  2,
    "BFLOAT16": 2,
    "INT8":     1,
    "UINT8":    1,
    "INT16":    2,
    "UINT16":   2,
    "INT32":    4,
    "UINT32":   4,
    "INT64":    8,
    "UINT64":   8,
    "BOOL":     1,
    "STRING":   0,   # variable-length, skip
    "COMPLEX64":  8,
    "COMPLEX128": 16,
    "UNDEFINED":  0,
}

_QUANTIZED_OPS = frozenset({
    # QDQ nodes
    "QuantizeLinear", "DequantizeLinear",
    # QOperator fused conv/matmul
    "QLinearConv", "QLinearMatMul",
    "MatMulInteger", "ConvInteger",
    # QOperator fused elementwise
    "QLinearAdd", "QLinearMul",
    "QLinearAveragePool", "QLinearLeakyRelu",
    "QLinearSigmoid", "QLinearConcat",
    # Dynamic quantization (ORT-specific)
    "DynamicQuantizeLSTM",
    "DynamicQuantizeLinear",
    "QAttention",
    "MatMulIntegerToFloat",
})


def _dtype_name(data_type: int) -> str:
    """Return the string name for an ONNX TensorProto data_type integer."""
    try:
        return onnx.TensorProto.DataType.Name(data_type)
    except Exception:
        return f"UNKNOWN({data_type})"


def inspect_onnx_model(model_path: str) -> None:
    if not os.path.exists(model_path):
        print(f"Error: file not found: {model_path}")
        return

    model = onnx.load(model_path)

    print("=" * 60)
    print(f"ONNX Model Inspection: {os.path.basename(model_path)}")
    print("=" * 60)

    # ── 1. Basic metadata ────────────────────────────────────────────────────
    print(f"File Size:   {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")
    print(f"IR Version:  {model.ir_version}")
    print(f"Producer:    {model.producer_name} {model.producer_version}")
    print(f"Opset(s):    {', '.join(f'{e.domain or chr(39)+chr(39)}:{e.version}' for e in model.opset_import)}")

    # ── 2. I/O tensor shapes ─────────────────────────────────────────────────
    def _shape_str(type_proto) -> str:
        if not type_proto.HasField("tensor_type"):
            return "?"
        tt = type_proto.tensor_type
        dtype = _dtype_name(tt.elem_type)
        if tt.HasField("shape"):
            dims = []
            for d in tt.shape.dim:
                dims.append(str(d.dim_value) if d.dim_value else (d.dim_param or "?"))
            return f"{dtype}[{', '.join(dims)}]"
        return dtype

    print("\n[Model Inputs]")
    for inp in model.graph.input:
        print(f"  {inp.name.ljust(20)}: {_shape_str(inp.type)}")

    print("\n[Model Outputs]")
    for out in model.graph.output:
        print(f"  {out.name.ljust(20)}: {_shape_str(out.type)}")

    # ── 3. Recursive graph analysis ──────────────────────────────────────────
    stats = {
        "total_nodes":      0,
        "quantized_nodes":  0,
        "op_types":         {},
        "weight_count":     0,
        "weight_size_bytes": 0,   # int to avoid float accumulation errors
        "precisions":       set(),
        "subgraph_depth":   0,
    }

    def process_graph(graph, depth: int = 0) -> None:
        stats["subgraph_depth"] = max(stats["subgraph_depth"], depth)

        for init in graph.initializer:
            stats["weight_count"] += 1
            dtype = _dtype_name(init.data_type)
            stats["precisions"].add(dtype)

            dims = list(init.dims)
            # np.prod([]) == 1.0 (scalar tensor) — cast to int explicitly
            num_elements = int(np.prod(dims)) if dims else 1
            item_size = _DTYPE_ITEMSIZE.get(dtype, 4)
            stats["weight_size_bytes"] += num_elements * item_size

        for node in graph.node:
            stats["total_nodes"] += 1
            op = node.op_type
            stats["op_types"][op] = stats["op_types"].get(op, 0) + 1

            if op in _QUANTIZED_OPS:
                stats["quantized_nodes"] += 1

            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    process_graph(attr.g, depth + 1)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        process_graph(g, depth + 1)

    process_graph(model.graph)

    # ── 4. Display results ───────────────────────────────────────────────────
    print("\n[Graph Structure]")
    print(f"  Total nodes (incl. subgraphs): {stats['total_nodes']}")
    print(f"  Quantized op nodes found:      {stats['quantized_nodes']}")
    print(f"  Max subgraph depth:            {stats['subgraph_depth']}")

    print("\n[Node Type Breakdown]")
    for op, count in sorted(stats["op_types"].items(), key=lambda x: x[1], reverse=True):
        marker = " ◀ quantized" if op in _QUANTIZED_OPS else ""
        print(f"  {op.ljust(22)}: {count}{marker}")

    print("\n[Weights / Initializers]")
    print(f"  Total weight tensors:   {stats['weight_count']}")
    print(f"  Estimated weights size: {stats['weight_size_bytes'] / (1024 * 1024):.2f} MB")
    print(f"  Detected precisions:    {', '.join(sorted(stats['precisions']))}")

    # ── 5. Verdict ───────────────────────────────────────────────────────────
    is_quantized = (
        stats["quantized_nodes"] > 0
        or any(p in ("INT8", "UINT8") for p in stats["precisions"])
    )
    print("\n" + "-" * 40)
    if is_quantized:
        print("RESULT: ✓  Model IS quantized (INT8 nodes or weights detected).")
    else:
        print("RESULT: ✗  Model is NOT quantized (full FP32).")
    print("-" * 40)


if __name__ == "__main__":
    path = "model/qwise_vad_int8.onnx"
    inspect_onnx_model(path)