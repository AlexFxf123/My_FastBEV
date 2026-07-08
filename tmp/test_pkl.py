import os, pickle, glob, numpy as np
import sys

# ========= 只在这里改一行 =========
DATAROOT = "/home/radardepth/data/nuscenes/"
# ===================================

FILE_PATH = "/home/radardepth/data/nuscenes/nuscenes_infos_val_4d_interval3_max60_wradar.pkl"  # ← 改成你的文件路径

# ──────────────────────────────────────────────
#  智能打印辅助函数
# ──────────────────────────────────────────────
def smart_print(obj, depth=0, max_items=30, max_str=200, indent="  "):
    """递归友好地打印任意 Python 对象（带截断保护）"""
    prefix = indent * depth

    # ---- numpy ndarray ----
    if _is_name(obj, ("numpy.ndarray", "numpy.matrix")):
        shape = getattr(obj, "shape", "???")
        dtype = getattr(obj, "dtype", "???")
        print(f"{prefix}<ndarray shape={shape} dtype={dtype}>")
        if obj.size <= 20:
            print(f"{prefix}{indent}{obj}")
        else:
            flat = obj.flatten()
            preview = flat[:10]
            print(f"{prefix}{indent}preview(first 10): {preview}")
        return

    # ---- pandas DataFrame / Series ----
    if _is_name(obj, ("pandas.core.frame.DataFrame", "pandas.core.series.Series")):
        print(f"{prefix}<{type(obj).__name__}>")
        print(f"{prefix}{indent}shape = {obj.shape}")
        if hasattr(obj, "columns"):
            print(f"{prefix}{indent}columns = {list(obj.columns)[:50]}")
        print(f"{prefix}{indent}dtypes:")
        print(obj.dtypes.to_string(indent=len(prefix)+2))
        print(f"{prefix}{indent}--- head(5) ---")
        print(obj.head(5).to_string())
        return

    # ---- torch Tensor ----
    if _is_name(obj, ("torch.Tensor",)):
        try:
            print(f"{prefix}<Tensor shape={tuple(obj.shape)} dtype={obj.dtype}>")
        except Exception:
            print(f"{prefix}<Tensor>")
        return

    # ---- dict ----
    if isinstance(obj, dict):
        print(f"{prefix}dict (keys={len(obj)}) {{")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{prefix}{indent}... ({len(obj) - max_items} more keys omitted)")
                break
            print(f"{prefix}{indent}{repr(k)}: ", end="")
            if isinstance(v, (str, int, float, bool, type(None))):
                val_repr = repr(v)
                if isinstance(val_repr, str) and len(val_repr) > max_str:
                    val_repr = val_repr[:max_str] + "..."
                print(val_repr)
            else:
                print()
                smart_print(v, depth + 2, max_items, max_str, indent)
        print(f"{prefix}}}")
        return

    # ---- list / tuple / set ----
    if isinstance(obj, (list, tuple, set)):
        name = type(obj).__name__
        length = len(obj)
        print(f"{prefix}{name} (len={length}) [")
        show = min(length, max_items)
        for i in range(show):
            print(f"{prefix}{indent}[{i}] ", end="")
            smart_print(obj[i], depth + 2, max_items, max_str, indent)
        if length > show:
            print(f"{prefix}{indent}... ({length - show} more items omitted)")
        closing = "]" if isinstance(obj, list) else ")" if isinstance(obj, tuple) else "}"
        print(f"{prefix}{closing}")
        return

    # ---- bytes（太长就显示 hex 前缀） ----
    if isinstance(obj, bytes):
        n = min(80, len(obj))
        print(f"{prefix}<bytes len={len(obj)} hex={obj[:n].hex()}...>")
        return

    # ---- 普通标量 / str ----
    r = repr(obj)
    if isinstance(r, str) and len(r) > max_str * 2:
        r = r[:max_str * 2] + f"... (total len={len(r)})"
    print(f"{prefix}{r}")


def _is_name(obj, names):
    """按类型名匹配（避免 hard import 某些包）"""
    t = type(obj).__name__
    m = type(obj).__module__
    full = f"{m}.{t}"
    for n in names:
        if t == n or full.endswith(n):
            return True
    return False


# ──────────────────────────────────────────────
#  主流程：打开 pkl
# ──────────────────────────────────────────────
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else FILE_PATH

    print(f"\n{'='*60}")
    print(f"📂 打开 pkl 文件: {path}")
    print(f"{'='*60}\n")

    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {path}")
        sys.exit(1)
    except pickle.UnpicklingError as e:
        print(f"❌ 不是有效的 pickle 文件: {e}")
        sys.exit(1)
    except EOFError:
        print(f"❌ 文件被截断或损坏（EOFError）")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 加载失败: {type(e).__name__}: {e}")
        sys.exit(1)

    print(f"✅ 顶层对象类型: {type(data)}\n")
    print("─" * 60)
    smart_print(data)
    print("─" * 60)

    # 如果是 dict，额外以 key-value 节选方式再打一遍（更直观）
    if isinstance(data, dict):
        print("\n📋 所有顶层 key 一览:")
        for k in data.keys():
            print(f"  - {k!r}  →  {type(data[k]).__name__}")


if __name__ == "__main__":
    main()