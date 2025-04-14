import contextlib
import copy
import io

import torch


def find_nonleaf_tensors(obj, path="", visited=None):
    """
    引数:
    obj: 走査対象のオブジェクト（モデル、オプティマイザ、辞書、リストなど）
    path: オブジェクト内でのパス（デバッグ用に場所を表示）
    visited: すでに走査済みのオブジェクトIDのセット（一回のループで同じオブジェクトへの無限再帰を防ぐ）
    """
    if visited is None:
        visited = set()

    # 同じオブジェクトを複数回訪れないようにする
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    # torch.Tensorの場合：非リーフなら表示
    if isinstance(obj, torch.Tensor):
        if not obj.is_leaf:
            print(f"非リーフTensor発見 {path}: shape={tuple(obj.shape)}, requires_grad={obj.requires_grad}")
    # 辞書の場合：キーを文字列にして再帰
    elif isinstance(obj, dict):
        for key, value in obj.items():
            find_nonleaf_tensors(value, f"{path}[{repr(key)}]", visited)
    # リスト、タプル、セットなどイテラブルな場合
    elif isinstance(obj, (list, tuple, set)):
        for i, item in enumerate(obj):
            find_nonleaf_tensors(item, f"{path}[{i}]", visited)
    # __dict__を持つ任意のオブジェクトの場合
    elif hasattr(obj, "__dict__"):
        for attr, value in vars(obj).items():
            find_nonleaf_tensors(value, f"{path}.{attr}" if path else attr, visited)
    # その他の場合は無視
    else:
        pass
    
def find_deepcopy_issue(model : torch.nn.Module, prefix=''):
    try:
        copy.deepcopy(model)
    except Exception as e:
        print(f'Error in module: {prefix}')
        print(f'Error type: {type(e).__name__}')
        print(f'Error message: {str(e)}')
        return
    
    for name, child in model.named_children():
        find_deepcopy_issue(child, f'{prefix}.{name}' if prefix else name)


def collect_deepcopy_issues(obj, path="", visited=None, issues=None):
    """
    オブジェクト内部を再帰的に探索し、torch.Tensorでかつ
    is_leaf==False（非リーフテンソル）のものを検出し、(パス, 詳細情報)のタプルのリストとして返す関数です。
    """
    if visited is None:
        visited = set()
    if issues is None:
        issues = []

    obj_id = id(obj)
    if obj_id in visited:
        return issues
    visited.add(obj_id)

    if isinstance(obj, torch.Tensor):
        if not obj.is_leaf:
            issues.append(
                (path, f"shape={tuple(obj.shape)}, requires_grad={obj.requires_grad}, grad_fn={obj.grad_fn}, device={obj.device}")
            )
    elif isinstance(obj, dict):
        for key, value in obj.items():
            collect_deepcopy_issues(value, f"{path}[{repr(key)}]", visited, issues)
    elif isinstance(obj, (list, tuple, set)):
        for i, item in enumerate(obj):
            collect_deepcopy_issues(item, f"{path}[{i}]", visited, issues)
    elif hasattr(obj, "__dict__"):
        for attr, value in vars(obj).items():
            new_path = f"{path}.{attr}" if path else attr
            collect_deepcopy_issues(value, new_path, visited, issues)
    return issues


def list_all_tensors(obj, path="", visited=None):
    """
    オブジェクト内部を再帰的に探索し、すべてのtorch.Tensorの情報（shape, is_leaf, requires_grad, grad_fn, device）を標準出力に表示する関数です。
    """
    if visited is None:
        visited = set()
        obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if isinstance(obj, torch.Tensor):
        print(f"{path} : shape={tuple(obj.shape)}, is_leaf={obj.is_leaf}, requires_grad={obj.requires_grad}, grad_fn={obj.grad_fn}, device={obj.device}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            list_all_tensors(value, f"{path}[{repr(key)}]", visited)
    elif isinstance(obj, (list, tuple, set)):
        for i, item in enumerate(obj):
            list_all_tensors(item, f"{path}[{i}]", visited)
    elif hasattr(obj, "__dict__"):
        for attr, value in vars(obj).items():
            new_path = f"{path}.{attr}" if path else attr
            list_all_tensors(value, new_path, visited)

def save_debug_info(obj, filename="debug_info.txt"):
    """
    上記のlist_all_tensors()とcollect_deepcopy_issues()の出力結果をテキストとしてキャプチャし、
    指定ファイルに書き出す関数です。
    """
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        print("--- 全テンソル情報 ---")
        list_all_tensors(obj, "obj")
        print("\n--- deepcopyで問題となる可能性のあるテンソル情報 ---")
        issues = collect_deepcopy_issues(obj, "obj")
        if issues:
            for path, info in issues:
                print(f"Path: {path} -> {info}")
        else:
            print("問題となるテンソルは検出されませんでした。")
        content = output.getvalue()
        with open(filename, 'w', encoding="utf-8") as f:
            f.write(content)
        print(f"デバッグ情報が '{filename}' に書き出されました。")
        
if __name__ == "__main__":
    # モデルの例を作成
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 5),
    )
    
    # 非リーフテンソルを探す
    find_nonleaf_tensors(model)
    
    # deepcopyの問題を探す
    find_deepcopy_issue(model)
    
    print("全ての非リーフテンソルとdeepcopyの問題を確認しました。")
