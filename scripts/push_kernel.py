"""Push a kernel to Kaggle, with or without running it.

`kaggle kernels push` always does Save & Run All, so even a comment fix burns
a full GPU run. The Kaggle API also has QUICK_SAVE (save without running),
which the CLI does not expose; this script uses it by default.

    # update the source only (no run, no GPU used)
    python scripts/push_kernel.py -p notebooks/infer

    # save and run
    python scripts/push_kernel.py -p notebooks/sweep --run
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")


def load_metadata(folder: pathlib.Path) -> dict:
    path = folder / "kernel-metadata.json"
    if not path.exists():
        sys.exit(f"{path} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_body(path: pathlib.Path) -> str:
    """Drop outputs and flatten each cell source into the single string the server expects."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
        if isinstance(cell.get("source"), list):
            cell["source"] = "".join(cell["source"])
    return json.dumps(nb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True, help="folder containing kernel-metadata.json")
    ap.add_argument("--run", action="store_true", help="run the kernel after saving")
    args = ap.parse_args()

    folder = pathlib.Path(args.path)
    meta = load_metadata(folder)
    code_file = folder / meta["code_file"]

    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

    api = KaggleApi()
    api.authenticate()

    owner, _name = meta["id"].split("/")

    request = ApiSaveKernelRequest()
    # The server wants the full "owner/slug"; the bare slug is rejected as invalid.
    request.slug = meta["id"]
    request.new_title = meta.get("title")
    request.text = notebook_body(code_file)
    request.language = meta["language"]
    request.kernel_type = meta["kernel_type"]
    request.is_private = str(meta.get("is_private", "true")).lower() == "true"
    request.enable_gpu = str(meta.get("enable_gpu", "false")).lower() == "true"
    request.enable_tpu = str(meta.get("enable_tpu", "false")).lower() == "true"
    request.enable_internet = str(meta.get("enable_internet", "false")).lower() == "true"
    request.dataset_data_sources = meta.get("dataset_sources", [])
    request.competition_data_sources = meta.get("competition_sources", [])
    request.kernel_data_sources = meta.get("kernel_sources", [])
    request.model_data_sources = meta.get("model_sources", [])
    request.category_ids = meta.get("keywords", [])
    if meta.get("machine_shape"):
        request.machine_shape = meta["machine_shape"]
    request.kernel_execution_type = (
        KernelExecutionType.SAVE_AND_RUN_ALL if args.run else KernelExecutionType.QUICK_SAVE
    )

    mode = "SAVE_AND_RUN_ALL" if args.run else "QUICK_SAVE (no run)"
    print(f"{meta['id']} <- {code_file}  [{mode}]")

    with api.build_kaggle_client() as kaggle:
        response = kaggle.kernels.kernels_api_client.save_kernel(request)

    print("version:", getattr(response, "version_number", "?"))
    print("url    :", getattr(response, "url", f"https://www.kaggle.com/code/{owner}/{_name}"))
    for err in (getattr(response, "error", None) or "",):
        if err:
            print("error  :", err)


if __name__ == "__main__":
    main()
