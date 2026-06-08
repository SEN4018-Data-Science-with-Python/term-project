from __future__ import annotations
from e2b_code_interpreter import Sandbox as E2BSandbox


class Sandbox:
    def __init__(self) -> None:
        # E2BSandbox() constructor is deprecated in e2b-code-interpreter >=1.0;
        # use the Sandbox.create() classmethod instead.
        self._sbx = E2BSandbox.create()
        self._dataset_remote_path: str | None = None

    def upload(self, local_path: str, remote_name: str = "data.csv") -> str:
        with open(local_path, "rb") as f:
            self._sbx.files.write(f"/home/user/{remote_name}", f.read())
        self._dataset_remote_path = f"/home/user/{remote_name}"
        return self._dataset_remote_path

    def run(self, code: str) -> dict:
        exec_result = self._sbx.run_code(code)
        stdout = "\n".join(exec_result.logs.stdout)
        stderr = "\n".join(exec_result.logs.stderr)
        if exec_result.error:
            stderr += f"\n{exec_result.error.name}: {exec_result.error.value}"
        return {"stdout": stdout, "stderr": stderr}

    @property
    def dataset_path(self) -> str:
        if not self._dataset_remote_path:
            raise RuntimeError("No dataset uploaded")
        return self._dataset_remote_path

    def close(self) -> None:
        self._sbx.kill()
