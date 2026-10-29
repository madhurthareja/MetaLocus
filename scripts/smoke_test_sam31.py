"""Day-1 smoke test: confirm SAM 3.1 Object Multiplex actually runs on the
DGX Spark (GB10, aarch64), and clock real FPS with 4-6 concurrent concept
sessions before locking the live-demo frame rate or concept count into the
architecture. See docs/PROJECT_PLAN.md sections 8 and 9.

Run from the 'hackathon' conda env after scripts/setup_env.sh and after
`hf auth login` has access to the gated facebook/sam3.1 checkpoint.
"""

import time

import torch


def main():
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))

    # TODO: load SAM 3.1 Object Multiplex via sam3.model_builder, run one native
    # session on one AEA frame, then repeat with 4-6 concurrent sessions and
    # measure wall-clock FPS. Record the result in docs/PROJECT_PLAN.md section 8.
    raise NotImplementedError("Fill in once the env + gated checkpoint are both ready.")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"done in {time.time() - start:.1f}s")
