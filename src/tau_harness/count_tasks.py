import argparse

from dotenv import load_dotenv
from tau_bench.envs import get_env

from tau_harness.openai_user import install_user_patch
from tau_harness.run import configure_provider_env, env_flag, resolve_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Count available tau-bench tasks for one env/split.")
    parser.add_argument("--provider", choices=["openrouter", "openai"], required=True)
    parser.add_argument("--env", choices=["retail", "airline"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--user-model")
    parser.add_argument("--user-strategy", default="llm")
    parser.add_argument("--task-split", choices=["train", "test", "dev"], default="test")
    parser.add_argument("--enable-nvidia", action="store_true", default=env_flag("TAU_BENCH_ENABLE_NVIDIA"))
    args = parser.parse_args()

    load_dotenv()
    provider = configure_provider_env(args.provider, enable_nvidia=args.enable_nvidia)
    install_user_patch()
    env = get_env(
        args.env,
        user_strategy="human",
        user_model=resolve_model(provider, args.user_model or args.model),
        user_provider=provider,
        task_split=args.task_split,
    )
    print(len(env.tasks))


if __name__ == "__main__":
    main()
