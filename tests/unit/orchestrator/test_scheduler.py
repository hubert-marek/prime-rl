import asyncio
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from prime_rl.orchestrator.scheduler import GroupState, InflightRolloutInfo, Scheduler
from prime_rl.utils.async_utils import safe_cancel


def make_scheduler() -> Scheduler:
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.max_async_level = 1
    scheduler.strict_async_level = False
    scheduler.step = 9
    scheduler.ckpt_step = 7
    scheduler.config = SimpleNamespace(output_dir=Path("/tmp/prime-rl-test"))
    scheduler.logger = MagicMock()
    scheduler.checkpoint_ready = asyncio.Event()
    scheduler.checkpoint_ready.set()
    scheduler.lora_name = None
    scheduler.model_name = "test-model"
    scheduler.update_weights_time = 0
    scheduler.wait_for_ckpt_time = 0
    scheduler.inflight_requests = {}
    scheduler.groups = {}
    scheduler.max_off_policy_steps = 1
    scheduler.cancelled_rollouts_count = 0
    scheduler.policy_update_lock = asyncio.Lock()
    scheduler.inflight_policy_update_task = None
    scheduler.update_policy_task = None
    scheduler.max_group_reschedules_by_task = {"test": 3}
    scheduler.forced_zero_rollouts_by_task = defaultdict(int)
    scheduler.forced_zero_groups_by_task = defaultdict(int)
    scheduler.dropped_groups_by_task = defaultdict(int)
    return scheduler


def test_update_off_policy_does_not_increment_interleaved_on_policy_tasks():
    async def run() -> None:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.max_off_policy_steps = 1
        scheduler.cancelled_rollouts_count = 0
        scheduler.logger = MagicMock()

        client = SimpleNamespace(api_base_url="http://test")
        stale_task = asyncio.create_task(asyncio.sleep(60))
        survivor_task = asyncio.create_task(asyncio.sleep(60))
        interleaved_task = None

        scheduler.inflight_requests = {
            stale_task: InflightRolloutInfo(off_policy_steps=1, client_config=client, task="test", group_id=1),
            survivor_task: InflightRolloutInfo(off_policy_steps=0, client_config=client, task="test", group_id=2),
        }

        async def drop_group(group_id: int) -> int:
            tasks_to_remove = [
                task for task, info in list(scheduler.inflight_requests.items()) if info.group_id == group_id
            ]
            for task in tasks_to_remove:
                scheduler.inflight_requests.pop(task, None)
                task.cancel()

            await asyncio.sleep(0)

            nonlocal interleaved_task
            if interleaved_task is None:
                interleaved_task = asyncio.create_task(asyncio.sleep(60))
                scheduler.inflight_requests[interleaved_task] = InflightRolloutInfo(
                    off_policy_steps=0,
                    client_config=client,
                    task="test",
                    group_id=3,
                )
            return len(tasks_to_remove)

        scheduler.drop_group = drop_group

        await scheduler._update_off_policy()

        assert stale_task not in scheduler.inflight_requests
        assert scheduler.inflight_requests[survivor_task].off_policy_steps == 1
        assert interleaved_task is not None
        assert scheduler.inflight_requests[interleaved_task].off_policy_steps == 0
        assert scheduler.cancelled_rollouts_count == 1

        for task in (stale_task, survivor_task, interleaved_task):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_maybe_update_policy_reuses_inflight_update_after_cancellation():
    async def run() -> None:
        scheduler = make_scheduler()
        started = asyncio.Event()
        release = asyncio.Event()
        applied_steps: list[int] = []

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            applied_steps.append(step)
            started.set()
            await release.wait()

        scheduler.inference_pool = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            first = asyncio.create_task(scheduler.maybe_update_policy())
            await started.wait()
            await safe_cancel(first)

            second = asyncio.create_task(scheduler.maybe_update_policy())
            await asyncio.sleep(0)
            assert applied_steps == [8]

            release.set()
            await second

        assert applied_steps == [8]
        assert scheduler.ckpt_step == 8

    asyncio.run(run())


def test_stop_cancels_inflight_policy_update_task():
    async def run() -> None:
        scheduler = make_scheduler()
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def update_weights(weight_dir, lora_name=None, step=0) -> None:
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        scheduler.inference_pool = SimpleNamespace(
            update_weights=update_weights,
            update_model_name=MagicMock(),
        )
        scheduler._update_off_policy = AsyncMock()

        with (
            patch("prime_rl.orchestrator.scheduler.get_latest_ckpt_step", return_value=8),
            patch("prime_rl.orchestrator.scheduler.wait_for_path", new=AsyncMock()),
        ):
            scheduler.update_policy_task = asyncio.create_task(scheduler.maybe_update_policy())
            await started.wait()
            await asyncio.wait_for(scheduler.stop(), timeout=0.2)

        assert cancelled.is_set()
        assert scheduler.update_policy_task is None
        assert scheduler.inflight_policy_update_task is None

    asyncio.run(run())


def test_force_zero_reward_rollout_clears_error_and_keeps_tokens():
    rollout = {
        "trajectory": [
            {
                "tokens": {
                    "prompt_ids": [1, 2],
                    "prompt_mask": [1, 1],
                    "completion_ids": [3, 4],
                    "completion_mask": [1, 1],
                    "completion_logprobs": [-1.0, -1.0],
                }
            }
        ],
        "reward": 1.0,
        "error": {"error_chain_repr": "BadRequestError('bad json')"},
        "stop_condition": None,
        "metrics": {},
    }

    forced = Scheduler._force_zero_reward_rollout(rollout, "bad json")

    assert forced["reward"] == 0.0
    assert forced["error"] is None
    assert forced["stop_condition"] == "forced_zero_reward_failure"
    assert forced["_prime_rl_forced_zero_reward"] is True
    assert forced["metrics"]["forced_zero_reward_failure"] == 1.0
    assert forced["trajectory"][0]["tokens"]["completion_ids"] == [3, 4]


def test_rollout_has_generated_tokens_detects_completion_ids():
    rollout = {
        "trajectory": [
            {"tokens": {"completion_ids": []}},
            {"tokens": {"completion_ids": [1]}},
        ]
    }
    assert Scheduler._rollout_has_generated_tokens(rollout) is True
    assert Scheduler._rollout_has_generated_tokens({"trajectory": [{"tokens": {"completion_ids": []}}]}) is False


def test_capped_group_with_generated_tokens_becomes_forced_zero_and_completes():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.rollouts_per_example = 1
        scheduler.batch_size = 1
        scheduler.token_batch_size = None
        scheduler.inference_pool = SimpleNamespace(get_metrics=lambda: {})
        scheduler.buffer = MagicMock()
        scheduler.buffer.sample_examples = MagicMock(return_value=[{"task": "test"}])
        scheduler.buffer.sample_rollouts = MagicMock(side_effect=lambda n: scheduler.buffer.update.call_args[0][0])
        scheduler.env = MagicMock()
        scheduler._score_group_if_deferred = AsyncMock(side_effect=lambda xs: xs)
        scheduler._fill_inflight_requests = AsyncMock()
        scheduler.checkpoint_ready = asyncio.Event()
        scheduler.checkpoint_ready.set()
        scheduler.json_logging = False
        scheduler.step = 0
        scheduler.ckpt_step = 0

        group = GroupState(example={"task": "test"}, rollouts_to_schedule=0, reschedule_count=3)
        scheduler.groups = {1: group}

        rollout = {
            "task": "test",
            "trajectory": [
                {
                    "tokens": {
                        "prompt_ids": [1],
                        "prompt_mask": [1],
                        "completion_ids": [2],
                        "completion_mask": [1],
                        "completion_logprobs": [-1.0],
                    }
                }
            ],
            "reward": 1.0,
            "error": {"error_chain_repr": "BadRequestError('bad json')"},
            "stop_condition": None,
            "metrics": {},
        }
        task = asyncio.get_running_loop().create_future()
        task.set_result(rollout)
        scheduler.inflight_requests = {task: InflightRolloutInfo(0, SimpleNamespace(), "test", 1)}

        with patch("prime_rl.orchestrator.scheduler.asyncio.wait", new=AsyncMock(return_value=({task}, set()))):
            batch = await scheduler.generate_batch(step=0)

        assert len(batch) == 1
        assert batch[0]["reward"] == 0.0
        assert batch[0]["error"] is None
        assert batch[0]["_prime_rl_forced_zero_reward"] is True
        assert scheduler.buffer.update.call_count == 1

    asyncio.run(run())


def test_capped_group_without_generated_tokens_is_dropped():
    async def run() -> None:
        scheduler = make_scheduler()
        scheduler.rollouts_per_example = 1
        scheduler.batch_size = 1
        scheduler.token_batch_size = None
        scheduler.inference_pool = SimpleNamespace(get_metrics=lambda: {})
        scheduler.buffer = MagicMock()
        scheduler.buffer.sample_examples = MagicMock(return_value=[{"task": "test"}])
        scheduler.buffer.sample_rollouts = MagicMock(return_value=[])
        scheduler.env = MagicMock()
        scheduler._score_group_if_deferred = AsyncMock(side_effect=lambda xs: xs)
        scheduler._fill_inflight_requests = AsyncMock(side_effect=[None, asyncio.CancelledError()])
        scheduler.checkpoint_ready = asyncio.Event()
        scheduler.checkpoint_ready.set()
        scheduler.json_logging = False
        scheduler.step = 0
        scheduler.ckpt_step = 0

        group = GroupState(example={"task": "test"}, rollouts_to_schedule=0, reschedule_count=3)
        scheduler.groups = {1: group}

        rollout = {
            "task": "test",
            "trajectory": [],
            "reward": 1.0,
            "error": {"error_chain_repr": "BadRequestError('bad json')"},
            "stop_condition": None,
            "metrics": {},
        }
        task = asyncio.get_running_loop().create_future()
        task.set_result(rollout)
        scheduler.inflight_requests = {task: InflightRolloutInfo(0, SimpleNamespace(), "test", 1)}
        scheduler.drop_group = AsyncMock(return_value=1)

        with patch("prime_rl.orchestrator.scheduler.asyncio.wait", new=AsyncMock(return_value=({task}, set()))):
            try:
                await scheduler.generate_batch(step=0)
            except asyncio.CancelledError:
                pass

        scheduler.drop_group.assert_awaited_with(1)
        assert scheduler.buffer.update.call_count == 0

    asyncio.run(run())
