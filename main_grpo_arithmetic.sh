#!/bin/bash
# Lightweight GRPO smoke test for RL-Factory without external RAG/tools.

set -e -x

export MODEL_PATH=${MODEL_PATH:-models/Qwen3-4B}
export DATA_DIR=${DATA_DIR:-data/arithmetic_toy}
export RESULT_DIR=${RESULT_DIR:-/data/chenyuxuan/RL-Factory-assets/logs/arithmetic_grpo}
export N_GPUS=${N_GPUS:-8}

python3 -m verl.trainer.main_ppo --config-name=rl_factory_ppo_trainer \
    algorithm.adv_estimator=grpo\
    data.train_files=$DATA_DIR/train.parquet\
    data.val_files=$DATA_DIR/test.parquet\
    data.train_batch_size=16\
    data.max_prompt_length=512\
    data.max_response_length=128\
    actor_rollout_ref.model.path=$MODEL_PATH\
    actor_rollout_ref.model.use_remove_padding=True\
    actor_rollout_ref.model.enable_gradient_checkpointing=True\
    actor_rollout_ref.actor.optim.lr=1e-6\
    actor_rollout_ref.actor.ppo_mini_batch_size=8\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1\
    actor_rollout_ref.actor.use_kl_loss=True\
    actor_rollout_ref.actor.kl_loss_coef=0.001\
    actor_rollout_ref.actor.kl_loss_type=low_var_kl\
    actor_rollout_ref.actor.fsdp_config.param_offload=True\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True\
    actor_rollout_ref.actor.state_masking=True\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1\
    actor_rollout_ref.rollout.name=vllm\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55\
    actor_rollout_ref.rollout.n=2\
    actor_rollout_ref.rollout.max_turns=1\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1\
    actor_rollout_ref.ref.fsdp_config.param_offload=False\
    actor_rollout_ref.rollout.enforce_eager=False\
    actor_rollout_ref.rollout.free_cache_engine=True\
    actor_rollout_ref.env.name=arithmetic\
    actor_rollout_ref.env.mcp_mode=stdio\
    actor_rollout_ref.env.tool_manager=qwen3\
    actor_rollout_ref.env.enable_thinking=False\
    actor_rollout_ref.env.config_path=envs/configs/no_tools.pydata\
    actor_rollout_ref.env.use_process_reward=False\
    reward_rollout.if_use_reward_rollout=False\
    reward_model.reward_manager=parallel\
    algorithm.kl_ctrl.kl_coef=0.001\
    trainer.critic_warmup=0\
    trainer.logger=['console','tensorboard']\
    trainer.project_name='GRPO_arithmetic_toy'\
    trainer.experiment_name='qwen3_4b_smoke'\
    trainer.n_gpus_per_node=$N_GPUS\
    trainer.nnodes=1\
    trainer.val_before_train=True\
    trainer.default_local_dir=$RESULT_DIR\
    trainer.default_hdfs_dir=null\
    trainer.save_freq=-1\
    trainer.test_freq=5\
    trainer.total_epochs=1 $@
