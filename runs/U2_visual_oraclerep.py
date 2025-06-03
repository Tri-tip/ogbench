import os
import random


def main():
    preset = 'T'
    if preset == 'P':
        num_tasks_per_gpu = 4
        num_cpus_per_task = 2

        gpu_list = [0, 1, 2, 3, 4, 5, 6, 7]  # 0-based
        set_cpu_idx = False
        start_cpu_idx = 1  # 1-based
        exclude_cpus = []  # 1-based
        seeds = None

        pre_command = 'MUJOCO_GL=egl WANDB__SERVICE_WAIT=86400 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 WANDB_API_KEY=<KEY HERE>'
        python_command = 'python3 main.py'
        conda_command = 'conda activate scalerl'
    else:
        num_job_group = 1
        sh_command = './run.sh'
        pre_sbatch_command = 'MUJOCO_GL=egl WANDB__SERVICE_WAIT=86400 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 WANDB_API_KEY=<KEY HERE>'
        if preset == 'T':
            num_groups = 4
            num_cpus = 1
            sbatch_command = f'-A co_rail -p savio4_gpu --gres=gpu:A5000:1 -N 1 -n {num_groups} -c {num_cpus} --qos=rail_gpu4_normal -t 10-00:00:00 --mem=60G'
        elif preset == 'TL':
            num_groups = 4
            sbatch_command = f'-A co_rail -p savio3_gpu --gres=gpu:TITAN:1 -N 1 -n {num_groups} -c 1 --qos=savio_lowprio -t 3-00:00:00'
        elif preset == 'G':
            num_groups = 2
            sbatch_command = f'-A co_rail -p savio3_gpu --gres=gpu:GTX2080TI:1 -N 1 -n {num_groups} -c 1 --mem 12G --qos=savio_lowprio -t 3-00:00:00'
        elif preset == 'A':
            num_groups = 8
            sbatch_command = f'-A co_rail -p savio3_gpu --gres=gpu:A40:1 -N 1 -n {num_groups} -c 1 --qos=savio_lowprio -t 3-00:00:00'
        else:
            raise NotImplementedError
        python_command = 'python3 main.py'

    run_group = os.path.splitext(os.path.basename(__file__))[0]

    print(run_group)

    default_args = dict(
        run_group=run_group,
        eval_interval=100000,
        save_interval=10000000,
        log_interval=10000,
        eval_episodes=20,
        video_episodes=1,
    )

    tests = []
    group_num = int(run_group[1:].split('_')[0])
    seed = group_num * 10000
    print('seed', seed)

    for env_name, dataset_dir in [
        ('antmaze-large-navigate-v0', None),
        ('humanoidmaze-large-navigate-v0', None),
        ('cube-double-play-v0', None),
        ('puzzle-3x3-play-v0', None),
    ]:
        for offline_steps in [1000000]:
            for agent in ['agents/gcsacbc.py']:
                for actor_geom_sample, actor_p_trajgoal in [(False, 1.0)]:
                    for value_geom_sample in [True]:
                        for discount in [0.995] if 'giant' in env_name or 'humanoid' in env_name else [0.99]:
                            for alpha in [0.01, 0.03, 0.1, 0.3] if 'navigate' in env_name else [0.3, 1, 3, 10]:
                                for hidden_dims in ['"(512, 512, 512)"']:
                                    for value_loss_type, gc_negative in [('bce', False)]:
                                        for q_agg in ['mean'] if 'navigate' in env_name else ['min']:
                                            for value_type in ['monolithic', 'bilinear']:
                                                for i in range(4):
                                                    seed += 1
                                                    base_dict = dict(
                                                        default_args,
                                                    )
                                                    tests.append(dict(
                                                        base_dict,
                                                        seed=seed,
                                                        env_name=env_name,
                                                        dataset_dir=dataset_dir,
                                                        dataset_replace_interval=1000,
                                                        offline_steps=offline_steps,
                                                        agent=agent,
                                                        agentIbatch_size=1024,
                                                        agentIactor_layer_norm=False,  # For the compatibility with OGBench.
                                                        agentIactor_hidden_dims=hidden_dims,
                                                        agentIvalue_hidden_dims=hidden_dims,
                                                        agentIdiscount=discount,
                                                        agentIactor_p_trajgoal=actor_p_trajgoal,
                                                        agentIactor_p_randomgoal=1 - actor_p_trajgoal,
                                                        agentIactor_geom_sample=actor_geom_sample,
                                                        agentIvalue_geom_sample=value_geom_sample,
                                                        agentIalpha=alpha,
                                                        agentIq_agg=q_agg,
                                                        agentIvalue_loss_type=value_loss_type,
                                                        agentIgc_negative=gc_negative,
                                                        agentIvalue_type=value_type,
                                                    ))

    print(len(tests))

    test_commands = []
    for test in tests:
        test_command = ''
        for k, v in test.items():
            if v is None:
                continue
            test_command += f' --{k.replace("I", ".")}={v}'
        test_commands.append(test_command)

    if preset == 'P':
        if seeds is not None:
            test_commands = [test_commands[i] for i in seeds]
            print(len(test_commands))

        contents = []
        contents.append(f'tmux new-window -d -n {run_group}')
        for i in range(len(test_commands)):
            contents.append(f'tmux split -t ":{run_group}" -h')
            contents.append(f'tmux select-layout -t ":{run_group}" tiled')
        current_cpu_idx = start_cpu_idx - 1
        pseudo_slurm_job_id = random.randint(100000, 999999)
        for i, test_command in enumerate(test_commands):
            gpu_idx = gpu_list[i // num_tasks_per_gpu]
            cpu_idxs = []
            while len(cpu_idxs) < num_cpus_per_task:
                if current_cpu_idx + 1 not in exclude_cpus:
                    cpu_idxs.append(str(current_cpu_idx))
                current_cpu_idx += 1
            cpu_idxs = ','.join(cpu_idxs)

            cpu_command = f"taskset -c {cpu_idxs} " if set_cpu_idx else ''
            command = f"{pre_command} CUDA_VISIBLE_DEVICES={gpu_idx} SLURM_JOB_ID={pseudo_slurm_job_id} {cpu_command}{python_command}{test_command}"
            contents.append(f"tmux send-keys -t ':{run_group}.{i}' '{conda_command}' Enter")
            contents.append(f"tmux send-keys -t ':{run_group}.{i}' '{command}' Enter")
        contents.append(f'tmux send-keys -t ":{run_group}.{len(test_commands)}" "cd logs/{run_group}" Enter')
        with open('../sbatch.sh', 'w') as f:
            f.write('\n'.join(contents))
    else:
        contents = []
        content = ''
        target_remainder = num_groups - 1
        for i, test_command in enumerate(test_commands):
            if i % num_groups == 0:
                content += f'{pre_sbatch_command} sbatch {sbatch_command} --parsable --comment="{run_group}.{i // num_groups}" {sh_command}'
                if i + num_groups >= len(test_commands):
                    target_remainder = len(test_commands) - i - 1
            content += f" '{python_command}{test_command}'"
            if i % num_groups == target_remainder:
                contents.append(content)
                content = ''
        if num_job_group is not None:
            for i, content in enumerate(contents):
                contents[i] = f'jobid{i}=$({content}) && echo $jobid{i}'
            for i, content in enumerate(contents):
                if i % num_job_group != 0:
                    cur = content.split('sbatch')
                    cur[1] = f' --dependency=afterany:$jobid{i - 1}' + cur[1]
                    contents[i] = 'sbatch'.join(cur)
        with open('sbatch.sh', 'w') as f:
            f.write('\n'.join(contents))


if __name__ == '__main__':
    main()
