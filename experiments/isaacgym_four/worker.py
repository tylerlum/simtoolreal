"""IsaacGym campaign worker: native SAPG with explicit provenance and W&B."""
import argparse,json,os,sys,time,traceback,shutil
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--smoke-envs',type=int,default=0)
parser.add_argument('--smoke-epochs',type=int,default=4)
parser.add_argument('--resume',default='')
parser.add_argument('--smoke-capture',action='store_true')
a=parser.parse_args()
os.environ.pop('WANDB_SERVICE',None)
from isaacgym import gymapi,gymtorch
import torch,numpy as np
import wandb
from omegaconf import OmegaConf,open_dict
import isaacgymenvs
from isaacgymenvs.utils.rlgames_utils import RLGPUEnv,RLGPUAlgoObserver
from isaacgymenvs.utils.observation_action_utils_sharpa import JOINT_NAMES_ISAACGYM
from isaacgymenvs.utils.utils import set_seed
from rl_games.common import env_configurations,vecenv
from rl_games.torch_runner import Runner,_restore

cfg=OmegaConf.load(a.config)
smoke=bool(a.smoke_envs)
if smoke:
    assert a.smoke_envs%6==0
    cfg.task.env.numEnvs=a.smoke_envs
    cfg.train.params.config.num_actors=a.smoke_envs
    cfg.train.params.config.expl_coef_block_size=a.smoke_envs//6
    cfg.train.params.config.minibatch_size=min(a.smoke_envs*16,98304)
    cfg.train.params.config.central_value_config.minibatch_size=min(a.smoke_envs*16,98304)
    cfg.train.params.config.max_epochs=a.smoke_epochs
    cfg.campaign.run_dir=str(Path(cfg.campaign.run_dir).parent/'smokes'/(cfg.campaign.key+'_'+datetime.now().strftime('%Y%m%dT%H%M%S')))
    cfg.campaign.state_capture=a.smoke_capture
    cfg.train.params.config.train_dir=cfg.campaign.run_dir
    cfg.train.params.config.full_experiment_name='0_smoke_'+cfg.campaign.key
run_dir=Path(cfg.campaign.run_dir)
run_dir.mkdir(parents=True,exist_ok=True)
os.chdir(run_dir)
(run_dir/'worker.pid').write_text(str(os.getpid())+'\n')
(run_dir/'effective_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
set_seed(int(cfg.seed),torch_deterministic=False)
torch.set_num_threads(4)

class StateWindows:
    def __init__(self,env):
        self.env=env
        self.frames=1800
        self.count=min(4,env.num_envs)
        self.used=0
        self.active=False
        self.next_epoch=0
        self.epoch=0
        self.buffers=None
    def sample(self):
        epoch=self.env.total_train_env_frames//(self.env.num_envs*16)
        if not self.active:
            if epoch<self.next_epoch:return
            self.epoch=epoch
            self.active=True
            self.used=0
            if epoch<3000:self.next_epoch=(epoch//500+1)*500
            else:self.next_epoch=(epoch//1000+1)*1000
        e,n=self.env,self.count
        terms={'joint_pos':e.arm_hand_dof_pos[:n],'joint_target':e.cur_targets[:n,:29],
               'object_pose':e.root_state_tensor[e.object_indices[:n],:7],
               'goal_pose':e.goal_states[:n,:7],
               'table_pose':e.root_state_tensor[e.table_indices[:n],:7],
               'reset':e.reset_buf[:n]}
        if self.buffers is None:
            self.buffers={k:torch.empty((self.frames,*v.shape),dtype=v.dtype,device='cpu',pin_memory=True) for k,v in terms.items()}
        for k,v in terms.items(): self.buffers[k][self.used].copy_(v,non_blocking=True)
        self.used+=1
        if self.used==self.frames:self.flush()
    def flush(self):
        if not self.active or self.used==0:return
        torch.cuda.current_stream().synchronize()
        out=run_dir/'state_captures'/('epoch_%06d'%self.epoch);out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'states.npz',**{k:v[:self.used].numpy() for k,v in self.buffers.items()})
        manifest={'format':'simtoolreal_gym_state_window_v1','epoch':self.epoch,'num_frames':self.used,
            'control_dt':float(self.env.control_dt),'joint_names':list(JOINT_NAMES_ISAACGYM),
            'quaternion_order':'xyzw','pose_frame':'native_isaacgym_root_state',
            'object_scales':self.env.object_scales[:self.count].cpu().tolist(),
            'object_assets':[str(run_dir/'capture_assets'/Path(p).name) for p in self.env.object_asset_files[:self.count]],
            'env_origins':[[self.env.gym.get_env_origin(e).x,self.env.gym.get_env_origin(e).y,self.env.gym.get_env_origin(e).z] for e in self.env.envs[:self.count]],
            'robot_asset':str(ROOT/'assets'/self.env.robot_asset_file),
            'wandb_id':cfg.campaign.wandb_id}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('[capture] saved',out,flush=True)
        self.active=False

capture=None
def create_env(**kwargs):
    global capture
    env=isaacgymenvs.make(int(cfg.seed),'SimToolReal',int(cfg.task.env.numEnvs),'cuda:0','cuda:0',-1,True,False,False,False,cfg)
    if cfg.campaign.state_capture:
        capture=StateWindows(env)
        step=env.step
        def capture_step(actions):
            result=step(actions)
            capture.sample()
            return result
        env.step=capture_step
    return env

class Observer(RLGPUAlgoObserver):
    def before_init(self,base_name,config,experiment_name):
        if smoke:return
        r=wandb.init(entity=cfg.wandb_entity,project=cfg.wandb_project,group=cfg.wandb_group,
            id=cfg.campaign.wandb_id,name=cfg.campaign.key+'_s'+str(cfg.seed),
            resume='must' if a.resume else 'never',mode='online',dir=str(run_dir),
            config=OmegaConf.to_container(cfg,resolve=True),sync_tensorboard=True,
            tags=['isaacgym','expanded_limits',cfg.campaign.key,'finetune' if cfg.checkpoint else 'scratch'],
            settings=wandb.Settings(start_method='thread',init_timeout=120))
        if r is None or r.offline:raise RuntimeError('W&B online logging is required')
        wandb.define_metric('train/env_steps')
        wandb.define_metric('train/*',step_metric='train/env_steps')
        (run_dir/'wandb.json').write_text(json.dumps({'id':r.id,'url':r.url},indent=2)+'\n')
        print('[campaign] W&B ONLINE',r.url,flush=True)
    def after_init(self,algo):
        super().after_init(algo)
        env=algo.vec_env.env
        actor=env.gym.find_actor_handle(env.envs[0],'robot')
        names=env.gym.get_actor_dof_names(env.envs[0],actor)
        assert names==list(JOINT_NAMES_ISAACGYM),(names,JOINT_NAMES_ISAACGYM)
        props=env.gym.get_actor_dof_properties(env.envs[0],actor)
        assert env.num_obs==140 and env.num_states==162 and env.num_actions==29
        assert np.all(props['friction'][7:]>0),props['friction']
        assert np.isclose(props['upper'][names.index('left_index_MCP_AA')],.3491)
        assert np.isclose(props['upper'][names.index('left_thumb_CMC_AA')],.3491)
        assert algo.num_actors//algo.intr_coef_block_size==6
        if not smoke:assert algo.max_epochs==-1 and algo.max_frames==-1 and 'score_to_win' not in algo.config
        expected_eigen='eigendexplore' in cfg.campaign.key
        assert hasattr(algo.model.a2c_network,'noise_eigadd_basis')==expected_eigen
        eigen_shape=None
        if expected_eigen:
            net=algo.model.a2c_network
            eigen_shape=list(net.noise_eigadd_logsig.shape)
            assert eigen_shape==[6,net.noise_eigadd_basis.shape[0]],eigen_shape
        info={'joint_names':names,'joint_friction':props['friction'].tolist(),
              'joint_lower':props['lower'].tolist(),'joint_upper':props['upper'].tolist(),
              'obs_dim':env.num_obs,'state_dim':env.num_states,'action_dim':env.num_actions,
              'num_envs':env.num_envs,'sapg_blocks':6,'success_tolerance':float(env.success_tolerance),
              'expl_reward_coef_scale':float(algo.config['expl_reward_coef_scale']),
              'eigen_sigma_shape':eigen_shape,
              'max_epochs':algo.max_epochs,'max_frames':algo.max_frames,'smoke':smoke}
        (run_dir/'startup_contract.json').write_text(json.dumps(info,indent=2)+'\n')
        print('[campaign] STARTUP CONTRACT PASS',json.dumps(info),flush=True)
    def after_steps(self):
        if self.algo.epoch_num<=4:
            assert torch.isfinite(self.algo.obs['obs']).all()
            assert torch.isfinite(self.algo.obs['states']).all()
            assert torch.isfinite(self.algo.experience_buffer.tensor_dict['rewards']).all()
    def after_print_stats(self,frame,epoch_num,total_time):
        super().after_print_stats(frame,epoch_num,total_time)
        algo=self.algo
        env=algo.vec_env.env
        metrics={'train/env_steps':int(algo.frame),'train/epoch':int(epoch_num),
            'train/parent_plus_env_steps':int(cfg.campaign.parent_frame)+int(algo.frame),
            'train/success_tolerance':float(env.success_tolerance),'train/learning_rate':float(algo.last_lr),
            'train/elapsed_seconds':float(total_time),'train/peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/1024**3}
        if algo.game_rewards.current_size>0:metrics['train/episode_reward']=float(algo.game_rewards.get_mean()[0])
        metrics['train/mean_successes']=float(env.prev_episode_successes.mean())
        metrics['updated_utc']=datetime.now(timezone.utc).isoformat()
        tmp=run_dir/'progress.tmp';tmp.write_text(json.dumps(metrics,indent=2)+'\n');tmp.replace(run_dir/'progress.json')
        if not smoke:
            wandb.log({k:v for k,v in metrics.items() if k!='updated_utc'})
        if epoch_num==1:
            algo._save_rolling_checkpoint()
            algo.writer.flush()
        if epoch_num%10==0:algo.writer.flush()
        print('[campaign] PROGRESS',json.dumps(metrics),flush=True)

env_configurations.register('rlgpu',{'vecenv_type':'RLGPU','env_creator':create_env})
vecenv.register('RLGPU',lambda config_name,num_actors,**kwargs:RLGPUEnv(config_name,num_actors,**kwargs))
params=OmegaConf.to_container(cfg.train,resolve=True)
runner=Runner(Observer());runner.load(params);runner.set_vec_env(None);runner.reset()
status=1
try:
    # Construct the native agent, then explicitly initialize weights in Gym worlds.
    agent=runner.algo_factory.create(runner.algo_name,base_name='run',params=runner.params)
    runner.agent=agent
    checkpoint=a.resume or cfg.checkpoint
    if a.resume:
        saved=torch.load(checkpoint,map_location=agent.ppo_device,weights_only=False)
        saved=saved[0] if 0 in saved else saved
        env_state=saved.pop('env_state')
        # PhysX solver/contact caches cannot be restored from these tensors.
        # Resume training state and curriculum, starting clean Gym episodes.
        for key in ['rnn_states','dones','obs','current_rewards','current_shaped_rewards','current_lengths']:
            saved.pop(key,None)
        agent.set_full_state_weights(saved)
        env=agent.vec_env.env
        for key in ['success_tolerance','last_curriculum_update']:
            setattr(env,key,env_state[key])
        (run_dir/'resume_transfer.json').write_text(json.dumps({
            'checkpoint':checkpoint,'epoch':agent.epoch_num,'frame':agent.frame,
            'learning_rate':agent.last_lr,'success_tolerance':float(env.success_tolerance),
            'fresh_simulator_episodes':True,'optimizer_restored':True},indent=2)+'\n')
        del saved,env_state
        print('[campaign] RESUME TRAINING STATE; FRESH GYM EPISODES',flush=True)
    else:
        _restore(agent,{'checkpoint':checkpoint,'checkpoint_load_mode':'weights'})
    if checkpoint and not a.resume:
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for k,v in saved['model'].items():assert torch.equal(agent.model.state_dict()[k].detach().cpu(),v),k
        for k,v in saved['assymetric_vf_nets'].items():assert torch.equal(agent.central_value_net.state_dict()[k].detach().cpu(),v),k
        (run_dir/'weight_transfer.json').write_text(json.dumps({'actor_exact':True,'critic_exact':True,'optimizer_reset':True,'simulator_reset':True,'checkpoint':checkpoint,'initial_lr':agent.last_lr},indent=2)+'\n')
        del saved
        print('[campaign] EXACT ACTOR/CRITIC TRANSFER PASS',flush=True)
    (run_dir/'training_started.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'unix_time':time.time()},indent=2)+'\n')
    agent.train()
    status=0
finally:
    if hasattr(runner,'agent'):
        runner.agent.save_shutdown_checkpoint()
        runner.agent.writer.flush()
    if capture is not None:capture.flush()
    if wandb.run is not None:wandb.finish(exit_code=status)
    (run_dir/'exit_status.txt').write_text(str(status)+'\n')
