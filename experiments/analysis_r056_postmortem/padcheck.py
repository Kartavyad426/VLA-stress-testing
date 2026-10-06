import os,sys,torch
sys.path.insert(0,os.getcwd())
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
from torch.utils.data import default_collate
CKPT="nvidia/gr00t17-lerobot-libero_spatial-640"
cfg=PreTrainedConfig.from_pretrained(CKPT); cfg.base_model_path="nvidia/GR00T-N1.7-3B"; cfg.embodiment_tag="libero_sim"; cfg.device="cpu"
print('chunk_size',cfg.chunk_size,'n_action_steps',cfg.n_action_steps,'action_delta_indices',len(cfg.action_delta_indices))
meta=LeRobotDatasetMetadata("local/r054_train48",root="data/r054_train48")
delta=resolve_delta_timestamps(cfg,meta,{})
print('delta keys',{k:len(v) for k,v in delta.items()})
ds=LeRobotDataset("local/r054_train48",root="data/r054_train48",delta_timestamps=delta,video_backend="pyav",return_uint8=True,episodes=[0])
pre,_=make_pre_post_processors(policy_cfg=cfg,pretrained_path=CKPT,preprocessor_overrides={"device_processor":{"device":"cpu"}})
for i in [0,15,32,40,47]:
    it=ds[i]
    print(i,'task',repr(it['task'])[:60],'is_pad sum',int(it['action_is_pad'].sum()),'action shape',tuple(it['action'].shape))
    b=_preprocess_dataset_batch(default_collate([it]),ds.meta.camera_keys,{},pre)
    am=b.get('action_mask')
    if am is None:
        for k in b: 
            if 'mask' in str(k): print(' key',k)
    else:
        print('   action_mask shape',tuple(am.shape),'valid horizon steps',int((am[0].sum(-1)>0).sum()))
