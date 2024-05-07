# 模型推理的入口

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.test import TESTERS
from pointcept.engines.launch import launch


def main_worker(cfg):
    cfg = default_setup(cfg)
    inferer = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    inferer.infer()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)
    
    # 非test模式
    print('inputpath:', args.inputpath)
    print('outputpath:', args.outputpath)
    print('knn:',args.knn)
    # 配置推理类
    cfg.test.type = ('TgnetInferer')
    cfg.data.test['infer_mode'] = True
    cfg.data.test.test_mode = True
    # 配置输入输出路径
    cfg.inputpath = args.inputpath
    cfg.outputpath = args.outputpath
    cfg.data_root = args.inputpath
    # 配置数据集路径
    cfg.data.train.data_root = cfg.data_root
    cfg.data.val.data_root = cfg.data_root
    cfg.data.test.data_root = cfg.data_root
    # 配置是否使用knn
    cfg.knn = args.knn
    
    
    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
