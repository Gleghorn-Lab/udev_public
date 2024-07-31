from .vit.config_vit import VitConfig
from .vit.modeling_vit import VitForSegmentation, VitForClassification
from .attalexnet.config_attalexnet import AttAlexNetConfig
from .attalexnet.modeling_attalexnet import AttAlexNetForClassification
from .pit.config_pit import PitConfig
from .pit.modeling_pit import PitForSegmentation, PitForClassification
from .ijepa.config_ijepa import IJEPAConfig
from .ijepa.modeling_ijepa import IJEPAForSegmentation, IJEPAForClassification
from .unet.config_unet import UnetConfig
from .unet.modeling_unet import (
    ACCForSegmentation,
    ACCForClassification,
    UNETForSegmentation,
    UNETForClassification,
    R2UForSegmentation,
    R2UForClassification,
    AttUForSegmentation,
    AttUForClassification,
    R2AttUForSegmentation,
    R2AttUForClassification,
)


def get_model(args, model_config):
    pretrained_config = None
    if args.segmentation:
        if args.model_type == 'ACC':
            config = UnetConfig(**model_config)
            model = ACCForSegmentation(config)
        elif args.model_type == 'UNET':
            config = UnetConfig(**model_config)
            model = UNETForSegmentation(config)
        elif args.model_type == 'R2UNET':
            config = UnetConfig(**model_config)
            model = R2UForSegmentation(config)
        elif args.model_type == 'ATTUNET':
            config = UnetConfig(**model_config)
            model = AttUForSegmentation(config)
        elif args.model_type == 'R2ATTUNET':
            config = UnetConfig(**model_config)
            model = R2AttUForSegmentation(config)
        elif args.model_type == 'Vit':
            config = VitConfig(**model_config)
            model = VitForSegmentation(config)
        elif args.model_type == 'Pit':
            config = PitConfig(**model_config)
            model = PitForSegmentation(config)
        elif args.model_type == 'JEPA':
            config = IJEPAConfig(**model_config)
            model = IJEPAForSegmentation(config)

    elif args.classification:
        if args.model_type == 'ACC':
            config = UnetConfig(**model_config)
            model = ACCForClassification(config)
        elif args.model_type == 'UNET':
            config = UnetConfig(**model_config)
            model = UNETForClassification(config)
        elif args.model_type == 'R2UNET':
            config = UnetConfig(**model_config)
            model = R2UForClassification(config)
        elif args.model_type == 'ATTUNET':
            config = UnetConfig(**model_config)
            model = AttUForClassification(config)
        elif args.model_type == 'R2ATTUNET':
            config = UnetConfig(**model_config)
            model = R2AttUForClassification(config)
        elif args.model_type == 'Vit':
            config = VitConfig(**model_config)
            model = VitForClassification(config)
        elif args.model_type == 'Pit':
            config = PitConfig(**model_config)
            model = PitForClassification(config)
        elif args.model_type == 'JEPA':
            config = IJEPAConfig(**model_config)
            model = IJEPAForClassification(config)
        elif args.model_type == 'AttAlexNet':
            config = AttAlexNetConfig(**model_config)
            model = AttAlexNetForClassification(config)
        
    return model
