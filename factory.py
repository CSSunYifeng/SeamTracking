import models.tasks.ori

def generateModel(opt:dict):
    net = None
    if(opt['model']=='omnipose_mod_4stage'):
        net = models.tasks.ori.Omnipose_mod_mult("4stage")
    elif(opt['model']=='omnipose_mod_3stage'):
        net = models.tasks.ori.Omnipose_mod_mult("3stage")
    elif(opt['model']=='omnipose_mod_2stage'):
        net = models.tasks.ori.Omnipose_mod_mult("2stage")
    elif(opt['model']=='omnipose_mod_2stage_32'):
        net = models.tasks.ori.Omnipose_mod_mult("2stage_32")
    elif(opt['model']=='omnipose_mod_2stage_no_gauss'):
        net = models.tasks.ori.Omnipose_mod_mult("2stage",with_gauss_filter=False)
    elif(opt['model']=='omnipose_mod_1stage'):
        net = models.tasks.ori.Omnipose_mod_mult("1stage")
    elif(opt['model']=='omnipose_lite'):
        net = models.tasks.ori.Omnipose_mult("lite")# "hrnet_2tage_double"
    elif(opt['model']=='hrnet_mod_2stage'):
        net = models.tasks.ori.HRNet("2stage")
    elif(opt['model']=='hrnet_mod_4stage'):
        net = models.tasks.ori.HRNet("4stage")
    elif(opt['model']=='hrnet_2tage_double'):
        net = models.tasks.ori.HRNet("2stage_double")
    elif(opt['model']=='hrnet_mod_1stage'):
        net = models.tasks.ori.HRNet("1stage")
    elif(opt['model']=='cn_resnet18_decoder'):
        net = models.tasks.ori.MyModelResNet_head_MB_mult_innerdecoder(18)
    elif(opt['model']=='cn_resnet50_decoder'):
        net = models.tasks.ori.MyModelResNet_head_MB_mult_innerdecoder(50)
    return net