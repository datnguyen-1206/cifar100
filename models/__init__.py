def get_model(model_name):
    if model_name == 'vgg16':
        from models.vgg import vgg16_bn
        model = vgg16_bn()
    elif model_name == 'vgg13':
        from models.vgg import vgg13_bn
        model = vgg13_bn()
    elif model_name == 'vgg11':
        from models.vgg import vgg11_bn
        model = vgg11_bn()
    elif model_name == 'vgg19':
        from models.vgg import vgg19_bn
        model = vgg19_bn()
    elif model_name == 'densenet121':
        from models.densenet import densenet121
        model = densenet121()
    elif model_name == 'densenet161':
        from models.densenet import densenet161
        model = densenet161()
    elif model_name == 'densenet169':
        from models.densenet import densenet169
        model = densenet169()
    elif model_name == 'densenet201':
        from models.densenet import densenet201
        model = densenet201()
    elif model_name == 'googlenet':
        from models.googlenet import googlenet
        model = googlenet()
    elif model_name == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        model = inceptionv3()
    elif model_name == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        model = inceptionv4()
    elif model_name == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        model = inception_resnet_v2()
    elif model_name == 'xception':
        from models.xception import xception
        model = xception()
    elif model_name == 'resnet18':
        from models.resnet import resnet18
        model = resnet18()
    elif model_name == 'resnet34':
        from models.resnet import resnet34
        model = resnet34()
    elif model_name == 'resnet50':
        from models.resnet import resnet50
        model = resnet50()
    elif model_name == 'resnet101':
        from models.resnet import resnet101
        model = resnet101()
    elif model_name == 'resnet152':
        from models.resnet import resnet152
        model = resnet152()
    elif model_name == 'preactresnet18':
        from models.preactresnet import preactresnet18
        model = preactresnet18()
    elif model_name == 'preactresnet34':
        from models.preactresnet import preactresnet34
        model = preactresnet34()
    elif model_name == 'preactresnet50':
        from models.preactresnet import preactresnet50
        model = preactresnet50()
    elif model_name == 'preactresnet101':
        from models.preactresnet import preactresnet101
        model = preactresnet101()
    elif model_name == 'preactresnet152':
        from models.preactresnet import preactresnet152
        model = preactresnet152()
    elif model_name == 'resnext50':
        from models.resnext import resnext50
        model = resnext50()
    elif model_name == 'resnext101':
        from models.resnext import resnext101
        model = resnext101()
    elif model_name == 'resnext152':
        from models.resnext import resnext152
        model = resnext152()
    elif model_name == 'shufflenet':
        from models.shufflenet import shufflenet
        model = shufflenet()
    elif model_name == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        model = shufflenetv2()
    elif model_name == 'squeezenet':
        from models.squeezenet import squeezenet
        model = squeezenet()
    elif model_name == 'mobilenet':
        from models.mobilenet import mobilenet
        model = mobilenet()
    elif model_name == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        model = mobilenetv2()
    elif model_name == 'nasnet':
        from models.nasnet import nasnet
        model = nasnet()
    elif model_name == 'attention56':
        from models.attention import attention56
        model = attention56()
    elif model_name == 'attention92':
        from models.attention import attention92
        model = attention92()
    elif model_name == 'seresnet18':
        from models.senet import seresnet18
        model = seresnet18()
    elif model_name == 'seresnet34':
        from models.senet import seresnet34
        model = seresnet34()
    elif model_name == 'seresnet50':
        from models.senet import seresnet50
        model = seresnet50()
    elif model_name == 'seresnet101':
        from models.senet import seresnet101
        model = seresnet101()
    elif model_name == 'seresnet152':
        from models.senet import seresnet152
        model = seresnet152()
    elif model_name == 'wideresnet':
        from models.wideresidual import wideresnet
        model = wideresnet()
    elif model_name == 'stochasticdepth18':
        from models.stochasticdepth import stochastic_depth_resnet18
        model = stochastic_depth_resnet18()
    elif model_name == 'stochasticdepth34':
        from models.stochasticdepth import stochastic_depth_resnet34
        model = stochastic_depth_resnet34()
    elif model_name == 'stochasticdepth50':
        from models.stochasticdepth import stochastic_depth_resnet50
        model = stochastic_depth_resnet50()
    elif model_name == 'stochasticdepth101':
        from models.stochasticdepth import stochastic_depth_resnet101
        model = stochastic_depth_resnet101()
    else:
        raise ValueError(f"No support for model {model_name}")
    return model