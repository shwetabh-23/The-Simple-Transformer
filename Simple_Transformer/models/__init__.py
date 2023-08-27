from .transformer import Transformer

def make_model(name, **kwargs):

    model_class = eval(name)
    model = model_class(**kwargs)
    return model
