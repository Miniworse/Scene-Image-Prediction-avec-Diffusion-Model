import numpy as np

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def override(self, attrs):
        if isinstance(attrs, dict):
            self.__dict__.update(**attrs)
        elif isinstance(attrs, (list, tuple, set)):
            for attr in attrs:
                self.override(attr)
        elif attrs is not None:
            raise NotImplementedError
        return self

# Param settings

params_all = AttrDict(
    task_id=1,
    log_dir='./trainlog',
    model_dir='./model/Apr22-161504.pth',
    # data_dir='../data/data4debug',
    data_dir='../data/data6Mai/test',
    # test_dir='../data/data15Janv/TB',
    test_dir='../data/data6Mai/test',
    output_dir='./output',

    cond_dir=['./dataset/fmcw/cond'],
    fid_pred_dir = './dataset/fmcw/img_matric/pred',
    fid_data_dir = './dataset/fmcw/img_matric/data',
    # Training params
    load_pretrained = False,
    max_iter=None, # Unlimited number of iterations.
    batch_size=1,
    epochs=200,
    learning_rate=1e-3,
    max_grad_norm=None,
    train_ratio=0.95,
    test_ratio=0.04,
    val_ratio=0.01,
    seed=None, # 42
    num_workers=4,
    # Inference params
    inference_batch_size=1,
    robust_sampling=True,
    # Data params
    sample_rate=512,
    input_dim=128,
    extra_dim=[128],
    cond_dim=6,
    # Model params
    embed_dim=256,
    hidden_dim=256,
    num_heads=8,
    num_block=32,
    dropout=0.,
    mlp_ratio=4,
    learn_tfdiff=False,
    # Diffusion params
    signal_diffusion=True,
    max_step=100,
    # variance of the guassian blur applied on the spectrogram on each diffusion step [T]
    blur_schedule=((1e-5**2) * np.ones(100)).tolist(),
    # \beta_t, noise level added to the signal on each diffusion step [T]
    noise_schedule=np.linspace(1e-4, 0.003, 100).tolist(),
)