
config = {

    # atc
    "n_aircraft": 3,
    "speed": 10,
    "tt": 0.025,
    "step_dist": 5 * 0.025, # speed * tt
    "wind_direction": 0,
    "wind_speed": 0,
    # minimum separation radius around each aircraft
    "sep_radius": 0.2,

    # data generation parameters
    # temperature scaling for heading sampling
    "temperature": 5,
    # bias towards continuing present heading
    "bias": 0.5,

    # model 

    # vocab_size = number of possible headings, plus ss token and pad token
    "vocab_size": 38,
    
    # embedding dimensions
    "n_embed": 256,
    # number of attention heads
    "n_head": 8,
    "num_kv_groups": 4,
    "qk_norm": True,
    "rope_base": 1_000_000,
    # layers in the network
    "n_layer": 8,
    # dimension of hidden layers in network
    "hidden_dim": 4 * 256, # 4*n_embed
    # dropout rate
    "drop_rate": 0.1,

    # token id for the sequence padding
    "pad_token_id": 37,

    # training

    #sequences to process in each batch
    "batch_size": 64,
    # max context length
    "context_length": 32,
    # steps between loss evaluation
    "eval_interval": 200,
    # max training steps
    "max_iters": 100_000,
    "eval_iters": 200,

    # initial and peak learning rates
    "initial_lr": 1e-4,
    "peak_lr": 3e-3,

    "save_interval": 5_000,
    "save_dir": "checkpoints",

    # device
    "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",

}