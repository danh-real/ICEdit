trainable_model (train/src/train/train_route_rl.py) 
-> OminiModel (train/src/train/model.py) 
-> FluxFillPipeline (icedit/diffusers/pipelines/flux/pipeline_flux_fill.py)
-> FluxTransformer2DModel (icedit/diffusers/models/transformers/transformer_flux.py)
-> FluxTransformerBlock / FluxSingleTransformerBlock (icedit/diffusers/models/transformers/transformer_flux.py)
-> Linear (icedit/peft/tuners/lora/layer.py)

Clipped Surrogate Objective

ratio = pi / pi_old
min(r * advantage, clip(r, 1-eps, 1+eps) * advantage)

advantage = return - value

1st pass:
    save actions, rewards
