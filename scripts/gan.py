
def load_gan_model(filename):
    import dcgan
    
    data = pickle.load(open(filename))
    gen = data['generator']
    discr = data['discriminator']
    gen_weights = data['generator_weights']
    discr_weights = data['discriminator_weights']
    
    layers.set_all_param_values(gen, gen_weights)
    layers.set_all_param_values(discr, discr_weights)
    return gen, discr

