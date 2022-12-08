import gblvar

#gblvar variables updated in event_handlers are accessible here too

def opt_dummy():
    '''Placeholder for optimization stuff'''

    print(gblvar.vp.shape)
    return -1