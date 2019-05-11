def CoarseDropout(p=0, size_px=None, size_percent=None, per_channel=False, min_size=4, name=None, deterministic=False,
                  random_state=None):

    if size_px is not None:
        p3 = iap.FromLowerResolution(other_param=p2, size_px=size_px, min_size=min_size)
    elif size_percent is not None:
        p3 = iap.FromLowerResolution(other_param=p2, size_percent=size_percent, min_size=min_size)
    else:
        raise Exception("Either size_px or size_percent must be set.")

    if name is None:
        name = "Unnamed%s" % (ia.caller_name(),)

    return MultiplyElementwise(p3, per_channel=per_channel, name=name, deterministic=deterministic,
                               random_state=random_state)
