def running_mean(new_point, running_mean, i):
    return running_mean + (new_point - running_mean)/(i + 1)