from time import time


def timedmethod(precision):
    def timed_wrap(function):
        def wrapper(**kwargs):
            t1 = time()
            results = function(**kwargs)
            t2 = time()
            print(function.__name__ + " time: " +
                  str(format(t2 - t1, '.' + str(precision) + 'f')) +
                  " seconds")
            return results

        return wrapper

    return timed_wrap
