from torch import multiprocessing


class QueueProcess:
    def __init__(self, function, args):
        self._put = multiprocessing.Queue()
        self._get = multiprocessing.Queue()
        args = [self._put, self._get, *args]
        self.process = multiprocessing.Process(target=function, args=args)
        self.process.start()

    def close(self):
        self.put(StopIteration)
        self.process.join()
        del self._put
        del self._get

    def __iter__(self):
        return self

    def __next__(self):
        ret = self._get.get()
        if ret is StopIteration:
            raise ret()
        else:
            return ret

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def put(self, obj):
        return self._put.put(obj)

    def get(self):
        return self._get.get()
