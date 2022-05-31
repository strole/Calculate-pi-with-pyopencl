import time

import pyopencl as cl
import numpy
import random


class PP:
    def __init__(self, path):
        self.memfag = cl.mem_flags
        self.context = cl.create_some_context(interactive=True)
        self.queue = cl.CommandQueue(self.context)
        self.code = "".join(open(path, 'r').readlines())
        self.program = cl.Program(self.context, self.code).build()

    def getQueue(self):
        return self.queue

    def getProgram(self):
        return self.program

    def getFlags(self):
        return self.memfag

    def getContext(self):
        return self.context


if __name__ == '__main__':
    start_time = time.time()
    p = PP("./pi.cl")

    n, G, L = numpy.int32(pow(2,21)), pow(2,21), pow(2,8)
    x = [random.uniform(-100, 100) for i in range(n)]

    y = [random.uniform(-100, 100) for i in range(n)]

    a = numpy.array(x, dtype=numpy.float32)
    b = numpy.array(y, dtype=numpy.float32)

    a_buf = cl.Buffer(p.getContext(), p.getFlags().READ_ONLY | p.getFlags().COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(p.getContext(), p.getFlags().READ_ONLY | p.getFlags().COPY_HOST_PTR, hostbuf=b)
    dest_buf = cl.Buffer(p.getContext(), p.getFlags().WRITE_ONLY, b.nbytes)

    p.getProgram().picalc(p.getQueue(), (G,), (L,), a_buf, b_buf, dest_buf, n).wait()
    c = numpy.empty_like(a)
    cl._enqueue_read_buffer(p.getQueue(), dest_buf, c).wait()

    number_in_circle = 0
    for i in c:
        number_in_circle = number_in_circle + i
    pi = number_in_circle * 4 / c.size
    print('pi = ', pi)
    print(time.time() - start_time)
