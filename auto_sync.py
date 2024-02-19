import os
from threading import Timer
from datetime import datetime
import argparse


def callbackFun(interval, path):
    os.system('python azsync.py -L %s' % path)
    global timer
    timer = Timer(interval, callbackFun, [interval, path])
    timer.start()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--interval', type=int, default=30, help='time interval default 30 min')
    parser.add_argument('--path', type=str)

    args = parser.parse_args()

    if args.path is None:
        raise ValueError('no path error!!!')

    time_sec = int(args.interval * 60)
    timer = Timer(1, callbackFun, [time_sec, args.path])
    timer.start()

