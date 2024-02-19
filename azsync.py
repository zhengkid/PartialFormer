import os
import argparse
from pathlib import Path
parser = argparse.ArgumentParser(description='azsync')

parser.add_argument('-O','--root', type=str, default='/scratch/blobfuse/f_xdata/', help='Local Root Dir')
parser.add_argument('-R','--remote_path', type=str, help='Remote Path')
parser.add_argument('-L','--local_path', type=str,help='Local Path')
args = parser.parse_args()
local_rootdir=os.path.abspath(args.root)
if not os.path.exists(local_rootdir):
    raise ValueError(f"Please manually ensure that local_rootdir {local_rootdir} exist and has permissions.")
remote_rootdir='https://chenfei.blob.core.windows.net/data/'
keys="?sv=2020-10-02&st=2022-09-25T04%3A34%3A31Z&se=2024-09-26T04%3A34%3A00Z&sr=c&sp=racwdxl&sig=qhi5AaczGeVZnUnaBe%2F0jpes6LYqMlAqoqOHhH7TkuE%3D"

if (not args.remote_path and not args.local_path) or (args.remote_path and args.local_path) :
    raise ValueError("You must specify either remote dir or local dir."
                     "The other one will be calculated automatically.")
if args.remote_path:
    if args.remote_path.startswith('https:'):
        target_remote_path = args.remote_path
    else:
        target_remote_path= os.path.join(remote_rootdir, args.remote_path)
    relative_path = target_remote_path.replace(remote_rootdir, "")
    target_local_path = os.path.join(local_rootdir, relative_path)
    if Path(target_local_path).suffix:
        print('Detected file transfer R->L')
        method = "cp"
        if not os.path.exists(target_local_path):
            print('target_local_path: %s' % target_local_path)
            os.system(f'mkdir -p {os.path.basename(target_local_path)}')
    else:
        print('Detected dir transfer R->L.')
        method = "sync"
        if not os.path.exists(target_local_path):
            os.system(f'mkdir -p {target_local_path}')
    cmd = f'azcopy {method} {target_remote_path}"{keys}" "{target_local_path}"'
    print(f'Running Command:\n {cmd}')
    os.system(cmd)

if args.local_path:
    target_local_path = os.path.abspath(args.local_path)
    relative_path = os.path.relpath(target_local_path, local_rootdir)
    target_remote_path = os.path.join(remote_rootdir, relative_path)
    if Path(target_local_path).suffix:
        print('Detected file transfer L->R.')
        method = "copy"
    else:
        print('Detected dir transfer L->R.')
        method = "sync"
    cmd = f'azcopy {method} {target_local_path} {target_remote_path}"{keys}"'
    print(f'Running Command:\n {cmd}')
    os.system(cmd)

'''


azcopy copy /workspace/taming-transformers/logs/2021-07-31T20-23-25_faceshq_vqgan ${AVIP}/data/G/vqgan/official/2021-07-31T20-23-25_faceshq_vqgan${SVIP} --recursive
azcopy copy /workspace/taming-transformers/logs ${AVIP}/data/G/vqgan/${SVIP} --recursive
./azcopy copy ${AVIP}/data/imagenet${SVIP}  ${AVIP}/data/G/vqgan/data/ILSVRC2012_train/ILSVRC2012_img_train.tar${SVIP}


azcopy copy /workspace/vatex_videos.txt ${AVIP}/data/G/dataset/vatex/vatex_videos.txt${SVIP} --recursive
azcopy copy /workspace/pexels ${AVIP}/data/G/dataset/${SVIP} --recursive
azcopy copy /workspace/vspw ${AVIP}/data/G/dataset/${SVIP} --recursive


azcopy copy /home/chewu/ffhq-dataset/ffhq ${AVIP}/data/G/dataset/${SVIP} --recursive

azcopy copy /workspace/pexels/sky_v2 ${AVIP}/data/G/dataset/pexels/${SVIP} --recursive
azcopy copy /workspace/pexels/sky_v2.json ${AVIP}/data/G/dataset/pexels/${SVIP}
azcopy copy /workspace/ccvs/datasets/bairhd/softmotion_0511.tar.gz  ${AVIP}/data/G/dataset/bair/softmotion_0511.tar.gz${SVIP}
azcopy copy /workspace/GODIVA/cocostuff/dataset/annotations/train2017/* ${AVIP}/data/G/dataset/mscoco/train2017_stuff${SVIP} --recursive
azcopy copy /workspace/GODIVA/cocostuff/dataset/annotations/val2017 ${AVIP}/data/G/dataset/mscoco/val2017_stuff${SVIP} --recursive

azcopy copy /workspace/flintstones ${AVIP}/data/G/dataset/flintstones/${SVIP} --recursive
azcopy copy /workspace/kinetics ${AVIP}/data/G/dataset/kinetics/${SVIP} --recursive
'''